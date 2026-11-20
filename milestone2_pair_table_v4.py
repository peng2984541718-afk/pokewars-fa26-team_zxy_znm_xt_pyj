"""Milestone 2: 6-fold 100%-coverage ensemble of PairMLPWide + PairBilinearResidual
trained with exact Binomial BCE loss and Gen-2 high-confidence pseudo-battles.

Protocol:
1. Train Fold-0-clean `pair_v4_nofold0_table.npy` strictly excluding all Fold-0/Fold-0
   labeled battles and using Fold-0-clean unlabeled pseudo-battles.
2. Evaluate oracle RMSE on Fold-0 battles + full end-to-end Fold-0 Cavity score across
   table blend weights `w in {0.0, 0.25, 0.4, 0.5, 0.65, 0.8, 1.0}`, picking `(w, scale)`
   strictly on the Fold-0 `tune` half and verifying on the untouched `test` half.
3. If `w > 0` passes the Fold-0 `test` gate, train full 6-fold `pair_v4_full_table.npy`
   on 100% of labeled battles + Gen-2 pseudo-battles, blend with
   `pair_semi_v23_blend_table.npy`, and write `submission_rx_v2_table.csv`.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from evaluate_soft_score import DENOMINATOR, rmse
from image_experiment import SEED, stratified_folds
from joint_constrained_inference import (
    power_scale_posteriors,
    run_submission,
    run_unlabeled_bp,
    target_cavity_predict,
)

L = Path("outputs/large")
SCALES = (0.85, 0.90, 0.95, 1.00, 1.05)
BLEND_WEIGHTS = (0.0, 0.25, 0.40, 0.50, 0.65, 0.80, 1.00)


def score_fn(val: float) -> float:
    return 100.0 * (1.0 - val / DENOMINATOR)


class PairMLPWide(nn.Module):
    """Antisymmetric wide interaction MLP: T(a, b) + T(b, a) = 10."""

    def __init__(self, dim: int = 48, hidden: int = 1024, layers: int = 3,
                 dropout: float = 0.25):
        super().__init__()
        self.emb = nn.Embedding(1026, dim)
        nn.init.normal_(self.emb.weight, std=0.08)
        blocks: list[nn.Module] = []
        width = 4 * dim
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.LayerNorm(hidden),
                       nn.GELU(), nn.Dropout(dropout)]
            width = hidden
        blocks.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*blocks)

    def _half(self, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        feat = torch.cat((ea, eb, ea * eb, ea - eb), dim=-1)
        return self.net(feat).squeeze(-1)

    def logit(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ea, eb = self.emb(a), self.emb(b)
        return self._half(ea, eb) - self._half(eb, ea)


class PairBilinearResidual(nn.Module):
    """Bradley-Terry + skew-symmetric low-rank type-chart bilinear + MLP residual."""

    def __init__(self, rank: int = 48, dim: int = 32, hidden: int = 768,
                 dropout: float = 0.25):
        super().__init__()
        self.rank = rank
        self.strength = nn.Embedding(1026, 1)
        nn.init.zeros_(self.strength.weight)
        self.left = nn.Embedding(1026, rank)
        self.right = nn.Embedding(1026, rank)
        nn.init.normal_(self.left.weight, std=0.05)
        nn.init.normal_(self.right.weight, std=0.05)

        self.emb = nn.Embedding(1026, dim)
        nn.init.normal_(self.emb.weight, std=0.08)
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _res(self, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat((ea, eb, ea * eb), dim=-1)).squeeze(-1)

    def logit(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        base = self.strength(a).squeeze(-1) - self.strength(b).squeeze(-1)
        la, ra = self.left(a), self.right(a)
        lb, rb = self.left(b), self.right(b)
        bilin = ((la * rb - lb * ra).sum(dim=-1)) / math.sqrt(self.rank)
        ea, eb = self.emb(a), self.emb(b)
        return base + bilin + (self._res(ea, eb) - self._res(eb, ea))


@torch.inference_mode()
def eval_rmse(model: nn.Module, a: torch.Tensor, b: torch.Tensor,
              y: torch.Tensor, batch_size: int = 65536) -> float:
    model.eval()
    sq = 0.0
    for start in range(0, len(y), batch_size):
        end = min(start + batch_size, len(y))
        pred = 10.0 * torch.sigmoid(model.logit(a[start:end], b[start:end]))
        sq += float(((pred - y[start:end]) ** 2).sum().item())
    return math.sqrt(sq / len(y))


@torch.inference_mode()
def materialise_table(model: nn.Module, device: str) -> np.ndarray:
    model.eval()
    table = np.full((1026, 1026), 5.0, dtype=np.float32)
    cols = torch.arange(1026, device=device, dtype=torch.long)
    for r0 in range(1, 1026, 64):
        r1 = min(r0 + 64, 1026)
        rows = torch.arange(r0, r1, device=device, dtype=torch.long)
        aa = rows[:, None].expand(r1 - r0, 1026).reshape(-1)
        bb = cols[None, :].expand(r1 - r0, 1026).reshape(-1)
        scores = (10.0 * torch.sigmoid(model.logit(aa, bb))).reshape(r1 - r0, 1026)
        table[r0:r1] = scores.cpu().numpy()
    # Enforce exact antisymmetry on 1..1025
    sub = table[1:, 1:]
    sub = 0.5 * (sub + (10.0 - sub.T))
    np.fill_diagonal(sub, 5.0)
    table[1:, 1:] = sub
    table[0, :] = 5.0
    table[:, 0] = 5.0
    return table


def fit_one_model(model: nn.Module, ta: torch.Tensor, tb: torch.Tensor,
                  ty: torch.Tensor, tw: torch.Tensor,
                  va: torch.Tensor, vb: torch.Tensor, vy: torch.Tensor,
                  lr: float = 2.2e-3, weight_decay: float = 1.2e-2,
                  epochs: int = 750, patience: int = 75,
                  batch_size: int = 4096) -> tuple[float, int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay, fused=True)
    steps_per_epoch = math.ceil(len(ty) / batch_size)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=steps_per_epoch * epochs, pct_start=0.06)

    best_rmse = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(ty), device=ty.device)
        for part in perm.split(batch_size):
            a_b, b_b, y_b, w_b = ta[part], tb[part], ty[part], tw[part]
            flip = torch.rand(len(part), device=ty.device) < 0.5
            aa = torch.where(flip, b_b, a_b)
            bb = torch.where(flip, a_b, b_b)
            yy = torch.where(flip, 10.0 - y_b, y_b)
            logits = model.logit(aa, bb)
            loss_vec = F.binary_cross_entropy_with_logits(
                logits, (yy / 10.0).clamp(0.0, 1.0), reduction="none")
            loss = (w_b * loss_vec).sum() / w_b.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            schedule.step()

        if epoch % 3 == 0 or epoch == epochs:
            val_r = eval_rmse(model, va, vb, vy)
            if val_r < best_rmse - 1e-5:
                best_rmse = val_r
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            if epoch - best_epoch >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_rmse, best_epoch


def fit_kfold_ensemble(lab_a: np.ndarray, lab_b: np.ndarray, lab_y: np.ndarray,
                       pseudo_path: Path, min_pseudo_w: float,
                       n_folds: int, seed_base: int, device: str,
                       tag: str) -> tuple[np.ndarray, list[dict]]:
    with np.load(pseudo_path) as pz:
        pa = pz["a"].astype(np.int64)
        pb = pz["b"].astype(np.int64)
        py = pz["y"].astype(np.float32)
        pw = pz["weight"].astype(np.float32)
    keep = pw >= min_pseudo_w
    pa, pb, py, pw = pa[keep], pb[keep], py[keep], pw[keep]
    print(f"[{tag}] labeled={len(lab_y)} pseudo(w>={min_pseudo_w})={len(py)} "
          f"mean_pw={float(pw.mean()):.4f}", flush=True)

    rng = np.random.default_rng(seed_base)
    perm = rng.permutation(len(lab_y))
    fold_ids = np.empty(len(lab_y), dtype=np.int64)
    fold_ids[perm] = np.arange(len(lab_y)) % n_folds

    pa_t = torch.as_tensor(pa, device=device, dtype=torch.long)
    pb_t = torch.as_tensor(pb, device=device, dtype=torch.long)
    py_t = torch.as_tensor(py, device=device, dtype=torch.float32)
    pw_t = torch.as_tensor(pw, device=device, dtype=torch.float32)

    acc_table = np.zeros((1026, 1026), dtype=np.float64)
    logs: list[dict] = []
    n_models = 0

    for fold in range(n_folds):
        val_m = fold_ids == fold
        tr_m = ~val_m
        la_tr = torch.as_tensor(lab_a[tr_m], device=device, dtype=torch.long)
        lb_tr = torch.as_tensor(lab_b[tr_m], device=device, dtype=torch.long)
        ly_tr = torch.as_tensor(lab_y[tr_m], device=device, dtype=torch.float32)
        lw_tr = torch.ones(len(la_tr), device=device, dtype=torch.float32)

        ta = torch.cat([la_tr, pa_t])
        tb = torch.cat([lb_tr, pb_t])
        ty = torch.cat([ly_tr, py_t])
        tw = torch.cat([lw_tr, pw_t])

        va = torch.as_tensor(lab_a[val_m], device=device, dtype=torch.long)
        vb = torch.as_tensor(lab_b[val_m], device=device, dtype=torch.long)
        vy = torch.as_tensor(lab_y[val_m], device=device, dtype=torch.float32)

        for arch_idx, (arch_name, factory) in enumerate((
            ("mlp_wide", lambda: PairMLPWide(dim=48, hidden=1024, layers=3, dropout=0.25)),
            ("bilin_res", lambda: PairBilinearResidual(rank=48, dim=32, hidden=768, dropout=0.25)),
        )):
            t0 = time.monotonic()
            torch.manual_seed(seed_base + fold * 17 + arch_idx * 101)
            model = factory().to(device)
            val_r, best_ep = fit_one_model(model, ta, tb, ty, tw, va, vb, vy)
            tab = materialise_table(model, device)
            acc_table += tab
            n_models += 1
            rec = {
                "fold": fold, "arch": arch_name, "val_rmse": round(val_r, 5),
                "best_epoch": best_ep, "seconds": round(time.monotonic() - t0, 1),
            }
            logs.append(rec)
            print(f"[{tag}] fold={fold}/{n_folds} arch={arch_name} "
                  f"val_rmse={val_r:.5f} ep={best_ep} ({rec['seconds']}s)", flush=True)
            del model
            torch.cuda.empty_cache()

    mean_table = (acc_table / n_models).astype(np.float32)
    return mean_table, logs


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    m1 = json.loads((L / "milestone1_rx_fusion.json").read_text())["winner"]
    debias = float(m1["debias"])

    rows = pd.read_csv("data/train.csv.gz")
    identity = {}
    for side in (1, 2):
        for p, v in zip(rows[f"noisy_image{side}"], rows[f"pokemon_id{side}"]):
            if pd.notna(v):
                identity[p] = int(v)
    paths = sorted(identity)
    true = np.array([identity[p] for p in paths], dtype=np.int64)
    folds = stratified_folds(true - 1, 3, SEED)
    index = pd.Index(paths)
    f1, f2 = index.get_indexer(rows.noisy_image1), index.get_indexer(rows.noisy_image2)
    known = (f1 >= 0) & (f2 >= 0)
    f1, f2 = f1[known], f2[known]
    a, b = true[f1], true[f2]
    y = rows.result.to_numpy(np.float32)[known]
    pair_f0 = (folds[f1] == 0) & (folds[f2] == 0)

    rng = np.random.default_rng(3780)
    pair_rows = np.flatnonzero(pair_f0)
    rng.shuffle(pair_rows)
    split = len(pair_rows) // 2

    # 1. Train leak-free Fold-0 table strictly on ~pair_f0 + Fold-0-clean pseudo-battles
    nof0_path = L / "pair_v4_nofold0_table.npy"
    if nof0_path.exists():
        v4_nof0 = np.load(nof0_path)
        nof0_logs = []
    else:
        v4_nof0, nof0_logs = fit_kfold_ensemble(
            a[~pair_f0], b[~pair_f0], y[~pair_f0],
            L / "unlabeled_pseudo_battles_f0clean.npz",
            min_pseudo_w=0.80, n_folds=6, seed_base=202611, device=device, tag="nofold0")
        np.save(nof0_path, v4_nof0)

    old_nof0 = np.load(L / "pair_semi_nofold0_new_table_round1.npy")
    f0_a, f0_b, f0_y = a[pair_rows], b[pair_rows], y[pair_rows]
    oracle_old = rmse(old_nof0[f0_a, f0_b], f0_y)
    oracle_v4 = rmse(v4_nof0[f0_a, f0_b], f0_y)
    print(f"Fold-0 ORACLE table RMSE: old={oracle_old:.5f} v4={oracle_v4:.5f}", flush=True)

    # 2. End-to-end Fold-0 evaluation across table blend weights
    known_allowed = np.triu(np.ones((1026, 1026), dtype=np.float32), k=1)
    known_allowed[0, :] = 0.0
    known_allowed[:, 0] = 0.0
    known_allowed[a[~pair_f0], b[~pair_f0]] = 0.0

    unk = rows.pokemon_id1.isna() & rows.pokemon_id2.isna()
    u_idx = pd.Index(Path("data/unlabeled_paths.txt").read_text().split())
    eu = u_idx.get_indexer(rows.noisy_image1[unk])
    ev = u_idx.get_indexer(rows.noisy_image2[unk])
    uy = rows.result[unk].to_numpy(np.float32)

    with np.load(L / "rx_v1_unlabeled_f0only.npz") as uz:
        u_cand = uz["candidates"].astype(np.int64)
        u_pri = uz["posteriors"].astype(np.float32)
    with np.load(L / "rx_v1_fold0_scores.npz") as fz:
        v_idx = fz["valid_indices"].astype(np.int64)
        cand = fz["candidates"].astype(np.int64)
        post = fz["posteriors"].astype(np.float32)
        oop = fz["oop_mass"].astype(np.float32)

    pos = np.full(len(paths), -1, dtype=np.int32)
    pos[v_idx] = np.arange(len(v_idx))
    one, two = pos[f1[pair_rows]], pos[f2[pair_rows]]

    sweep = []
    for w_v4 in BLEND_WEIGHTS:
        tab = ((1.0 - w_v4) * old_nof0 + w_v4 * v4_nof0).astype(np.float32)
        orc = rmse(tab[f0_a, f0_b], f0_y)
        _, _, unl_lnot0 = run_unlabeled_bp(
            u_cand, u_pri, eu, ev, uy, tab, known_allowed, device=device)
        _, _, test_lnot, _, _ = target_cavity_predict(
            post, oop, cand, one, two, known_allowed, unl_lnot0, tab,
            debias=debias, device=device)
        _, _, unl_lnot = run_unlabeled_bp(
            u_cand, u_pri, eu, ev, uy, tab, known_allowed,
            external_log_avail=test_lnot, device=device)

        best = None
        for scale in SCALES:
            qs = power_scale_posteriors(post, scale)
            pred, top_p, _, _, _ = target_cavity_predict(
                qs, oop, cand, one, two, known_allowed, unl_lnot, tab,
                debias=debias, cavity_iters=12, use_escape=False, device=device)
            both_ok = float(((top_p[:, 0] == f0_a) & (top_p[:, 1] == f0_b))[split:].mean())
            tr_r = rmse(pred[:split], f0_y[:split])
            te_r = rmse(pred[split:], f0_y[split:])
            item = {
                "scale": scale, "tune_rmse": tr_r, "test_rmse": te_r,
                "tune_score": score_fn(tr_r), "test_score": score_fn(te_r),
                "test_both_correct": both_ok,
            }
            if best is None or item["tune_rmse"] < best["tune_rmse"]:
                best = item
        rec = {"w_v4": w_v4, "oracle_rmse": orc, "selected": best}
        sweep.append(rec)
        print(f"[table_blend] w_v4={w_v4:.2f} oracle_rmse={orc:.5f} "
              f"tune={best['tune_score']:.4f} test={best['test_score']:.4f} "
              f"both={best['test_both_correct']:.5f}", flush=True)

    anchor = next(r for r in sweep if r["w_v4"] == 0.0)
    chosen = min(sweep, key=lambda r: (r["selected"]["tune_rmse"], r["w_v4"]))
    passed = (chosen["w_v4"] > 0.0
              and chosen["selected"]["test_rmse"] < anchor["selected"]["test_rmse"])
    print(f"Milestone 2 gate: w_v4={chosen['w_v4']} passed={passed} "
          f"(test {anchor['selected']['test_score']:.4f} -> {chosen['selected']['test_score']:.4f})",
          flush=True)

    active_nof0_table = L / "pair_semi_nofold0_new_table_round1.npy"
    active_full_table = L / "pair_semi_v23_blend_table.npy"
    active_scale = anchor["selected"]["scale"]
    sub_csv = None
    full_logs: list[dict] = []

    if passed:
        w_win = chosen["w_v4"]
        active_scale = chosen["selected"]["scale"]
        blended_nof0 = ((1.0 - w_win) * old_nof0 + w_win * v4_nof0).astype(np.float32)
        active_nof0_table = L / "pair_v4_nofold0_blend_table.npy"
        np.save(active_nof0_table, blended_nof0)

        full_raw_path = L / "pair_v4_full_table.npy"
        if full_raw_path.exists():
            v4_full = np.load(full_raw_path)
        else:
            with np.load("data/known_battles.npz") as kz:
                ka_all = kz["a"].astype(np.int64)
                kb_all = kz["b"].astype(np.int64)
                ky_all = kz["y"].astype(np.float32)
            v4_full, full_logs = fit_kfold_ensemble(
                ka_all, kb_all, ky_all,
                L / "unlabeled_pseudo_battles_gen2.npz",
                min_pseudo_w=0.80, n_folds=6, seed_base=202699, device=device, tag="full")
            np.save(full_raw_path, v4_full)

        old_full = np.load(L / "pair_semi_v23_blend_table.npy")
        blended_full = ((1.0 - w_win) * old_full + w_win * v4_full).astype(np.float32)
        active_full_table = L / "pair_v4_full_blend_table.npy"
        np.save(active_full_table, blended_full)

        sub_csv = L / "submission_rx_v2_table.csv"
        sub_args = argparse.Namespace(
            battles=Path("data/known_battles.npz"),
            train_csv=Path("data/train.csv.gz"),
            test_csv=Path("data/test.csv.gz"),
            unlabeled_paths=Path("data/unlabeled_paths.txt"),
            data_dir=Path("."),
            unlabeled_priors=L / "rx_v1_unlabeled.npz",
            test_npz=L / "rx_v1_test.npz",
            table=active_full_table,
            scale=active_scale,
            debias=debias,
            cavity_iters=12,
            use_escape=False,
            save_pseudo_battles=L / "unlabeled_pseudo_battles_gen2_v2.npz",
            save_transductive_bank=None,
            bank_min_conf=0.92,
            test_bank_min_conf=0.93,
            output=sub_csv,
        )
        run_submission(sub_args, device)
        print(f"Milestone 2 complete -> {sub_csv}", flush=True)

    report = {
        "oracle_old_rmse": oracle_old,
        "oracle_v4_rmse": oracle_v4,
        "anchor": anchor,
        "chosen": chosen,
        "passed": passed,
        "active_nof0_table": str(active_nof0_table),
        "active_full_table": str(active_full_table),
        "active_scale": active_scale,
        "active_debias": debias,
        "submission_csv": str(sub_csv) if sub_csv else None,
        "sweep": sweep,
        "nof0_logs": nof0_logs,
        "full_logs": full_logs,
    }
    (L / "milestone2_pair_table_v4.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
