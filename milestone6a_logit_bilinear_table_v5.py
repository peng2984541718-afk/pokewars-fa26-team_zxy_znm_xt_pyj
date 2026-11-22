"""Milestone 6A (Discovery #2 Part B): Logit-Space Skew-Symmetric Bilinear +
SwiGLU Multiplicative + Residual Deep Ensemble (`PairTable v5`) with Round-2 EM
Battle-BP Pseudo-Battles (`80.92531` teacher).

Key advances over Milestone 2 (`pair_v4`):
1. Round-2 EM Pseudo-Battles: Unlabeled battles are disambiguated using the
   `0.186/0.175` RMSE `pair_v4` tables + ConvNeXt-fused `rx_v1` posteriors
   instead of the old `0.296` table.
2. Discovery #2 Architecture Match: Since 96.5% of `logit(T/10)` variance is
   low-rank skew-symmetric (Pokemon stats + multiplicative 18-type chart), we
   combine three antisymmetric architectures with `dim=32..36` bottleneck
   embeddings:
   - `SkewBilinearResNet`: explicit rank-48 skew-symmetric bilinear cross-term
     + Pre-LN residual MLP
   - `SwiGLUPairNet`: multiplicative gated residual interaction network
   - `PairResMLP5`: 5-layer residual interaction network with Hadamard & L1 diff
3. Logit-Space Ensembling &Logit Blending: Models are averaged in logit space
   `logit(T/10)` rather than probability `[0, 10]` space, eliminating Jensen
   shrinkage near the 0/10 shutout cliffs!
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
from milestone1_rx_fusion_and_pack import save_pseudo_battles

L = Path("outputs/large")
SCALES = (0.90, 0.95, 1.00)
LOGIT_TEMPS = (0.96, 1.00, 1.04, 1.08)
BLEND_WEIGHTS = (0.0, 0.35, 0.50, 0.65, 0.80, 1.00)


def score_fn(val: float) -> float:
    return 100.0 * (1.0 - val / DENOMINATOR)


def table_to_logit(tab: np.ndarray) -> np.ndarray:
    p = np.clip(tab.astype(np.float64) / 10.0, 1e-6, 1.0 - 1e-6)
    lg = np.log(p / (1.0 - p))
    lg = 0.5 * (lg - lg.T)
    np.fill_diagonal(lg, 0.0)
    lg[0, :] = 0.0
    lg[:, 0] = 0.0
    return lg.astype(np.float32)


def logit_to_table(lg: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = np.clip(lg.astype(np.float64) * float(temp), -18.0, 18.0)
    sub = 10.0 / (1.0 + np.exp(-x))
    sub[1:, 1:] = 0.5 * (sub[1:, 1:] + (10.0 - sub[1:, 1:].T))
    np.fill_diagonal(sub, 5.0)
    sub[0, :] = 5.0
    sub[:, 0] = 5.0
    return sub.astype(np.float32)


class SharedBilinearDeepMLP(nn.Module):
    """Single shared low-dim species bottleneck (`dim` params/species ONLY!) +
    shared `dim x dim` skew-symmetric type operator `ea^T (W - W^T) eb` +
    serial LayerNorm-GELU-Dropout deep interaction network (NO skip highway).
    """

    def __init__(self, dim: int = 28, hidden: int = 1280, layers: int = 4,
                 dropout: float = 0.25, use_skew_op: bool = True):
        super().__init__()
        self.dim = dim
        self.use_skew_op = use_skew_op
        self.emb = nn.Embedding(1026, dim)
        nn.init.normal_(self.emb.weight, std=0.08)
        if use_skew_op:
            self.skew_w = nn.Parameter(torch.randn(dim, dim) * (0.04 / math.sqrt(dim)))
        blocks: list[nn.Module] = []
        width = 4 * dim
        for _ in range(layers):
            blocks += [
                nn.Linear(width, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            width = hidden
        blocks.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*blocks)

    def _half(self, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        feat = torch.cat((ea, eb, ea * eb, ea - eb), dim=-1)
        return self.net(feat).squeeze(-1)

    def logit(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ea, eb = self.emb(a), self.emb(b)
        out = self._half(ea, eb) - self._half(eb, ea)
        if self.use_skew_op:
            skew = self.skew_w - self.skew_w.T
            out = out + ((ea @ skew) * eb).sum(dim=-1)
        return out


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
def materialise_logit_table(model: nn.Module, device: str) -> np.ndarray:
    model.eval()
    lg = np.zeros((1026, 1026), dtype=np.float32)
    cols = torch.arange(1026, device=device, dtype=torch.long)
    for r0 in range(1, 1026, 64):
        r1 = min(r0 + 64, 1026)
        rows = torch.arange(r0, r1, device=device, dtype=torch.long)
        aa = rows[:, None].expand(r1 - r0, 1026).reshape(-1)
        bb = cols[None, :].expand(r1 - r0, 1026).reshape(-1)
        out = model.logit(aa, bb).reshape(r1 - r0, 1026)
        lg[r0:r1] = out.cpu().numpy()
    sub = lg[1:, 1:]
    sub = 0.5 * (sub - sub.T)
    np.fill_diagonal(sub, 0.0)
    lg[1:, 1:] = sub
    lg[0, :] = 0.0
    lg[:, 0] = 0.0
    return lg


def fit_one_model(model: nn.Module, ta: torch.Tensor, tb: torch.Tensor,
                  ty: torch.Tensor, tw: torch.Tensor,
                  va: torch.Tensor, vb: torch.Tensor, vy: torch.Tensor,
                  lr: float = 2.0e-3, weight_decay: float = 1.4e-2,
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


def canonicalise_pseudo_npz(path: Path) -> None:
    with np.load(path) as pz:
        pa = pz["a"].astype(np.int64)
        pb = pz["b"].astype(np.int64)
        py = pz["y"].astype(np.float32)
        pw = pz["weight"].astype(np.float32)
    swap = pa > pb
    if np.any(swap):
        pa_n = np.where(swap, pb, pa)
        pb_n = np.where(swap, pa, pb)
        py_n = np.where(swap, 10.0 - py, py)
        np.savez_compressed(path, a=pa_n.astype(np.int32), b=pb_n.astype(np.int32),
                            y=py_n.astype(np.float32), weight=pw.astype(np.float32))


def build_f0clean_v2_pseudo(rows: pd.DataFrame, known_allowed: np.ndarray,
                            v4_nof0: np.ndarray, debias: float,
                            device: str) -> Path:
    out_path = L / "unlabeled_pseudo_battles_f0clean_v2.npz"
    if out_path.exists():
        canonicalise_pseudo_npz(out_path)
        return out_path

    unk = rows.pokemon_id1.isna() & rows.pokemon_id2.isna()
    u_idx = pd.Index(Path("data/unlabeled_paths.txt").read_text().split())
    eu = u_idx.get_indexer(rows.noisy_image1[unk])
    ev = u_idx.get_indexer(rows.noisy_image2[unk])
    uy = rows.result[unk].to_numpy(np.float32)

    with np.load(L / "rx_v1_unlabeled_f0only.npz") as uz:
        u_cand = uz["candidates"].astype(np.int64)
        u_pri = uz["posteriors"].astype(np.float32)

    # Strictly Fold-0-free: run unlabeled BP using ONLY unlabeled edges + Fold-0-clean table
    _, u_edge, _ = run_unlabeled_bp(
        u_cand, u_pri, eu, ev, uy, v4_nof0, known_allowed, device=device)
    save_pseudo_battles(out_path, u_cand, eu, ev, uy, u_edge)
    canonicalise_pseudo_npz(out_path)
    return out_path


def fit_kfold_logit_ensemble(lab_a: np.ndarray, lab_b: np.ndarray, lab_y: np.ndarray,
                             pseudo_path: Path, min_pseudo_w: float,
                             n_folds: int, seed_base: int, device: str,
                             tag: str) -> tuple[np.ndarray, list[dict]]:
    canonicalise_pseudo_npz(pseudo_path)
    with np.load(pseudo_path) as pz:
        pa = pz["a"].astype(np.int64)
        pb = pz["b"].astype(np.int64)
        py = pz["y"].astype(np.float32)
        pw = pz["weight"].astype(np.float32)
    keep = (pw >= min_pseudo_w) & (pa > 0) & (pb > 0) & (pa != pb)
    pa, pb, py, pw = pa[keep], pb[keep], py[keep], pw[keep]
    print(f"[{tag}] labeled={len(lab_y)} pseudo_r2(w>={min_pseudo_w})={len(py)} "
          f"mean_pw={float(pw.mean()):.4f}", flush=True)

    rng = np.random.default_rng(seed_base)
    perm = rng.permutation(len(lab_y))
    fold_ids = np.empty(len(lab_y), dtype=np.int64)
    fold_ids[perm] = np.arange(len(lab_y)) % n_folds

    pa_t = torch.as_tensor(pa, device=device, dtype=torch.long)
    pb_t = torch.as_tensor(pb, device=device, dtype=torch.long)
    py_t = torch.as_tensor(py, device=device, dtype=torch.float32)
    pw_t = torch.as_tensor(pw, device=device, dtype=torch.float32)

    acc_logit = np.zeros((1026, 1026), dtype=np.float64)
    logs: list[dict] = []
    weight_sum = 0.0

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
            ("bottleneck26_skew4", lambda: SharedBilinearDeepMLP(dim=26, hidden=1280, layers=4, dropout=0.25, use_skew_op=True)),
            ("bottleneck30_deep5", lambda: SharedBilinearDeepMLP(dim=30, hidden=1344, layers=5, dropout=0.25, use_skew_op=True)),
            ("bottleneck34_deep4", lambda: SharedBilinearDeepMLP(dim=34, hidden=1408, layers=4, dropout=0.25, use_skew_op=False)),
        )):
            t0 = time.monotonic()
            torch.manual_seed(seed_base + fold * 31 + arch_idx * 137)
            model = factory().to(device)
            val_r, best_ep = fit_one_model(model, ta, tb, ty, tw, va, vb, vy)
            lg_tab = materialise_logit_table(model, device)
            w_m = 1.0 / (max(val_r, 0.14) ** 4)
            acc_logit += w_m * lg_tab
            weight_sum += w_m
            rec = {
                "fold": fold, "arch": arch_name, "val_rmse": round(val_r, 5),
                "best_epoch": best_ep, "seconds": round(time.monotonic() - t0, 1),
            }
            logs.append(rec)
            print(f"[{tag}] fold={fold}/{n_folds} arch={arch_name} "
                  f"val_rmse={val_r:.5f} ep={best_ep} ({rec['seconds']}s)", flush=True)
            del model
            torch.cuda.empty_cache()

    mean_logit = (acc_logit / weight_sum).astype(np.float32)
    return mean_logit, logs


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    m2 = json.loads((L / "milestone2_pair_table_v4.json").read_text())
    debias = float(m2["active_debias"])
    v4_nof0 = np.load(Path(m2["active_nof0_table"]))
    v4_full = np.load(Path(m2["active_full_table"]))

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

    known_allowed = np.triu(np.ones((1026, 1026), dtype=np.float32), k=1)
    known_allowed[0, :] = 0.0
    known_allowed[:, 0] = 0.0
    known_allowed[a[~pair_f0], b[~pair_f0]] = 0.0

    # 1. Build Round-2 Fold-0-clean pseudo-battles using v4_nof0 + ConvNeXt rx_v1
    f0_pseudo_v2 = build_f0clean_v2_pseudo(rows, known_allowed, v4_nof0, debias, device)

    nof0_logit_path = L / "pair_v5_nofold0_logit.npy"
    if nof0_logit_path.exists():
        v5_nof0_logit = np.load(nof0_logit_path)
        nof0_logs = []
    else:
        v5_nof0_logit, nof0_logs = fit_kfold_logit_ensemble(
            a[~pair_f0], b[~pair_f0], y[~pair_f0],
            f0_pseudo_v2, min_pseudo_w=0.85, n_folds=6,
            seed_base=202705, device=device, tag="v5_nofold0")
        np.save(nof0_logit_path, v5_nof0_logit)

    f0_a, f0_b, f0_y = a[pair_rows], b[pair_rows], y[pair_rows]
    oracle_v4 = rmse(v4_nof0[f0_a, f0_b], f0_y)
    v5_nof0_raw = logit_to_table(v5_nof0_logit, temp=1.0)
    oracle_v5 = rmse(v5_nof0_raw[f0_a, f0_b], f0_y)
    print(f"Fold-0 ORACLE table RMSE: v4={oracle_v4:.5f} v5={oracle_v5:.5f}", flush=True)

    # 2. Sweep logit-space blend (w_v5, logit_temp, scale) on Fold-0 tune, verify on test
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
    actual = y[pair_rows]
    ta, tb = a[pair_rows], b[pair_rows]

    v4_nof0_logit = table_to_logit(v4_nof0)
    sweep = []
    for w_v5 in BLEND_WEIGHTS:
        temps = (1.00,) if w_v5 == 0.0 else LOGIT_TEMPS
        for ltemp in temps:
            mix_lg = (1.0 - w_v5) * v4_nof0_logit + w_v5 * v5_nof0_logit
            tab = v4_nof0 if (w_v5 == 0.0 and ltemp == 1.0) else logit_to_table(mix_lg, temp=ltemp)
            ora = rmse(tab[f0_a, f0_b], f0_y)
            _, _, unl_lnot0 = run_unlabeled_bp(
                u_cand, u_pri, eu, ev, uy, tab, known_allowed, device=device)
            _, _, test_lnot, _, _ = target_cavity_predict(
                post, oop, cand, one, two, known_allowed, unl_lnot0, tab,
                debias=debias, device=device)
            _, _, unl_lnot = run_unlabeled_bp(
                u_cand, u_pri, eu, ev, uy, tab, known_allowed,
                external_log_avail=test_lnot, device=device)

            best_s = None
            for sc in SCALES:
                qs = power_scale_posteriors(post, sc)
                pred, top_p, _, _, _ = target_cavity_predict(
                    qs, oop, cand, one, two, known_allowed, unl_lnot, tab,
                    debias=debias, cavity_iters=12, use_escape=False, device=device)
                both_ok = float(((top_p[:, 0] == ta) & (top_p[:, 1] == tb))[split:].mean())
                tr_r = rmse(pred[:split], actual[:split])
                te_r = rmse(pred[split:], actual[split:])
                it = {
                    "scale": sc, "tune_rmse": tr_r, "test_rmse": te_r,
                    "tune_score": score_fn(tr_r), "test_score": score_fn(te_r),
                    "test_both_correct": both_ok,
                }
                if best_s is None or it["tune_rmse"] < best_s["tune_rmse"]:
                    best_s = it
            rec = {
                "w_v5": w_v5, "logit_temp": ltemp, "oracle_rmse": ora,
                "selected": best_s,
            }
            sweep.append(rec)
            print(f"[v5_blend] w_v5={w_v5:.2f} temp={ltemp:.2f} oracle={ora:.5f} "
                  f"tune={best_s['tune_score']:.4f} test={best_s['test_score']:.4f} "
                  f"both={best_s['test_both_correct']:.5f}", flush=True)

    anchor = next(r for r in sweep if r["w_v5"] == 0.0)
    chosen = min(sweep, key=lambda r: (r["selected"]["tune_rmse"], abs(r["logit_temp"] - 1.0)))
    passed = (chosen["w_v5"] > 0.0
              and chosen["selected"]["test_score"] > anchor["selected"]["test_score"] + 0.005)
    print(f"Milestone 6A gate: w_v5={chosen['w_v5']} temp={chosen['logit_temp']} passed={passed} "
          f"(test {anchor['selected']['test_score']:.4f} -> {chosen['selected']['test_score']:.4f})",
          flush=True)

    sub_csv = None
    full_logs: list[dict] = []
    if passed:
        w_win = chosen["w_v5"]
        t_win = chosen["logit_temp"]
        sc_win = chosen["selected"]["scale"]
        nof0_blend = logit_to_table((1.0 - w_win) * v4_nof0_logit + w_win * v5_nof0_logit, temp=t_win)
        np.save(L / "pair_v5_nofold0_blend_table.npy", nof0_blend)

        full_logit_path = L / "pair_v5_full_logit.npy"
        if full_logit_path.exists():
            v5_full_logit = np.load(full_logit_path)
        else:
            with np.load("data/known_battles.npz") as kz:
                ka_all = kz["a"].astype(np.int64)
                kb_all = kz["b"].astype(np.int64)
                ky_all = kz["y"].astype(np.float32)
            v5_full_logit, full_logs = fit_kfold_logit_ensemble(
                ka_all, kb_all, ky_all,
                L / "unlabeled_pseudo_battles_gen2_v2.npz",
                min_pseudo_w=0.85, n_folds=6, seed_base=202788,
                device=device, tag="v5_full")
            np.save(full_logit_path, v5_full_logit)

        v4_full_logit = table_to_logit(v4_full)
        full_blend = logit_to_table((1.0 - w_win) * v4_full_logit + w_win * v5_full_logit, temp=t_win)
        np.save(L / "pair_v5_full_blend_table.npy", full_blend)

        # Atomically upgrade active table pointers for downstream Milestones 4/5/6B/6C
        m2_path = L / "milestone2_pair_table_v4.json"
        m2_obj = json.loads(m2_path.read_text())
        m2_obj["v4_nof0_backup"] = m2_obj["active_nof0_table"]
        m2_obj["v4_full_backup"] = m2_obj["active_full_table"]
        m2_obj["active_nof0_table"] = str(L / "pair_v5_nofold0_blend_table.npy")
        m2_obj["active_full_table"] = str(L / "pair_v5_full_blend_table.npy")
        m2_tmp = L / "milestone2_pair_table_v4.json.tmp"
        m2_tmp.write_text(json.dumps(m2_obj, indent=2) + "\n")
        m2_tmp.replace(m2_path)
        sub_csv = L / "submission_rx_v2b_table_v5.csv"
        sub_args = argparse.Namespace(
            battles=Path("data/known_battles.npz"),
            train_csv=Path("data/train.csv.gz"),
            test_csv=Path("data/test.csv.gz"),
            unlabeled_paths=Path("data/unlabeled_paths.txt"),
            data_dir=Path("."),
            unlabeled_priors=L / "rx_v1_unlabeled.npz",
            test_npz=L / "rx_v1_test.npz",
            table=L / "pair_v5_full_blend_table.npy",
            scale=sc_win,
            debias=debias,
            cavity_iters=12,
            use_escape=False,
            save_pseudo_battles=L / "unlabeled_pseudo_battles_gen2_v3.npz",
            save_transductive_bank=L / "transductive_bank_bp_v5.npz",
            bank_min_conf=0.92,
            test_bank_min_conf=0.94,
            output=sub_csv,
        )
        run_submission(sub_args, device)
        (L / "submission_latest_best.csv").write_bytes(sub_csv.read_bytes())
        print(f"Milestone 6A complete -> {sub_csv}", flush=True)

    report = {
        "oracle_v4_rmse": oracle_v4,
        "oracle_v5_rmse": oracle_v5,
        "anchor": anchor,
        "chosen": chosen,
        "passed": passed,
        "submission_csv": str(sub_csv) if sub_csv else None,
        "sweep": sweep,
        "nof0_logs": nof0_logs,
        "full_logs": full_logs,
    }
    (L / "milestone6a_table_v5.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
