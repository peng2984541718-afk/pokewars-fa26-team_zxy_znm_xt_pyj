"""Milestone 3: Heteroscedastic Binomial Unlabeled BP + Laplacian-noise-adaptive
posterior scaling, strictly gated on the untouched Fold-0 test half.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_soft_score import DENOMINATOR, rmse
from image_experiment import SEED, stratified_folds
from joint_constrained_inference import (
    chunked_unordered_log_not,
    target_cavity_predict,
)

L = Path("outputs/large")
BP_KERNELS = (
    (0.50, 0.00),  # exact homoscedastic baseline
    (0.35, 0.45),
    (0.28, 0.55),
    (0.22, 0.65),
    (0.18, 0.35),  # matched to PairMLP v4 0.175 RMSE + 2048-Bernoulli floor
)
SCALE_BASES = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10)
BETA_LAPS = (0.00, 0.05, 0.10, 0.16, 0.22)
DEBIASES = (1.00, 1.15, 1.25, 1.35)


def score_fn(val: float) -> float:
    return 100.0 * (1.0 - val / DENOMINATOR)


def adaptive_power_scale(post: np.ndarray, scale_vec: np.ndarray) -> np.ndarray:
    s = scale_vec[:, None].astype(np.float64)
    lp = s * np.log(np.clip(post.astype(np.float64), 1e-12, 1.0))
    lp -= lp.max(1, keepdims=True)
    p = np.exp(lp)
    return (p / np.maximum(p.sum(1, keepdims=True), 1e-12)).astype(np.float32)


@torch.no_grad()
def run_unlabeled_bp_hetero(cand: np.ndarray, pri: np.ndarray,
                            eu: np.ndarray, ev: np.ndarray,
                            y_edge: np.ndarray, table: np.ndarray,
                            known_allowed: np.ndarray,
                            s0: float = 0.50, s1: float = 0.00,
                            external_log_avail: np.ndarray | None = None,
                            target_per_class: float = 67.0, iters: int = 24,
                            device: str = "cuda"):
    _, K = cand.shape
    E = len(eu)
    cand_t = torch.as_tensor(cand, device=device, dtype=torch.long)
    eu_t = torch.as_tensor(eu, device=device, dtype=torch.long)
    ev_t = torch.as_tensor(ev, device=device, dtype=torch.long)
    c_u, c_v = cand_t[eu_t], cand_t[ev_t]
    T_t = torch.as_tensor(table, device=device, dtype=torch.float32)
    M_t = torch.as_tensor(known_allowed, device=device, dtype=torch.float32)

    pred = T_t[c_u[:, :, None], c_v[:, None, :]]
    obs = torch.as_tensor(y_edge, device=device, dtype=torch.float32)[:, None, None]
    p01 = (pred / 10.0).clamp(0.0, 1.0)
    sigma = torch.sqrt(torch.clamp(s0 * s0 + (s1 * s1) * p01 * (1.0 - p01), min=1e-4))
    psi = (0.95 * torch.exp(-0.5 * ((obs - pred) / sigma) ** 2)
           / (np.sqrt(2.0 * np.pi) * sigma) + 0.05 / 10.0)
    mask_uv = M_t[c_u[:, :, None], c_v[:, None, :]]
    base_log_psi = torch.log((psi * (mask_uv + 1e-6 * (1.0 - mask_uv))).clamp_min(1e-30))
    if external_log_avail is not None:
        ext_t = torch.as_tensor(external_log_avail, device=device, dtype=torch.float32)
        base_log_psi = base_log_psi + ext_t[c_u[:, :, None], c_v[:, None, :]].clamp(-20.0, 0.0)

    log_pri = torch.log(torch.as_tensor(pri, device=device, dtype=torch.float32).clamp_min(1e-12))
    msg_u2v = torch.zeros((E, K), device=device)
    msg_v2u = torch.zeros((E, K), device=device)
    class_bias = torch.zeros(1026, device=device)
    cav_log = torch.zeros((E, K, K), device=device)

    for it in range(iters):
        tot = log_pri + class_bias[cand_t]
        tot.index_add_(0, ev_t, msg_u2v)
        tot.index_add_(0, eu_t, msg_v2u)
        if target_per_class > 0 and it >= 3:
            b_prob = torch.softmax(tot, dim=1)
            cnt = torch.zeros(1026, device=device)
            cnt.scatter_add_(0, cand_t.reshape(-1), b_prob.reshape(-1))
            cnt[0] = target_per_class
            class_bias = class_bias + 0.5 * torch.log(target_per_class / cnt.clamp_min(1e-3))
            class_bias[0] = 0.0
            tot = log_pri + class_bias[cand_t]
            tot.index_add_(0, ev_t, msg_u2v)
            tot.index_add_(0, eu_t, msg_v2u)

        in_u = tot[eu_t] - msg_v2u
        in_v = tot[ev_t] - msg_u2v
        if it >= 5:
            le = in_u[:, :, None] + in_v[:, None, :] + base_log_psi + cav_log
            le = le - le.amax(dim=(1, 2), keepdim=True)
            ep = torch.exp(le)
            ep = ep / ep.sum(dim=(1, 2), keepdim=True)
            tot_lnot, self_lnot = chunked_unordered_log_not(ep, c_u, c_v, clip_max=0.9999)
            cav_new = (tot_lnot[c_u[:, :, None], c_v[:, None, :]] - self_lnot).clamp(-20.0, 0.0)
            cav_log = 0.5 * cav_log + 0.5 * cav_new

        eff_psi = base_log_psi + cav_log
        new_u2v = torch.logsumexp(in_u[:, :, None] + eff_psi, dim=1)
        new_u2v = new_u2v - new_u2v.max(dim=1, keepdim=True).values
        new_v2u = torch.logsumexp(in_v[:, :, None] + eff_psi.transpose(1, 2), dim=1)
        new_v2u = new_v2u - new_v2u.max(dim=1, keepdim=True).values
        msg_u2v = 0.5 * msg_u2v + 0.5 * new_u2v
        msg_v2u = 0.5 * msg_v2u + 0.5 * new_v2u

    tot = log_pri + class_bias[cand_t]
    tot.index_add_(0, ev_t, msg_u2v)
    tot.index_add_(0, eu_t, msg_v2u)
    node_post = torch.softmax(tot, dim=1)
    in_u = tot[eu_t] - msg_v2u
    in_v = tot[ev_t] - msg_u2v
    le = in_u[:, :, None] + in_v[:, None, :] + base_log_psi + cav_log
    le = le - le.amax(dim=(1, 2), keepdim=True)
    ep = torch.exp(le)
    ep = ep / ep.sum(dim=(1, 2), keepdim=True)
    unl_lnot, _ = chunked_unordered_log_not(ep, c_u, c_v, clip_max=0.99999)
    return node_post.cpu().numpy(), ep.cpu().numpy(), unl_lnot.cpu().numpy()


def z_lap_from_gallery(gallery_npz: Path, expected_indices: np.ndarray | None = None) -> np.ndarray:
    with np.load(gallery_npz) as gz:
        if expected_indices is not None:
            if not np.array_equal(gz["valid_indices"].astype(np.int64), expected_indices):
                raise SystemExit(f"valid_indices mismatch in {gallery_npz}")
        lap = gz["image_quality"].astype(np.float32)[:, 0]
    return np.clip((lap - 28.0) / 12.0, -1.0, 2.5).astype(np.float32)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m2 = json.loads((L / "milestone2_pair_table_v4.json").read_text())
    nof0_table = np.load(Path(m2["active_nof0_table"]))
    full_table = np.load(Path(m2["active_full_table"]))
    base_debias = float(m2["active_debias"])
    base_scale = float(m2["active_scale"])

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

    z_lap_f0 = z_lap_from_gallery(L / "gallery_rx_top40_f0.npz", v_idx)
    pos = np.full(len(paths), -1, dtype=np.int32)
    pos[v_idx] = np.arange(len(v_idx))
    one, two = pos[f1[pair_rows]], pos[f2[pair_rows]]
    actual = y[pair_rows]
    ta, tb = a[pair_rows], b[pair_rows]

    debiases = tuple(sorted(set(DEBIASES + (base_debias,))))
    scale_bases = tuple(sorted(set(SCALE_BASES + (base_scale,))))
    records = []
    for s0, s1 in BP_KERNELS:
        _, _, unl_lnot0 = run_unlabeled_bp_hetero(
            u_cand, u_pri, eu, ev, uy, nof0_table, known_allowed,
            s0=s0, s1=s1, device=device)
        for debias in debiases:
            _, _, test_lnot, _, _ = target_cavity_predict(
                post, oop, cand, one, two, known_allowed, unl_lnot0, nof0_table,
                debias=debias, device=device)
            _, _, unl_lnot = run_unlabeled_bp_hetero(
                u_cand, u_pri, eu, ev, uy, nof0_table, known_allowed,
                s0=s0, s1=s1, external_log_avail=test_lnot, device=device)

            best_cell = None
            for scale_b in scale_bases:
                for beta_l in BETA_LAPS:
                    s_vec = np.clip(scale_b - beta_l * z_lap_f0, 0.55, 1.45)
                    qs = adaptive_power_scale(post, s_vec)
                    pred, top_p, _, _, _ = target_cavity_predict(
                        qs, oop, cand, one, two, known_allowed, unl_lnot, nof0_table,
                        debias=debias, cavity_iters=12, use_escape=False, device=device)
                    both_ok = float(((top_p[:, 0] == ta) & (top_p[:, 1] == tb))[split:].mean())
                    tr_r = rmse(pred[:split], actual[:split])
                    te_r = rmse(pred[split:], actual[split:])
                    item = {
                        "s0": s0, "s1": s1, "debias": debias,
                        "scale_base": scale_b, "beta_lap": beta_l,
                        "tune_rmse": tr_r, "test_rmse": te_r,
                        "tune_score": score_fn(tr_r), "test_score": score_fn(te_r),
                        "test_both_correct": both_ok,
                    }
                    records.append(item)
                    if best_cell is None or item["tune_rmse"] < best_cell["tune_rmse"]:
                        best_cell = item
            print(f"[cal] bp=({s0:.2f},{s1:.2f}) deb={debias:.2f} "
                  f"scale={best_cell['scale_base']:.2f} beta={best_cell['beta_lap']:.2f} "
                  f"tune={best_cell['tune_score']:.4f} test={best_cell['test_score']:.4f}",
                  flush=True)

    anchor = next(
        r for r in records
        if r["s0"] == 0.50 and r["s1"] == 0.00
        and abs(r["debias"] - base_debias) < 1e-6
        and abs(r["scale_base"] - base_scale) < 1e-6
        and r["beta_lap"] == 0.00
    )
    chosen = min(records, key=lambda r: (
        r["tune_rmse"], r["beta_lap"], abs(r["s1"]), abs(r["debias"] - base_debias)))
    passed = chosen["test_score"] > anchor["test_score"] + 0.005
    print(f"Milestone 3 gate: passed={passed} "
          f"(test {anchor['test_score']:.4f} -> {chosen['test_score']:.4f})", flush=True)

    sub_csv = None
    if passed:
        with np.load("data/known_battles.npz") as kz:
            ka_all, kb_all = kz["a"].astype(np.int64), kz["b"].astype(np.int64)
        known_full = np.triu(np.ones((1026, 1026), dtype=np.float32), k=1)
        known_full[0, :] = 0.0
        known_full[:, 0] = 0.0
        known_full[ka_all, kb_all] = 0.0

        test_paths = Path("test_paths.txt").read_text().splitlines()
        p_idx = pd.Index(test_paths)
        test_df = pd.read_csv("data/test.csv.gz")
        idx1 = p_idx.get_indexer(test_df.noisy_image1)
        idx2 = p_idx.get_indexer(test_df.noisy_image2)

        with np.load(L / "rx_v1_unlabeled.npz") as uz:
            tu_cand = uz["candidates"].astype(np.int64)
            tu_pri = uz["posteriors"].astype(np.float32)
        with np.load(L / "rx_v1_test.npz") as tz:
            t_cand = tz["candidates"].astype(np.int64)
            t_post_raw = tz["posteriors"].astype(np.float32)
            t_oop = tz["oop_mass"].astype(np.float32)

        z_lap_test = z_lap_from_gallery(L / "gallery_rx_top40_test.npz", None)
        s_vec_t = np.clip(chosen["scale_base"] - chosen["beta_lap"] * z_lap_test, 0.55, 1.45)
        t_post = adaptive_power_scale(t_post_raw, s_vec_t)

        _, _, u_lnot = run_unlabeled_bp_hetero(
            tu_cand, tu_pri, eu, ev, uy, full_table, known_full,
            s0=chosen["s0"], s1=chosen["s1"], device=device)
        for outer in range(2):
            pred, _, test_lnot, _, _ = target_cavity_predict(
                t_post, t_oop, t_cand, idx1, idx2, known_full, u_lnot, full_table,
                debias=chosen["debias"], cavity_iters=12, use_escape=False, device=device)
            if outer == 0:
                _, _, u_lnot = run_unlabeled_bp_hetero(
                    tu_cand, tu_pri, eu, ev, uy, full_table, known_full,
                    s0=chosen["s0"], s1=chosen["s1"],
                    external_log_avail=test_lnot, device=device)

        pred = np.clip(pred, 0.0, 10.0)
        sub_csv = L / "submission_rx_v3_calibrated.csv"
        with sub_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_id", "result"])
            for row_idx, val in enumerate(pred):
                writer.writerow([f"test-{row_idx:06d}", f"{float(val):.6f}"])
        print(f"Milestone 3 complete -> {sub_csv} (mean={float(pred.mean()):.4f} "
              f"std={float(pred.std()):.4f})", flush=True)

    active = chosen if passed else anchor
    report = {
        "anchor": anchor,
        "chosen": chosen,
        "passed": passed,
        "active": active,
        "submission_csv": str(sub_csv) if sub_csv else None,
    }
    (L / "milestone3_calibration.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
