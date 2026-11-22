"""Milestone 9A (v8 Submission): 224-Exact-Quota Zero-Perturbation Equilibrium Debiaser.

Mathematical Foundation:
  - Forensic test on 181 100%-accurate classes proved:
      * `labeled` (160,925 = 157 * 1025) and `unlabeled` (68,675 = 67 * 1025) have EXACT
        deterministic uniform species quotas (`std_true = 0.000`).
      * `test.csv.gz` (80,000 battles, 160,000 degree-1 images) was sampled without replacement
        from remaining pairs of K_1025 (`std_true ~ 11.4` hypergeometric!).
  - Therefore, fitting a Dirichlet-smoothed Locked-Quota (`p_max >= 0.95` locked!) chemical
    potential `mu_c` on the Exact-Quota pools (`3-Fold OOF + Unlabeled = 224.00 * 1025` images)
    isolates pure asymmetric twin classifier bias (e.g., #993 vs #770) with ZERO hypergeometric
    noise and ZERO perturbation on the 91.35% confident test images!

Strict Gate Protocol:
  1. Export 6-view TTA `logpolar_spec_f{1,2}_rot5.npy` so all 3 folds match `f0` & `test`.
  2. On Fold-0 (100% leak-free: using ONLY `unl_f0only` [67/cls] + `f1,f2 OOF` [104.7/cls]),
     select `(w_oof, kappa, alpha_t, alpha_u, debias_ep)` strictly on `HG_tune` subject to
     passing the Image Brier Gate (`brier < anchor_brier - 5e-5`) and Top-1 Gate.
  3. Require simultaneous gains on BOTH untouched `M7_test` (> 80.1903) and `HG_test` (> 80.1559).
  4. Export `outputs/large/submission_rx_v8_exact_quota_debias.csv` and verify integrity vs v7.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from degradation import clean_impulses_gpu
from evaluate_soft_score import DENOMINATOR, rmse
from export_probs_rot5 import score_model
from image_experiment import NUM_CLASSES, SEED, stratified_folds
from joint_constrained_inference import target_cavity_predict
from milestone1_rx_fusion_and_pack import gather_cand
from milestone3_inference_calibration import (
    adaptive_power_scale,
    run_unlabeled_bp_hetero,
    z_lap_from_gallery,
)
from milestone7_specialist_council import (
    CentroidLogPolarNet,
    compute_physical_regime_zscores,
    fuse_council_moe,
)
from mix_distill_posteriors import fuse_posteriors

L = Path("outputs/large")


def score_fn(val: float) -> float:
    return 100.0 * (1.0 - val / DENOMINATOR)


@torch.inference_mode()
def ensure_logpolar_all_folds_rot5(images_gpu: torch.Tensor, labels_np: np.ndarray) -> dict[int, Path]:
    """Ensure 6-view TTA Fold-0/1/2 OOF probability files exist for `CentroidLogPolarNet`."""
    folds = stratified_folds(labels_np, 3, SEED)
    zf = np.zeros(len(labels_np), dtype=np.uint8)
    out_paths: dict[int, Path] = {}
    for fold in (0, 1, 2):
        p = L / f"logpolar_spec_f{fold}_rot5.npy"
        out_paths[fold] = p
        if p.exists():
            continue
        t0 = time.monotonic()
        v_idx = np.flatnonzero(folds == fold)
        ckpt = L / f"logpolar_spec_gen2_f{fold}_best.pt"
        model = CentroidLogPolarNet().cuda().eval()
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        probs = score_model(model, "logpolar", images_gpu, v_idx, batch_size=256, facing=zf)
        np.save(p, probs.astype(np.float16))
        top1 = float((probs.argmax(1) == labels_np[v_idx]).mean())
        print(f"[9A] Exported {p.name}: shape={probs.shape} top1={top1:.5f} ({time.monotonic()-t0:.1f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    return out_paths


def build_all_folds_v7_oof(raw_labeled_np: np.ndarray, labels_np: np.ndarray, lp_paths: dict[int, Path]):
    """Build exact 6-view v7 Council Top-40 posteriors across all 3 folds (160,925 images)."""
    folds = stratified_folds(labels_np, 3, SEED)
    with np.load(L / "rx_ranker_oof.npz") as z:
        order = np.argsort(z["indices"].astype(np.int64))
        cand_all = z["candidates"][order].astype(np.int64)
        post_ranker = z["posteriors"][order].astype(np.float32)

    rx_mm = np.load(L / "oof_resnext29_dist.npy", mmap_mode="r")
    dn_mm = np.load(L / "oof_dense_gen2.npy", mmap_mode="r")
    cv_mm = np.load(L / "oof_conv_p2imp.npy", mmap_mode="r")

    q_all = np.empty_like(post_ranker, dtype=np.float32)

    # Fold 0 uses exact `rx_v1_fold0_scores.npz` bit-for-bit!
    with np.load(L / "rx_v1_fold0_scores.npz") as fz0:
        v0 = fz0["valid_indices"].astype(np.int64)
        c0 = fz0["candidates"].astype(np.int64)
        b0 = fz0["posteriors"].astype(np.float32)
    z_hf0, z_bd0 = compute_physical_regime_zscores(raw_labeled_np[v0])
    lp0 = gather_cand(np.load(lp_paths[0], mmap_mode="r"), c0)
    q_all[v0] = fuse_council_moe(b0, lp0, None, None, z_hf0, z_bd0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.85)
    cand_all[v0] = c0

    # Folds 1 & 2 use the exact same v1 + v7 fusion recipe!
    for fold in (1, 2):
        vf = np.flatnonzero(folds == fold)
        cf = cand_all[vf]
        pf = post_ranker[vf]
        rx = gather_cand(rx_mm, cf, rows=vf)
        dn = gather_cand(dn_mm, cf, rows=vf)
        stu = 0.65 * rx + 0.35 * dn
        stu /= np.maximum(stu.sum(1, keepdims=True), 1e-8)
        qf = fuse_posteriors(pf, stu, 0.35, "log", 0.75)
        cv = gather_cand(cv_mm, cf, rows=vf)
        qf = fuse_posteriors(qf, cv, 0.09, "log", 0.85)
        z_hff, z_bdf = compute_physical_regime_zscores(raw_labeled_np[vf])
        lpf = gather_cand(np.load(lp_paths[fold], mmap_mode="r"), cf)
        q_all[vf] = fuse_council_moe(qf, lpf, None, None, z_hff, z_bdf, 0.05, 0.05, 0.0, 0.0, 0.0, 0.85)

    return cand_all, q_all, folds


def fit_smoothed_exact_quota_bias(
    pools: list[tuple[np.ndarray, np.ndarray, float]],
    kappa: float = 8.0,
    iters: int = 28,
    lock_thr: float = 0.95,
) -> np.ndarray:
    """Fit zero-perturbation chemical potential `mu[1..1025]` across one or more Exact-Quota pools.
    Each pool is `(post, cand_1based, pool_weight)` where every species has exact uniform true count!
    """
    prepared = []
    for post, cand, w_pool in pools:
        if w_pool <= 0.0:
            continue
        N = len(post)
        target = float(N) / 1025.0
        unc = np.clip((lock_thr - post[:, 0]) / 0.45, 0.0, 1.0).astype(np.float64)
        locked = unc == 0.0
        unc_idx = np.flatnonzero(~locked)
        cnt_locked = np.bincount(cand[locked].ravel(), weights=post[locked].ravel(), minlength=1026)
        lq_u = np.log(np.clip(post[unc_idx].astype(np.float64), 1e-12, 1.0))
        prepared.append({
            "w": float(w_pool),
            "target": target,
            "cnt_locked": cnt_locked,
            "c_u": cand[unc_idx],
            "lq_u": lq_u,
            "u_w": unc[unc_idx, None],
            "q_u": post[unc_idx].astype(np.float64).copy(),
        })

    mu = np.zeros(1026, dtype=np.float64)
    for _ in range(iters):
        tot_resid_target = np.zeros(1026, dtype=np.float64)
        tot_cnt_unc = np.zeros(1026, dtype=np.float64)
        w_sum = 0.0
        for item in prepared:
            w = item["w"]
            cnt_unc = np.bincount(item["c_u"].ravel(), weights=item["q_u"].ravel(), minlength=1026)
            resid_t = np.maximum(item["target"] - item["cnt_locked"], 0.0)
            # Normalize each pool to per-100-per-class units so kappa has consistent scale!
            scale_norm = 100.0 / item["target"]
            tot_resid_target += w * resid_t * scale_norm
            tot_cnt_unc += w * cnt_unc * scale_norm
            w_sum += w

        tot_resid_target /= max(w_sum, 1e-8)
        tot_cnt_unc /= max(w_sum, 1e-8)
        step = 0.45 * np.log((tot_resid_target + kappa) / (tot_cnt_unc + kappa))
        mu = np.clip(mu + step, -2.5, 2.5)
        mu[0] = 0.0

        for item in prepared:
            lg = item["lq_u"] + item["u_w"] * mu[item["c_u"]]
            lg -= lg.max(axis=1, keepdims=True)
            p = np.exp(lg)
            p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
            item["q_u"] = p

    return mu.astype(np.float32)


def apply_zero_perturbation_bias(
    q_cand: np.ndarray,
    cand_1based: np.ndarray,
    mu: np.ndarray,
    alpha: float,
    lock_thr: float = 0.95,
) -> np.ndarray:
    if abs(alpha) < 1e-8:
        return q_cand.copy()
    unc = np.clip((lock_thr - q_cand[:, 0]) / 0.45, 0.0, 1.0).astype(np.float32)
    out = q_cand.copy()
    m = unc > 0.0
    lq = np.log(np.clip(out[m].astype(np.float64), 1e-12, 1.0))
    lq += (alpha * unc[m, None]) * mu[cand_1based[m]]
    lq -= lq.max(axis=1, keepdims=True)
    p = np.exp(lq)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    out[m] = p.astype(np.float32)
    return out


def main() -> None:
    t_start = time.monotonic()
    device = "cuda"

    raw_labeled_np = np.load("labeled_32x32.npy", mmap_mode="r")
    raw_unl_np = np.load("unl/test_32x32.npy", mmap_mode="r")
    images_gpu = clean_impulses_gpu(torch.as_tensor(np.asarray(raw_labeled_np).copy(), device=device))
    labels_np = np.load("labeled_ids.npy").astype(np.int64)

    lp_paths = ensure_logpolar_all_folds_rot5(images_gpu, labels_np)
    del images_gpu
    torch.cuda.empty_cache()

    cand_all, q_oof_all, folds_lbl = build_all_folds_v7_oof(raw_labeled_np, labels_np, lp_paths)
    f12_mask = folds_lbl != 0
    print(f"[9A] 3-Fold v7 OOF ready: overall top1="
          f"{float((cand_all[np.arange(len(labels_np)), q_oof_all.argmax(1)] - 1 == labels_np).mean()):.5f}", flush=True)

    # Load Fold-0 battle benchmark & active tables
    m2 = json.loads((L / "milestone2_pair_table_v4.json").read_text())
    nof0_tab = np.load(Path(m2["active_nof0_table"]))
    full_tab = np.load(Path(m2["active_full_table"]))
    m3 = json.loads((L / "milestone3_calibration.json").read_text())["active"]
    s0, s1, debias_orig, beta_lap = float(m3["s0"]), float(m3["s1"]), float(m3["debias"]), float(m3["beta_lap"])

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

    norm_rows = lambda m: m / np.maximum(m.sum(1, keepdims=True), 1e-8)
    with np.load(L / "rx_v1_fold0_scores.npz") as fz:
        v_idx = fz["valid_indices"].astype(np.int64)
        cand_f0 = fz["candidates"].astype(np.int64)
        q_base_f0 = fz["posteriors"].astype(np.float32)
        oop_f0 = fz["oop_mass"].astype(np.float32)
    with np.load(L / "rx_v1_unlabeled_f0only.npz") as uz:
        u_cand_f0 = uz["candidates"].astype(np.int64)
        u_base_f0 = uz["posteriors"].astype(np.float32)

    z_hf_f0, z_bd_f0 = compute_physical_regime_zscores(raw_labeled_np[v_idx])
    z_hf_u, z_bd_u = compute_physical_regime_zscores(raw_unl_np)
    q_lp_f0 = norm_rows(gather_cand(np.load(lp_paths[0], mmap_mode="r"), cand_f0))
    u_lp_f0 = norm_rows(gather_cand(np.load(L / "unl_f0_logpolar_spec.npy", mmap_mode="r"), u_cand_f0))

    qm_v7 = fuse_council_moe(q_base_f0, q_lp_f0, None, None, z_hf_f0, z_bd_f0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.85)
    um_v7 = fuse_council_moe(u_base_f0, u_lp_f0, None, None, z_hf_u, z_bd_u, 0.05, 0.05, 0.0, 0.0, 0.0, 0.85)
    z_lap_f0 = z_lap_from_gallery(L / "gallery_rx_top40_f0.npz", v_idx)

    pos = np.full(len(paths), -1, dtype=np.int32)
    pos[v_idx] = np.arange(len(v_idx))
    one, two = pos[f1[pair_rows]], pos[f2[pair_rows]]
    actual = y[pair_rows]
    ta, tb = a[pair_rows], b[pair_rows]
    in_c_f0 = (cand_f0 == true[v_idx, None]).astype(np.float32)

    # Degree-1 Hypergeometric Fold-0 benchmark (1:1 structural match to test.csv.gz!)
    E = len(pair_rows)
    ep_cand = np.concatenate([cand_f0[one], cand_f0[two]], axis=0)
    ep_oop = np.concatenate([oop_f0[one], oop_f0[two]], axis=0)
    ep_one = np.arange(E, dtype=np.int32)
    ep_two = np.arange(E, 2 * E, dtype=np.int32)
    ep_zl = np.concatenate([z_lap_f0[one], z_lap_f0[two]], axis=0)

    # Locked v7 Anchor on BOTH benchmarks
    _, _, ul0 = run_unlabeled_bp_hetero(u_cand_f0, um_v7, eu, ev, uy, nof0_tab, known_allowed, s0=s0, s1=s1, device=device)
    _, _, tl0, _, _ = target_cavity_predict(qm_v7, oop_f0, cand_f0, one, two, known_allowed, ul0, nof0_tab, debias=debias_orig, device=device)
    _, _, ul1_anchor = run_unlabeled_bp_hetero(u_cand_f0, um_v7, eu, ev, uy, nof0_tab, known_allowed, s0=s0, s1=s1, external_log_avail=tl0, device=device)

    qs_v7 = adaptive_power_scale(qm_v7, np.clip(1.0 - beta_lap * z_lap_f0, 0.55, 1.45))
    pr_m7_anc, tp_m7_anc, _, _, _ = target_cavity_predict(
        qs_v7, oop_f0, cand_f0, one, two, known_allowed, ul1_anchor, nof0_tab,
        debias=debias_orig, cavity_iters=12, device=device,
    )
    ep_qs_v7 = adaptive_power_scale(np.concatenate([qm_v7[one], qm_v7[two]], 0), np.clip(1.0 - beta_lap * ep_zl, 0.55, 1.45))
    pr_hg_anc, tp_hg_anc, _, _, _ = target_cavity_predict(
        ep_qs_v7, ep_oop, ep_cand, ep_one, ep_two, known_allowed, ul1_anchor, nof0_tab,
        debias=debias_orig, cavity_iters=12, device=device,
    )

    anc = {
        "img_top1": float((cand_f0[np.arange(len(v_idx)), qm_v7.argmax(1)] == true[v_idx]).mean()),
        "img_brier": float(((qm_v7 - in_c_f0) ** 2).sum(1).mean()),
        "m7_tune": score_fn(rmse(pr_m7_anc[:split], actual[:split])),
        "m7_test": score_fn(rmse(pr_m7_anc[split:], actual[split:])),
        "hg_tune": score_fn(rmse(pr_hg_anc[:split], actual[:split])),
        "hg_test": score_fn(rmse(pr_hg_anc[split:], actual[split:])),
    }
    print(f"[9A Anchor v7] top1={anc['img_top1']:.5f} brier={anc['img_brier']:.5f} | "
          f"M7_tune={anc['m7_tune']:.4f} M7_test={anc['m7_test']:.4f} | "
          f"HG_tune={anc['hg_tune']:.4f} HG_test={anc['hg_test']:.4f}", flush=True)

    # Leak-free Fold-0 grid search: select ONLY on `hg_tune` subject to Image Brier & Top-1 gates!
    # Notice: `q_oof_all[f12_mask]` (107,283 exact-quota images from folds 1 & 2) + `um_v7` (68,675 exact-67 images)
    # uses ZERO Fold-0 labels and ZERO Fold-0 validation images!
    trials = []
    for w_oof in (0.0, 0.5, 1.0):
        for kappa in (6.0, 9.0, 12.0):
            pools_f0 = [(um_v7, u_cand_f0, 1.0)]
            if w_oof > 0.0:
                pools_f0.append((q_oof_all[f12_mask], cand_all[f12_mask], w_oof))
            mu_f0 = fit_smoothed_exact_quota_bias(pools_f0, kappa=kappa, iters=28, lock_thr=0.95)

            for alpha_u in (0.0, 0.20):
                if alpha_u > 0.0:
                    um_deb = apply_zero_perturbation_bias(um_v7, u_cand_f0, mu_f0, alpha=alpha_u)
                    _, _, u0 = run_unlabeled_bp_hetero(u_cand_f0, um_deb, eu, ev, uy, nof0_tab, known_allowed, s0=s0, s1=s1, device=device)
                    _, _, t0, _, _ = target_cavity_predict(qm_v7, oop_f0, cand_f0, one, two, known_allowed, u0, nof0_tab, debias=debias_orig, device=device)
                    _, _, ul_cur = run_unlabeled_bp_hetero(u_cand_f0, um_deb, eu, ev, uy, nof0_tab, known_allowed, s0=s0, s1=s1, external_log_avail=t0, device=device)
                else:
                    ul_cur = ul1_anchor

                for alpha_t in (0.18, 0.25, 0.32, 0.40):
                    qd = apply_zero_perturbation_bias(qm_v7, cand_f0, mu_f0, alpha=alpha_t)
                    t1 = float((cand_f0[np.arange(len(v_idx)), qd.argmax(1)] == true[v_idx]).mean())
                    br = float(((qd - in_c_f0) ** 2).sum(1).mean())
                    if br > anc["img_brier"] - 5e-5 or t1 < anc["img_top1"] - 1e-5:
                        continue

                    qs_d = adaptive_power_scale(qd, np.clip(1.0 - beta_lap * z_lap_f0, 0.55, 1.45))
                    ep_qd = np.concatenate([qd[one], qd[two]], 0)
                    ep_qs_d = adaptive_power_scale(ep_qd, np.clip(1.0 - beta_lap * ep_zl, 0.55, 1.45))

                    for db_ep in (0.65, 0.90, 1.25):
                        pr_hg, _, _, _, _ = target_cavity_predict(
                            ep_qs_d, ep_oop, ep_cand, ep_one, ep_two, known_allowed, ul_cur, nof0_tab,
                            debias=db_ep, cavity_iters=12, device=device,
                        )
                        pr_m7, _, _, _, _ = target_cavity_predict(
                            qs_d, oop_f0, cand_f0, one, two, known_allowed, ul_cur, nof0_tab,
                            debias=db_ep, cavity_iters=12, device=device,
                        )
                        rec = {
                            "w_oof": w_oof, "kappa": kappa, "alpha_u": alpha_u,
                            "alpha_t": alpha_t, "debias_ep": db_ep,
                            "img_top1": t1, "img_brier": br,
                            "hg_tune": score_fn(rmse(pr_hg[:split], actual[:split])),
                            "hg_test": score_fn(rmse(pr_hg[split:], actual[split:])),
                            "m7_tune": score_fn(rmse(pr_m7[:split], actual[:split])),
                            "m7_test": score_fn(rmse(pr_m7[split:], actual[split:])),
                        }
                        trials.append(rec)

    if not trials:
        raise RuntimeError("No candidate passed strict Image Brier + Top-1 gates!")

    chosen = max(trials, key=lambda r: r["hg_tune"])
    passed = (
        chosen["img_brier"] <= anc["img_brier"] - 5e-5
        and chosen["img_top1"] >= anc["img_top1"] - 1e-5
        and chosen["hg_test"] > anc["hg_test"] + 0.02
        and chosen["m7_test"] > anc["m7_test"] + 0.02
    )
    print(f"\n=== [9A Chosen on HG_tune] passed={passed} ===\n"
          f"  cfg: w_oof={chosen['w_oof']} kappa={chosen['kappa']} alpha_u={chosen['alpha_u']} "
          f"alpha_t={chosen['alpha_t']} debias={chosen['debias_ep']}\n"
          f"  img_top1 : {anc['img_top1']:.5f} -> {chosen['img_top1']:.5f} ({chosen['img_top1']-anc['img_top1']:+.5f})\n"
          f"  img_brier: {anc['img_brier']:.5f} -> {chosen['img_brier']:.5f} ({chosen['img_brier']-anc['img_brier']:+.5f})\n"
          f"  HG_tune  : {anc['hg_tune']:.4f} -> {chosen['hg_tune']:.4f} ({chosen['hg_tune']-anc['hg_tune']:+.4f})\n"
          f"  HG_test  : {anc['hg_test']:.4f} -> {chosen['hg_test']:.4f} ({chosen['hg_test']-anc['hg_test']:+.4f})\n"
          f"  M7_test  : {anc['m7_test']:.4f} -> {chosen['m7_test']:.4f} ({chosen['m7_test']-anc['m7_test']:+.4f})",
          flush=True)

    sub_csv = None
    diag = {}
    if passed:
        with np.load(L / "rx_v7_council_unlabeled.npz") as uz:
            tu_cand = uz["candidates"].astype(np.int64)
            tu_pri_v7 = uz["posteriors"].astype(np.float32)
        with np.load(L / "rx_v7_council_test.npz") as tz:
            t_cand = tz["candidates"].astype(np.int64)
            t_post_v7 = tz["posteriors"].astype(np.float32)
            t_oop = tz["oop_mass"].astype(np.float32)

        # Build Full 224-Exact-Quota (`67` Unlabeled + `157` 3-Fold OOF = 229,600 exact-quota images!)
        pools_full = [(tu_pri_v7, tu_cand, 1.0)]
        if chosen["w_oof"] > 0.0:
            pools_full.append((q_oof_all, cand_all, chosen["w_oof"]))
        mu_full = fit_smoothed_exact_quota_bias(pools_full, kappa=chosen["kappa"], iters=28, lock_thr=0.95)

        tu_pri_v8 = apply_zero_perturbation_bias(tu_pri_v7, tu_cand, mu_full, alpha=chosen["alpha_u"])
        t_post_v8_raw = apply_zero_perturbation_bias(t_post_v7, t_cand, mu_full, alpha=chosen["alpha_t"])

        # Strict Zero-Perturbation Invariant Check on `p_max >= 0.95`!
        conf_mask = t_post_v7[:, 0] >= 0.95
        zero_pert_err = float(np.abs(t_post_v8_raw[conf_mask] - t_post_v7[conf_mask]).max())
        assert zero_pert_err == 0.0, f"Zero-perturbation violated: {zero_pert_err}"

        # Check #993 vs #770 and #99 argmax counts before & after on Test!
        top1_v7 = t_cand[np.arange(len(t_cand)), t_post_v7.argmax(1)]
        top1_v8 = t_cand[np.arange(len(t_cand)), t_post_v8_raw.argmax(1)]
        c_v7 = np.bincount(top1_v7, minlength=1026)
        c_v8 = np.bincount(top1_v8, minlength=1026)
        print(f"[9A Test Twin Audit] Zero-Perturbation on {conf_mask.sum()}/{len(conf_mask)} "
              f"({conf_mask.mean()*100:.2f}%) images: max_diff = {zero_pert_err:.2e}!!\n"
              f"  #993 count: {c_v7[993]} -> {c_v8[993]} | #770 count: {c_v7[770]} -> {c_v8[770]} | "
              f"#99 count: {c_v7[99]} -> {c_v8[99]}", flush=True)

        z_lap_test = z_lap_from_gallery(L / "gallery_rx_top40_test.npz", None)
        t_post_v8 = adaptive_power_scale(t_post_v8_raw, np.clip(1.0 - beta_lap * z_lap_test, 0.55, 1.45))

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

        _, _, u_lnot = run_unlabeled_bp_hetero(
            tu_cand, tu_pri_v8, eu, ev, uy, full_tab, known_full, s0=s0, s1=s1, device=device
        )
        for outer in range(2):
            pred_v8, _, test_lnot, _, _ = target_cavity_predict(
                t_post_v8, t_oop, t_cand, idx1, idx2, known_full, u_lnot, full_tab,
                debias=chosen["debias_ep"], cavity_iters=12, use_escape=False, device=device,
            )
            if outer == 0:
                _, _, u_lnot = run_unlabeled_bp_hetero(
                    tu_cand, tu_pri_v8, eu, ev, uy, full_tab, known_full,
                    s0=s0, s1=s1, external_log_avail=test_lnot, device=device,
                )

        pred_v8 = np.clip(pred_v8, 0.0, 10.0)
        np.savez_compressed(
            L / "rx_v8_debias_test.npz",
            candidates=t_cand.astype(np.uint16),
            posteriors=t_post_v8_raw.astype(np.float32),
            oop_mass=t_oop.astype(np.float32),
            mu_full=mu_full,
        )
        np.savez_compressed(
            L / "rx_v8_debias_unlabeled.npz",
            candidates=tu_cand.astype(np.uint16),
            posteriors=tu_pri_v8.astype(np.float32),
        )

        sub_csv = L / "submission_rx_v8_exact_quota_debias.csv"
        with sub_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_id", "result"])
            for row_idx, val in enumerate(pred_v8):
                writer.writerow([f"test-{row_idx:06d}", f"{float(val):.6f}"])

        # Compare directly against locked 81.04776 v7 submission!
        v7_df = pd.read_csv(L / "submission_rx_v7_specialist_council.csv")
        v7_vals = v7_df["result"].to_numpy(np.float64)
        diff = pred_v8.astype(np.float64) - v7_vals
        both_conf = conf_mask[idx1] & conf_mask[idx2]
        diag = {
            "rows": int(len(pred_v8)),
            "nan_count": int(np.isnan(pred_v8).sum()),
            "min": float(pred_v8.min()),
            "max": float(pred_v8.max()),
            "mean": float(pred_v8.mean()),
            "std": float(pred_v8.std()),
            "corr_with_v7": float(np.corrcoef(pred_v8, v7_vals)[0, 1]),
            "rms_diff_vs_v7": float(np.sqrt(np.mean(diff ** 2))),
            "rms_diff_on_both_confident_battles": float(np.sqrt(np.mean(diff[both_conf] ** 2))),
            "frac_battles_within_0p01_of_v7": float((np.abs(diff) < 0.01).mean()),
            "count_993_before_after": [int(c_v7[993]), int(c_v8[993])],
            "count_770_before_after": [int(c_v7[770]), int(c_v8[770])],
            "count_99_before_after": [int(c_v7[99]), int(c_v8[99])],
        }
        print("[9A Submission Audit vs v7 (81.04776)]:", json.dumps(diag, indent=2), flush=True)

        (L / "submission_latest_best.csv").write_bytes(sub_csv.read_bytes())
        Path("submission_latest_best.csv").write_bytes(sub_csv.read_bytes())
        print(f"[9A] Locked new active submission -> {sub_csv} (elapsed {time.monotonic()-t_start:.1f}s)", flush=True)

    out_report = {
        "anchor": anc,
        "chosen": chosen,
        "passed": passed,
        "submission_csv": str(sub_csv) if sub_csv else None,
        "diagnostics": diag,
    }
    (L / "milestone9a_exact_quota_debias.json").write_text(json.dumps(out_report, indent=2) + "\n")


if __name__ == "__main__":
    main()
