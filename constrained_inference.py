"""Global constrained graphical inference for Pokewars battle score prediction.

Combines four structural properties of the competition data-generating process:
1. Multiclass NLL calibration of top-20 candidate ranker + WRN posteriors.
2. Bayes class-prior debiasing q(c | x) / mean_q(c) before applying pair priors.
3. Unordered-pair exclusion across known battles and constrained-BP unlabeled
   battles (67 images/species Sinkhorn + pairwise battle likelihood + internal
   pair-exclusivity across the 79,200 unlabeled battles).
4. Iterative Bethe cavity message passing enforcing unordered-pair exclusivity
   across all target battles simultaneously.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from evaluate_soft_score import DENOMINATOR, rmse
from image_experiment import SEED, stratified_folds


def fit_multiclass_calibrator(candidates: np.ndarray, ranker_scores: np.ndarray,
                              wrn_probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fit [w_logit_rk, w_log_rk, w_log_wrn] minimizing 20-way NLL on OOF rows."""
    in_pool = candidates == labels[:, None]
    keep = in_pool.any(1)
    target = in_pool[keep].argmax(1)
    rk = np.clip(ranker_scores[keep].astype(np.float64), 1e-7, 1.0 - 1e-7)
    wrn = np.clip(wrn_probs[keep].astype(np.float64), 1e-9, 1.0)
    f_logit = np.log(rk / (1.0 - rk))
    f_logr = np.log(rk)
    f_logw = np.log(wrn)

    def nll(weights: np.ndarray) -> float:
        s = weights[0] * f_logit + weights[1] * f_logr + weights[2] * f_logw
        s -= s.max(1, keepdims=True)
        log_q = s - np.log(np.exp(s).sum(1, keepdims=True))
        return -float(np.mean(log_q[np.arange(len(target)), target]))

    res = minimize(nll, np.array([0.21, 0.64, 0.06]), method="Nelder-Mead")
    return res.x.astype(np.float64)


def calibrated_posteriors(ranker_scores: np.ndarray, wrn_probs: np.ndarray,
                          weights: np.ndarray, scale: float = 1.0) -> np.ndarray:
    rk = np.clip(ranker_scores.astype(np.float64), 1e-7, 1.0 - 1e-7)
    wrn = np.clip(wrn_probs.astype(np.float64), 1e-9, 1.0)
    s = scale * (weights[0] * np.log(rk / (1.0 - rk))
                 + weights[1] * np.log(rk)
                 + weights[2] * np.log(wrn))
    s -= s.max(1, keepdims=True)
    p = np.exp(s)
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


def merge_unordered_log_not(edge_probs: torch.Tensor, cand1: torch.Tensor,
                            cand2: torch.Tensor,
                            clip_max: float = 0.99999) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum unordered-pair usage across battles and return exact per-battle self terms.

    Merges (a, b) and (b, a) within the same battle before computing log(1 - P_e({a, b})),
    avoiding self-inhibition when two images share candidate species.
    """
    dev = edge_probs.device
    E, K, _ = edge_probs.shape
    pmin = torch.minimum(cand1[:, :, None], cand2[:, None, :])
    pmax = torch.maximum(cand1[:, :, None], cand2[:, None, :])
    pk = (pmin * 1026 + pmax).reshape(E, K * K)
    ord_k = pk.argsort(dim=1)
    inv_k = ord_k.argsort(dim=1)
    pk_s = torch.gather(pk, 1, ord_k)
    dup = torch.zeros_like(pk_s, dtype=torch.bool)
    dup[:, 1:] = pk_s[:, 1:] == pk_s[:, :-1]

    ws = torch.gather(edge_probs.reshape(E, K * K), 1, ord_k)
    wm = ws.clone()
    wm[:, :-1] = torch.where(dup[:, 1:], ws[:, :-1] + ws[:, 1:], ws[:, :-1])
    wm_uniq = torch.where(dup, torch.zeros_like(wm), wm)
    lnot_uniq = torch.log(1.0 - wm_uniq.clamp(0.0, clip_max))

    tot_flat = torch.zeros(1026 * 1026, device=dev)
    tot_flat.scatter_add_(0, pk_s.reshape(-1), lnot_uniq.reshape(-1))
    tot = tot_flat.reshape(1026, 1026)
    tot = tot + tot.T

    wm_both = wm.clone()
    wm_both[:, 1:] = torch.where(dup[:, 1:], wm[:, :-1], wm_both[:, 1:])
    self_lnot = torch.gather(torch.log(1.0 - wm_both.clamp(0.0, clip_max)), 1, inv_k)
    return tot, self_lnot.reshape(E, K, K)


@torch.no_grad()
def constrained_unlabeled_bp(cand: np.ndarray, pri: np.ndarray, eu: np.ndarray,
                             ev: np.ndarray, y_edge: np.ndarray, table: np.ndarray,
                             known_allowed: np.ndarray, target_per_class: float = 67.0,
                             iters: int = 25, device: str = "cuda"):
    """BP on unlabeled battles with known-pair exclusion, 67/class Sinkhorn, and cavity."""
    N, K = cand.shape
    E = len(eu)
    cand_t = torch.as_tensor(cand, device=device, dtype=torch.long)
    eu_t = torch.as_tensor(eu, device=device, dtype=torch.long)
    ev_t = torch.as_tensor(ev, device=device, dtype=torch.long)
    c_u, c_v = cand_t[eu_t], cand_t[ev_t]
    T_t = torch.as_tensor(table, device=device, dtype=torch.float32)
    M_t = torch.as_tensor(known_allowed, device=device, dtype=torch.float32)

    pred = T_t[c_u[:, :, None], c_v[:, None, :]]
    obs = torch.as_tensor(y_edge, device=device, dtype=torch.float32)[:, None, None]
    psi = (0.95 * torch.exp(-0.5 * ((obs - pred) / 0.5) ** 2)
           / (np.sqrt(2.0 * np.pi) * 0.5) + 0.05 / 10.0)
    mask_uv = M_t[c_u[:, :, None], c_v[:, None, :]]
    base_log_psi = torch.log((psi * (mask_uv + 1e-6 * (1.0 - mask_uv))).clamp_min(1e-30))

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
            tot_lnot, self_lnot = merge_unordered_log_not(ep, c_u, c_v, clip_max=0.9999)
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
    tot_lnot, _ = merge_unordered_log_not(ep, c_u, c_v, clip_max=0.99999)
    log_avail = np.where(known_allowed > 0, tot_lnot.cpu().numpy(), -30.0).astype(np.float32)
    np.fill_diagonal(log_avail, -30.0)
    return node_post.cpu().numpy(), ep.cpu().numpy(), log_avail


@torch.no_grad()
def cavity_pair_predict(q_img: np.ndarray, cand: np.ndarray, idx1: np.ndarray,
                        idx2: np.ndarray, log_avail: np.ndarray | None,
                        table: np.ndarray, debias: float = 1.0,
                        cavity_iters: int = 12, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """Predict expected battle scores under pair-availability and Bethe cavity constraints."""
    if debias != 0.0:
        mean_q = np.bincount(cand.ravel(), weights=q_img.ravel(), minlength=1026) / len(q_img)
        mean_q[0] = 1.0
        q = q_img * (np.clip(mean_q, 1e-5, None)[cand] ** (-debias))
        q = (q / q.sum(1, keepdims=True)).astype(np.float32)
    else:
        q = q_img.astype(np.float32)

    ids1, ids2 = cand[idx1], cand[idx2]
    qt = torch.as_tensor(q, device=device)
    c1_t = torch.as_tensor(ids1, device=device, dtype=torch.long)
    c2_t = torch.as_tensor(ids2, device=device, dtype=torch.long)
    T_sub = torch.as_tensor(table[ids1[:, :, None], ids2[:, None, :]],
                            device=device, dtype=torch.float32)
    w0 = qt[idx1, :, None] * qt[idx2, None, :]
    if log_avail is None:
        w = w0
    else:
        base_la = torch.as_tensor(log_avail[ids1[:, :, None], ids2[:, None, :]], device=device)
        w = w0 * torch.exp(base_la)
        sn = w.sum((1, 2), keepdim=True)
        w = torch.where(sn > 1e-25, w / sn.clamp_min(1e-25), w0)
        for _ in range(cavity_iters):
            tot_ln, self_ln = merge_unordered_log_not(w, c1_t, c2_t, clip_max=0.99999)
            cav = (base_la + tot_ln[c1_t[:, :, None], c2_t[:, None, :]] - self_ln).clamp(-30.0, 0.0)
            wn = w0 * torch.exp(cav)
            sn = wn.sum((1, 2), keepdim=True)
            wn = torch.where(sn > 1e-25, wn / sn.clamp_min(1e-25), w0)
            w = 0.5 * w + 0.5 * wn

    pred = (w * T_sub).sum((1, 2)).cpu().numpy()
    K = cand.shape[1]
    best_flat = w.reshape(len(w), K * K).argmax(1).cpu().numpy()
    top_pairs = np.stack([ids1[np.arange(len(w)), best_flat // K],
                          ids2[np.arange(len(w)), best_flat % K]], axis=1)
    return pred, top_pairs


def load_unlabeled_prior(path: Path, weights: np.ndarray | None, scale: float) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".npz":
        with np.load(path) as z:
            cand = z["candidates"].astype(np.int64)
            if "ranker_scores" in z and weights is not None:
                pri = calibrated_posteriors(z["ranker_scores"], z["wrn_probs"], weights, scale=scale)
            else:
                pri = z["probabilities"].astype(np.float32)
                pri /= pri.sum(1, keepdims=True)
        return cand, pri
    frame = pd.read_csv(path)
    cand = np.array([json.loads(v) for v in frame.topk_pokemon_ids], dtype=np.int64)
    pri = np.array([json.loads(v) for v in frame.topk_probabilities], dtype=np.float32)
    pri /= pri.sum(1, keepdims=True)
    return cand, pri


def run_fold0(args: argparse.Namespace, device: str) -> None:
    rows = pd.read_csv(args.train_csv)
    identity = {}
    for side in (1, 2):
        for p, v in zip(rows[f"noisy_image{side}"], rows[f"pokemon_id{side}"]):
            if pd.notna(v):
                identity[p] = int(v)
    paths = sorted(identity)
    true = np.array([identity[p] for p in paths], dtype=np.int64)
    folds = stratified_folds(true - 1, 3, SEED)
    idx = pd.Index(paths)
    f1, f2 = idx.get_indexer(rows.noisy_image1), idx.get_indexer(rows.noisy_image2)
    known = (f1 >= 0) & (f2 >= 0)
    f1, f2 = f1[known], f2[known]
    a, b = true[f1], true[f2]
    y = rows.result.to_numpy(np.float32)[known]
    pair_f0 = (folds[f1] == 0) & (folds[f2] == 0)
    rng = np.random.default_rng(3780)
    pair_rows = np.flatnonzero(pair_f0)
    rng.shuffle(pair_rows)
    split = len(pair_rows) // 2

    with np.load(args.oof_npz) as oz:
        keep_f12 = oz["folds"] != 0
        cal_weights = fit_multiclass_calibrator(
            oz["candidates"][keep_f12].astype(np.int64),
            oz["ranker_scores"][keep_f12],
            oz["wrn_probs"][keep_f12],
            oz["labels"][keep_f12].astype(np.int64),
        )
    print("Fold-1+2 OOF multiclass calibration weights:", cal_weights.round(4).tolist(), flush=True)

    table = np.load(args.table)
    known_allowed = np.ones((1026, 1026), dtype=np.float32)
    np.fill_diagonal(known_allowed, 0.0)
    ka, kb = a[~pair_f0], b[~pair_f0]
    known_allowed[ka, kb] = 0.0
    known_allowed[kb, ka] = 0.0
    log_avail_known = np.where(known_allowed > 0, 0.0, -30.0).astype(np.float32)

    unk = rows.pokemon_id1.isna() & rows.pokemon_id2.isna()
    u_idx = pd.Index(args.unlabeled_paths.read_text().split())
    eu = u_idx.get_indexer(rows.noisy_image1[unk])
    ev = u_idx.get_indexer(rows.noisy_image2[unk])
    uy = rows.result[unk].to_numpy(np.float32)
    u_cand, u_pri = load_unlabeled_prior(args.unlabeled_priors, cal_weights, scale=1.0)
    u_node, _, log_avail_both = constrained_unlabeled_bp(
        u_cand, u_pri, eu, ev, uy, table, known_allowed, device=device)
    u_counts = np.bincount(u_cand[np.arange(len(u_node)), u_node.argmax(1)], minlength=1026)[1:]

    with np.load(args.ranker_fold0_scores) as z:
        v_idx = z["valid_indices"]
        cand = z["candidates"].astype(np.int64)
        rk = z["ranker_scores"].astype(np.float32)
        wrn = z["wrn_probs"].astype(np.float32)
    pos = np.full(len(paths), -1, dtype=np.int32)
    pos[v_idx] = np.arange(len(v_idx))
    one, two = pos[f1[pair_rows]], pos[f2[pair_rows]]
    actual = y[pair_rows]
    score_fn = lambda r: 100.0 * (1.0 - r / DENOMINATOR)

    q_raw = rk / np.maximum(rk.sum(1, keepdims=True), 1e-12)
    ablations = {}
    for name, q_use, la, cav_it, db in [
        ("0_indep_raw_top20", q_raw, None, 0, 0.0),
        ("1_indep_calibrated", calibrated_posteriors(rk, wrn, cal_weights, 1.0), None, 0, 0.0),
        ("2_known_pair_mask", calibrated_posteriors(rk, wrn, cal_weights, 1.0), log_avail_known, 0, 1.0),
        ("3_plus_unlabeled_bp_mask", calibrated_posteriors(rk, wrn, cal_weights, 1.0), log_avail_both, 0, 1.0),
    ]:
        pred, top_p = cavity_pair_predict(q_use, cand, one, two, la, table, debias=db,
                                          cavity_iters=cav_it, device=device)
        both_ok = float(((top_p[:, 0] == a[pair_rows]) & (top_p[:, 1] == b[pair_rows]))[split:].mean())
        ablations[name] = {"tune_score": score_fn(rmse(pred[:split], actual[:split])),
                           "test_score": score_fn(rmse(pred[split:], actual[split:])),
                           "test_rmse": rmse(pred[split:], actual[split:]),
                           "test_both_correct": both_ok}
        print(name, json.dumps(ablations[name]), flush=True)

    trials = []
    for scale in (0.80, 0.85, 0.90, 0.95, 1.00):
        q_s = calibrated_posteriors(rk, wrn, cal_weights, scale=scale)
        pred, top_p = cavity_pair_predict(q_s, cand, one, two, log_avail_both, table,
                                          debias=1.0, cavity_iters=12, device=device)
        both_ok = float(((top_p[:, 0] == a[pair_rows]) & (top_p[:, 1] == b[pair_rows]))[split:].mean())
        trials.append({"scale": scale,
                       "tune_rmse": rmse(pred[:split], actual[:split]),
                       "test_rmse": rmse(pred[split:], actual[split:]),
                       "tune_score": score_fn(rmse(pred[:split], actual[:split])),
                       "test_score": score_fn(rmse(pred[split:], actual[split:])),
                       "test_both_correct": both_ok})
        print("cavity_grid", json.dumps(trials[-1]), flush=True)

    selected = min(trials, key=lambda item: item["tune_rmse"])
    report = {
        "calibration_weights": cal_weights.tolist(),
        "unlabeled_bp_exact67_fraction": float((u_counts == 67).mean()),
        "ablations": ablations,
        "cavity_scale_grid": trials,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("SELECTED", json.dumps(selected), flush=True)


def run_submission(args: argparse.Namespace, device: str) -> None:
    with np.load(args.oof_npz) as oz:
        cal_weights = fit_multiclass_calibrator(
            oz["candidates"].astype(np.int64),
            oz["ranker_scores"],
            oz["wrn_probs"],
            oz["labels"].astype(np.int64),
        )
    print("Full 3-fold OOF multiclass calibration weights:", cal_weights.round(4).tolist(), flush=True)

    with np.load(args.battles) as bz:
        ka, kb = bz["a"].astype(np.int64), bz["b"].astype(np.int64)
    known_allowed = np.ones((1026, 1026), dtype=np.float32)
    np.fill_diagonal(known_allowed, 0.0)
    known_allowed[ka, kb] = 0.0
    known_allowed[kb, ka] = 0.0

    rows = pd.read_csv(args.train_csv)
    unk = rows.pokemon_id1.isna() & rows.pokemon_id2.isna()
    u_idx = pd.Index(args.unlabeled_paths.read_text().split())
    eu = u_idx.get_indexer(rows.noisy_image1[unk])
    ev = u_idx.get_indexer(rows.noisy_image2[unk])
    uy = rows.result[unk].to_numpy(np.float32)

    table = np.load(args.table)
    u_cand, u_pri = load_unlabeled_prior(args.unlabeled_priors, cal_weights, scale=1.0)
    u_node, u_edge, log_avail = constrained_unlabeled_bp(
        u_cand, u_pri, eu, ev, uy, table, known_allowed, device=device)

    K_u = u_cand.shape[1]
    best_e = u_edge.reshape(len(eu), K_u * K_u).argmax(1)
    pa = u_cand[eu, best_e // K_u]
    pb = u_cand[ev, best_e % K_u]
    pweight = u_edge.reshape(len(eu), K_u * K_u)[np.arange(len(eu)), best_e]
    u_counts = np.bincount(u_cand[np.arange(len(u_node)), u_node.argmax(1)], minlength=1026)[1:]
    u_unique = len(set(zip(np.minimum(pa, pb).tolist(), np.maximum(pa, pb).tolist())))
    print(f"Unlabeled constrained BP: exact67={(u_counts == 67).mean():.3f}, "
          f"min={u_counts.min()}, max={u_counts.max()}, unique_pairs={u_unique}/{len(eu)}, "
          f"mean_edge_conf={pweight.mean():.4f}", flush=True)

    if args.save_pseudo_battles is not None:
        args.save_pseudo_battles.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_pseudo_battles,
                            a=pa.astype(np.uint16), b=pb.astype(np.uint16),
                            y=uy.astype(np.float32), weight=pweight.astype(np.float32))

    test_paths = (args.data_dir / "test_paths.txt").read_text().splitlines()
    path_to_idx = pd.Index(test_paths)
    test_df = pd.read_csv(args.test_csv)
    idx1 = path_to_idx.get_indexer(test_df.noisy_image1)
    idx2 = path_to_idx.get_indexer(test_df.noisy_image2)
    if np.any(idx1 < 0) or np.any(idx2 < 0):
        raise ValueError("Test battle image missing from test_paths.txt")

    with np.load(args.test_npz) as tz:
        t_cand = tz["candidates"].astype(np.int64)
        t_rk = tz["ranker_scores"].astype(np.float32)
        t_wrn = tz["wrn_probs"].astype(np.float32)
    q_test = calibrated_posteriors(t_rk, t_wrn, cal_weights, scale=args.scale)
    pred, top_pairs = cavity_pair_predict(
        q_test, t_cand, idx1, idx2, log_avail, table,
        debias=args.debias, cavity_iters=args.cavity_iters, device=device)
    pred = np.clip(pred, 0.0, 10.0)

    t_unique = len(set(zip(np.minimum(top_pairs[:, 0], top_pairs[:, 1]).tolist(),
                           np.maximum(top_pairs[:, 0], top_pairs[:, 1]).tolist())))
    known_set = set(zip(np.minimum(ka, kb).tolist(), np.maximum(ka, kb).tolist()))
    test_set = set(zip(np.minimum(top_pairs[:, 0], top_pairs[:, 1]).tolist(),
                       np.maximum(top_pairs[:, 0], top_pairs[:, 1]).tolist()))
    collisions_known = len(known_set & test_set)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "result"])
        for row_idx, value in enumerate(pred):
            writer.writerow([f"test-{row_idx:06d}", f"{float(value):.6f}"])

    meta = {
        "rows": len(pred),
        "table": str(args.table),
        "test_npz": str(args.test_npz),
        "unlabeled_priors": str(args.unlabeled_priors),
        "calibration_weights": cal_weights.tolist(),
        "scale": args.scale,
        "debias": args.debias,
        "cavity_iters": args.cavity_iters,
        "unlabeled_bp": {
            "exact67_fraction": float((u_counts == 67).mean()),
            "count_min": int(u_counts.min()),
            "count_max": int(u_counts.max()),
            "unique_unordered_pairs": u_unique,
            "mean_edge_confidence": float(pweight.mean()),
        },
        "test_diagnostics": {
            "unique_top1_unordered_pairs": t_unique,
            "collisions_with_known_battles": collisions_known,
            "prediction_mean": float(pred.mean()),
            "prediction_std": float(pred.std()),
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("evaluate-fold0", "predict-submission"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv.gz"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv.gz"))
    parser.add_argument("--battles", type=Path, default=Path("data/known_battles.npz"))
    parser.add_argument("--unlabeled-paths", type=Path, default=Path("data/unlabeled_paths.txt"))
    parser.add_argument("--oof-npz", type=Path, default=Path("outputs/large/meta_rerank_v2_top20_oof.npz"))
    parser.add_argument("--ranker-fold0-scores", type=Path,
                        default=Path("outputs/large/meta_v2_top20_fold0_scores.npz"))
    parser.add_argument("--unlabeled-priors", type=Path, required=True)
    parser.add_argument("--test-npz", type=Path, default=Path("outputs/large/test_meta_v2_top20.npz"))
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.90)
    parser.add_argument("--debias", type=float, default=1.0)
    parser.add_argument("--cavity-iters", type=int, default=12)
    parser.add_argument("--save-pseudo-battles", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "evaluate-fold0":
        run_fold0(args, device)
    else:
        run_submission(args, device)


if __name__ == "__main__":
    main()
