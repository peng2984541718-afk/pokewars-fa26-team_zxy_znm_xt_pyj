"""Bidirectional joint graphical inference (Unlabeled <-> Test) with top-K=40
candidates, chunked GPU cavity message passing, Bayesian out-of-pool escape,
and transductive pseudo-label bank export for Phase 2 backbone training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_soft_score import DENOMINATOR, rmse
from image_experiment import SEED, stratified_folds


@torch.no_grad()
def chunked_unordered_log_not(edge_probs: torch.Tensor, cand1: torch.Tensor,
                              cand2: torch.Tensor, clip_max: float = 0.99999,
                              chunk_size: int = 8000) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked version of merge_unordered_log_not supporting K=40 (1600 pairs/battle)."""
    dev = edge_probs.device
    E, K, _ = edge_probs.shape
    tot_flat = torch.zeros(1026 * 1026, device=dev, dtype=torch.float32)
    self_lnot_all = torch.empty((E, K, K), device=dev, dtype=torch.float32)

    for start in range(0, E, chunk_size):
        end = min(start + chunk_size, E)
        ep = edge_probs[start:end]
        c1 = cand1[start:end]
        c2 = cand2[start:end]
        b = end - start
        pmin = torch.minimum(c1[:, :, None], c2[:, None, :])
        pmax = torch.maximum(c1[:, :, None], c2[:, None, :])
        pk = (pmin * 1026 + pmax).reshape(b, K * K)
        ord_k = pk.argsort(dim=1)
        inv_k = ord_k.argsort(dim=1)
        pk_s = torch.gather(pk, 1, ord_k)
        dup = torch.zeros_like(pk_s, dtype=torch.bool)
        dup[:, 1:] = pk_s[:, 1:] == pk_s[:, :-1]

        ws = torch.gather(ep.reshape(b, K * K), 1, ord_k)
        wm = ws.clone()
        wm[:, :-1] = torch.where(dup[:, 1:], ws[:, :-1] + ws[:, 1:], ws[:, :-1])
        wm_uniq = torch.where(dup, torch.zeros_like(wm), wm)
        lnot_uniq = torch.log(1.0 - wm_uniq.clamp(0.0, clip_max))
        tot_flat.scatter_add_(0, pk_s.reshape(-1), lnot_uniq.reshape(-1))

        wm_both = wm.clone()
        wm_both[:, 1:] = torch.where(dup[:, 1:], wm[:, :-1], wm_both[:, 1:])
        self_ln = torch.gather(torch.log(1.0 - wm_both.clamp(0.0, clip_max)), 1, inv_k)
        self_lnot_all[start:end] = self_ln.reshape(b, K, K)

    tot = tot_flat.reshape(1026, 1026)
    tot = tot + tot.T
    return tot, self_lnot_all


def power_scale_posteriors(post: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        p = post.astype(np.float32)
        return p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    lp = scale * np.log(np.clip(post.astype(np.float64), 1e-12, 1.0))
    lp -= lp.max(1, keepdims=True)
    p = np.exp(lp)
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


@torch.no_grad()
def run_unlabeled_bp(cand: np.ndarray, pri: np.ndarray, eu: np.ndarray, ev: np.ndarray,
                     y_edge: np.ndarray, table: np.ndarray, known_allowed: np.ndarray,
                     external_log_avail: np.ndarray | None = None,
                     target_per_class: float = 67.0, iters: int = 24,
                     device: str = "cuda"):
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


@torch.no_grad()
def target_cavity_predict(q_img: np.ndarray, oop_mass: np.ndarray, cand: np.ndarray,
                          idx1: np.ndarray, idx2: np.ndarray,
                          known_allowed: np.ndarray, unl_lnot: np.ndarray | None,
                          table: np.ndarray, debias: float = 1.0,
                          cavity_iters: int = 12, use_escape: bool = False,
                          device: str = "cuda"):
    """Run Bethe cavity on target battles with Bayesian out-of-pool escape."""
    if debias != 0.0:
        mean_q = np.bincount(cand.ravel(), weights=q_img.ravel(), minlength=1026) / len(q_img)
        mean_q[0] = 1.0
        q = q_img * (np.clip(mean_q, 1e-5, None)[cand] ** (-debias))
        q = (q / q.sum(1, keepdims=True)).astype(np.float32)
    else:
        q = q_img.astype(np.float32)

    if unl_lnot is None:
        log_avail = np.where(known_allowed > 0, 0.0, -30.0).astype(np.float32)
    else:
        log_avail = np.where(known_allowed > 0, unl_lnot, -30.0).astype(np.float32)
    np.fill_diagonal(log_avail, -30.0)

    ids1, ids2 = cand[idx1], cand[idx2]
    E, K = ids1.shape
    qt = torch.as_tensor(q, device=device)
    c1_t = torch.as_tensor(ids1, device=device, dtype=torch.long)
    c2_t = torch.as_tensor(ids2, device=device, dtype=torch.long)
    base_la = torch.as_tensor(log_avail[ids1[:, :, None], ids2[:, None, :]], device=device)
    w0 = qt[idx1, :, None] * qt[idx2, None, :]
    w = w0 * torch.exp(base_la)
    sn = w.sum((1, 2), keepdim=True)
    w = torch.where(sn > 1e-25, w / sn.clamp_min(1e-25), w0)

    tot_ln = torch.zeros((1026, 1026), device=device)
    self_ln = torch.zeros_like(w)
    for _ in range(cavity_iters):
        tot_ln, self_ln = chunked_unordered_log_not(w, c1_t, c2_t, clip_max=0.99999)
        cav = (base_la + tot_ln[c1_t[:, :, None], c2_t[:, None, :]] - self_ln).clamp(-30.0, 0.0)
        wn = w0 * torch.exp(cav)
        sn = wn.sum((1, 2), keepdim=True)
        wn = torch.where(sn > 1e-25, wn / sn.clamp_min(1e-25), w0)
        w = 0.5 * w + 0.5 * wn

    # Final unnormalized cavity factor for exact out-of-pool escape weighting
    tot_ln, self_ln = chunked_unordered_log_not(w, c1_t, c2_t, clip_max=0.99999)
    cav = (base_la + tot_ln[c1_t[:, :, None], c2_t[:, None, :]] - self_ln).clamp(-30.0, 0.0)
    T_sub = torch.as_tensor(table[ids1[:, :, None], ids2[:, None, :]], device=device, dtype=torch.float32)

    if use_escape:
        avail_mat = np.exp(np.clip(log_avail + tot_ln.cpu().numpy(), -30.0, 0.0))
        np.fill_diagonal(avail_mat, 0.0)
        mean_avail = float(avail_mat[1:, 1:].sum() / (1025.0 * 1024.0))
        row_sum = np.maximum(avail_mat.sum(1), 1e-6)
        T_row_mean = torch.as_tensor((avail_mat * table).sum(1) / row_sum,
                                     device=device, dtype=torch.float32)
        T_col_mean = 10.0 - T_row_mean

        oop_t = torch.as_tensor(np.clip(oop_mass, 1e-6, 0.5), device=device, dtype=torch.float32)
        pin1 = (1.0 - oop_t[idx1])[:, None]
        pin2 = (1.0 - oop_t[idx2])[:, None]
        q1 = qt[idx1] * pin1
        q2 = qt[idx2] * pin2

        w_in_unnorm = q1[:, :, None] * q2[:, None, :] * torch.exp(cav)
        mass_in = w_in_unnorm.sum((1, 2))
        score_in = (w_in_unnorm * T_sub).sum((1, 2))

        w_1in = mean_avail * oop_t[idx2] * q1.sum(1)
        s_1in = mean_avail * oop_t[idx2] * (q1 * T_row_mean[c1_t]).sum(1)

        w_2in = mean_avail * oop_t[idx1] * q2.sum(1)
        s_2in = mean_avail * oop_t[idx1] * (q2 * T_col_mean[c2_t]).sum(1)

        w_both = mean_avail * oop_t[idx1] * oop_t[idx2]
        denom = (mass_in + w_1in + w_2in + w_both).clamp_min(1e-25)
        pred = ((score_in + s_1in + s_2in + 5.0 * w_both) / denom).cpu().numpy()

        w_norm = w_in_unnorm / mass_in[:, None, None].clamp_min(1e-25)
    else:
        w_norm = w0 * torch.exp(cav)
        w_norm = w_norm / w_norm.sum((1, 2), keepdim=True).clamp_min(1e-25)
        pred = (w_norm * T_sub).sum((1, 2)).cpu().numpy()

    best_flat = w_norm.reshape(E, K * K).argmax(1).cpu().numpy()
    top_pairs = np.stack([ids1[np.arange(E), best_flat // K],
                          ids2[np.arange(E), best_flat % K]], axis=1)
    # Image-level marginal posterior & confidence for transductive bank building
    m1 = w_norm.sum(2).cpu().numpy()
    m2 = w_norm.sum(1).cpu().numpy()
    return pred, top_pairs, tot_ln.cpu().numpy(), m1, m2


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

    table = np.load(args.table)
    known_allowed = np.ones((1026, 1026), dtype=np.float32)
    np.fill_diagonal(known_allowed, 0.0)
    ka, kb = a[~pair_f0], b[~pair_f0]
    known_allowed[ka, kb] = 0.0
    known_allowed[kb, ka] = 0.0

    unk = rows.pokemon_id1.isna() & rows.pokemon_id2.isna()
    u_idx = pd.Index(args.unlabeled_paths.read_text().split())
    eu = u_idx.get_indexer(rows.noisy_image1[unk])
    ev = u_idx.get_indexer(rows.noisy_image2[unk])
    uy = rows.result[unk].to_numpy(np.float32)

    with np.load(args.unlabeled_priors) as uz:
        u_cand = uz["candidates"].astype(np.int64)
        u_pri = uz["posteriors"].astype(np.float32)
    with np.load(args.fold0_scores) as fz:
        v_idx = fz["valid_indices"]
        cand = fz["candidates"].astype(np.int64)
        post = fz["posteriors"].astype(np.float32)
        oop = fz["oop_mass"].astype(np.float32)

    pos = np.full(len(paths), -1, dtype=np.int32)
    pos[v_idx] = np.arange(len(v_idx))
    one, two = pos[f1[pair_rows]], pos[f2[pair_rows]]
    actual = y[pair_rows]
    score_fn = lambda r: 100.0 * (1.0 - r / DENOMINATOR)

    # Bidirectional outer rounds between unlabeled BP and target fold-0 battles
    _, _, unl_lnot = run_unlabeled_bp(u_cand, u_pri, eu, ev, uy, table, known_allowed, device=device)
    _, _, test_lnot, _, _ = target_cavity_predict(
        post, oop, cand, one, two, known_allowed, unl_lnot, table, device=device)
    u_node, _, unl_lnot = run_unlabeled_bp(
        u_cand, u_pri, eu, ev, uy, table, known_allowed, external_log_avail=test_lnot, device=device)
    u_counts = np.bincount(u_cand[np.arange(len(u_node)), u_node.argmax(1)], minlength=1026)[1:]

    trials = []
    for scale in (0.85, 0.90, 0.95, 1.00, 1.05):
        qs = power_scale_posteriors(post, scale)
        pred, top_p, _, _, _ = target_cavity_predict(
            qs, oop, cand, one, two, known_allowed, unl_lnot, table,
            debias=1.0, cavity_iters=12, use_escape=args.use_escape, device=device)
        both_ok = float(((top_p[:, 0] == a[pair_rows]) & (top_p[:, 1] == b[pair_rows]))[split:].mean())
        trials.append({
            "scale": scale,
            "tune_rmse": rmse(pred[:split], actual[:split]),
            "test_rmse": rmse(pred[split:], actual[split:]),
            "tune_score": score_fn(rmse(pred[:split], actual[:split])),
            "test_score": score_fn(rmse(pred[split:], actual[split:])),
            "test_both_correct": both_ok,
        })
        print("trial", json.dumps(trials[-1]), flush=True)

    selected = min(trials, key=lambda item: item["tune_rmse"])
    report = {
        "candidate_k": int(cand.shape[1]),
        "fold0_top1_before_constraints": float((cand[:, 0] == true[v_idx]).mean()),
        "fold0_pool_recall": float((cand == true[v_idx, None]).any(1).mean()),
        "unlabeled_exact67_fraction": float((u_counts == 67).mean()),
        "trials": trials,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("SELECTED", json.dumps(selected), flush=True)


def run_submission(args: argparse.Namespace, device: str) -> None:
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

    test_paths = (args.data_dir / "test_paths.txt").read_text().splitlines()
    path_to_idx = pd.Index(test_paths)
    test_df = pd.read_csv(args.test_csv)
    idx1 = path_to_idx.get_indexer(test_df.noisy_image1)
    idx2 = path_to_idx.get_indexer(test_df.noisy_image2)

    table = np.load(args.table)
    with np.load(args.unlabeled_priors) as uz:
        u_cand = uz["candidates"].astype(np.int64)
        u_pri = uz["posteriors"].astype(np.float32)
    with np.load(args.test_npz) as tz:
        t_cand = tz["candidates"].astype(np.int64)
        t_post = power_scale_posteriors(tz["posteriors"].astype(np.float32), args.scale)
        t_oop = tz["oop_mass"].astype(np.float32)

    # Bidirectional joint inference: Unlabeled <-> Test (2 coupled outer rounds)
    u_node, u_edge, unl_lnot = run_unlabeled_bp(
        u_cand, u_pri, eu, ev, uy, table, known_allowed, device=device)
    for outer in range(2):
        pred, top_pairs, test_lnot, m1, m2 = target_cavity_predict(
            t_post, t_oop, t_cand, idx1, idx2, known_allowed, unl_lnot, table,
            debias=args.debias, cavity_iters=args.cavity_iters, use_escape=args.use_escape, device=device)
        if outer == 0:
            u_node, u_edge, unl_lnot = run_unlabeled_bp(
                u_cand, u_pri, eu, ev, uy, table, known_allowed,
                external_log_avail=test_lnot, device=device)

    pred = np.clip(pred, 0.0, 10.0)
    K_u = u_cand.shape[1]
    best_e = u_edge.reshape(len(eu), K_u * K_u).argmax(1)
    pa = u_cand[eu, best_e // K_u]
    pb = u_cand[ev, best_e % K_u]
    pweight = u_edge.reshape(len(eu), K_u * K_u)[np.arange(len(eu)), best_e]
    u_counts = np.bincount(u_cand[np.arange(len(u_node)), u_node.argmax(1)], minlength=1026)[1:]
    u_unique = len(set(zip(np.minimum(pa, pb).tolist(), np.maximum(pa, pb).tolist())))

    if args.save_pseudo_battles is not None:
        args.save_pseudo_battles.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_pseudo_battles,
                            a=pa.astype(np.uint16), b=pb.astype(np.uint16),
                            y=uy.astype(np.float32), weight=pweight.astype(np.float32))

    if args.save_transductive_bank is not None:
        unl_imgs = np.asarray(np.load(Path("unl") / "test_32x32.npy"))
        test_imgs = np.asarray(np.load(args.data_dir / "test_32x32.npy"))
        u_best = u_node.argmax(1)
        u_conf = u_node[np.arange(len(u_node)), u_best]
        u_lab0 = u_cand[np.arange(len(u_node)), u_best] - 1
        u_keep = u_conf >= args.bank_min_conf

        t_marg = np.zeros_like(t_post)
        t_marg[idx1] = m1
        t_marg[idx2] = m2
        t_best = t_marg.argmax(1)
        t_conf = t_marg[np.arange(len(t_marg)), t_best]
        t_lab0 = t_cand[np.arange(len(t_marg)), t_best] - 1
        t_keep = t_conf >= args.test_bank_min_conf

        bank_imgs = np.concatenate([unl_imgs[u_keep], test_imgs[t_keep]], axis=0)
        bank_labs = np.concatenate([u_lab0[u_keep], t_lab0[t_keep]], axis=0).astype(np.int16)
        bank_conf = np.concatenate([u_conf[u_keep], t_conf[t_keep]], axis=0).astype(np.float32)
        args.save_transductive_bank.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_transductive_bank,
                            images=bank_imgs, labels=bank_labs, confidence=bank_conf)
        print(f"Saved transductive bank {args.save_transductive_bank}: "
              f"unlabeled={int(u_keep.sum())}, test={int(t_keep.sum())}, "
              f"total={len(bank_imgs)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "result"])
        for row_idx, value in enumerate(pred):
            writer.writerow([f"test-{row_idx:06d}", f"{float(value):.6f}"])

    t_unique = len(set(zip(np.minimum(top_pairs[:, 0], top_pairs[:, 1]).tolist(),
                           np.maximum(top_pairs[:, 0], top_pairs[:, 1]).tolist())))
    meta = {
        "rows": len(pred),
        "candidate_k": int(t_cand.shape[1]),
        "scale": args.scale,
        "unlabeled_bp": {
            "exact67_fraction": float((u_counts == 67).mean()),
            "count_min": int(u_counts.min()),
            "count_max": int(u_counts.max()),
            "unique_unordered_pairs": u_unique,
            "mean_edge_confidence": float(pweight.mean()),
        },
        "test_diagnostics": {
            "unique_top1_unordered_pairs": t_unique,
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
    parser.add_argument("--fold0-scores", type=Path,
                        default=Path("outputs/large/phase1_ranker_fold0_scores.npz"))
    parser.add_argument("--unlabeled-priors", type=Path, required=True)
    parser.add_argument("--test-npz", type=Path, default=Path("outputs/large/phase1_ranker_test.npz"))
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--debias", type=float, default=1.0)
    parser.add_argument("--cavity-iters", type=int, default=12)
    parser.add_argument("--use-escape", action="store_true",
                        help="Enable out-of-pool escape weighting (disabled by default due to label smoothing)")
    parser.add_argument("--save-pseudo-battles", type=Path, default=None)
    parser.add_argument("--save-transductive-bank", type=Path, default=None)
    parser.add_argument("--bank-min-conf", type=float, default=0.92)
    parser.add_argument("--test-bank-min-conf", type=float, default=0.93)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "evaluate-fold0":
        run_fold0(args, device)
    else:
        run_submission(args, device)


if __name__ == "__main__":
    main()
