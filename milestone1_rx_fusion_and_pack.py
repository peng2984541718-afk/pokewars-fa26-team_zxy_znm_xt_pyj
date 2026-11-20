"""Milestone 1: 6-view TTA student post-fusion on rx_ranker, leak-free Fold-0
selection, immediate submission_rx_v1.csv export, and Gen-2 teacher/battle packs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from degradation import clean_impulses_gpu
from evaluate_soft_score import DENOMINATOR, rmse
from export_probs_rot5 import score_model
from image_experiment import NUM_CLASSES, SEED, stratified_folds
from joint_constrained_inference import (
    power_scale_posteriors,
    run_submission,
    run_unlabeled_bp,
    target_cavity_predict,
)
from mix_distill_posteriors import fuse_posteriors, mix
from train_large_image import make_model

L = Path("outputs/large")
ALPHAS = (0.0, 0.15, 0.25, 0.35, 0.45, 0.55)
TEMPERATURES = (0.65, 0.75, 0.85, 1.00)
DENSE_SHARES = (0.0, 0.20, 0.35)
DEBIASES = (1.0, 1.25)
SCALES = (0.85, 0.90, 0.95, 1.00, 1.05)


def score_fn(val: float) -> float:
    return 100.0 * (1.0 - val / DENOMINATOR)


@torch.inference_mode()
def ensure_dense_f0_rot5() -> tuple[Path, Path]:
    """Export 6-view TTA Fold-0 valid & unlabeled probs for densenet169_dist_f0_best.pt."""
    f0_out = L / "densenet169_dist_f0_rot5.npy"
    unl_out = L / "unl_f0_dense_dist.npy"
    ckpt = L / "densenet169_dist_f0_best.pt"
    if f0_out.exists() and unl_out.exists():
        return f0_out, unl_out

    model = make_model("densenet169_cifar", pretrained=False).cuda().eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))

    if not f0_out.exists():
        labels = np.load("labeled_ids.npy").astype(np.int64)
        valid_idx = np.flatnonzero(stratified_folds(labels, 3, SEED) == 0)
        imgs = torch.as_tensor(np.asarray(np.load("labeled_32x32.npy", mmap_mode="r")).copy(),
                               device="cuda")
        imgs = clean_impulses_gpu(imgs)
        probs = score_model(model, "densenet169_cifar", imgs, valid_idx, batch_size=256)
        np.save(f0_out, probs.astype(np.float16))
        top1 = float((probs.argmax(1) == labels[valid_idx]).mean())
        print(f"exported {f0_out.name}: shape={probs.shape} top1={top1:.5f}", flush=True)
        del imgs

    if not unl_out.exists():
        u_imgs = torch.as_tensor(np.asarray(np.load("unl/test_32x32.npy", mmap_mode="r")).copy(),
                                 device="cuda")
        u_imgs = clean_impulses_gpu(u_imgs)
        idx = np.arange(len(u_imgs), dtype=np.int64)
        u_probs = score_model(model, "densenet169_cifar", u_imgs, idx, batch_size=256)
        np.save(unl_out, u_probs.astype(np.float16))
        print(f"exported {unl_out.name}: shape={u_probs.shape}", flush=True)
        del u_imgs

    del model
    torch.cuda.empty_cache()
    return f0_out, unl_out


def gather_cand(mat: np.ndarray, cand_1based: np.ndarray,
                rows: np.ndarray | None = None) -> np.ndarray:
    sub = mat if rows is None else mat[rows]
    g = np.take_along_axis(np.asarray(sub, dtype=np.float32),
                           cand_1based.astype(np.int64) - 1, axis=1)
    g /= np.maximum(g.sum(1, keepdims=True), 1e-8)
    return np.ascontiguousarray(g)


def load_fold0_ctx(device: str) -> dict:
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

    table = np.load(L / "pair_semi_nofold0_new_table_round1.npy")
    known_allowed = np.triu(np.ones((1026, 1026), dtype=np.float32), k=1)
    known_allowed[0, :] = 0.0
    known_allowed[:, 0] = 0.0
    ka, kb = a[~pair_f0], b[~pair_f0]
    known_allowed[ka, kb] = 0.0

    unk = rows.pokemon_id1.isna() & rows.pokemon_id2.isna()
    u_idx = pd.Index(Path("data/unlabeled_paths.txt").read_text().split())
    eu = u_idx.get_indexer(rows.noisy_image1[unk])
    ev = u_idx.get_indexer(rows.noisy_image2[unk])
    uy = rows.result[unk].to_numpy(np.float32)

    with np.load(L / "rx_ranker_unlabeled_f0only.npz") as uz:
        u_cand = uz["candidates"].astype(np.int64)
        u_pri = uz["posteriors"].astype(np.float32)
        u_oop = uz["oop_mass"].astype(np.float32)
    with np.load(L / "rx_ranker_fold0_scores.npz") as fz:
        v_idx = fz["valid_indices"].astype(np.int64)
        cand = fz["candidates"].astype(np.int64)
        post = fz["posteriors"].astype(np.float32)
        oop = fz["oop_mass"].astype(np.float32)

    pos = np.full(len(paths), -1, dtype=np.int32)
    pos[v_idx] = np.arange(len(v_idx))
    one, two = pos[f1[pair_rows]], pos[f2[pair_rows]]

    _, _, base_unl_lnot = run_unlabeled_bp(
        u_cand, u_pri, eu, ev, uy, table, known_allowed, device=device)
    return {
        "a": a, "b": b, "pair_rows": pair_rows, "split": split,
        "actual": y[pair_rows], "one": one, "two": two,
        "table": table, "known_allowed": known_allowed,
        "u_cand": u_cand, "u_pri": u_pri, "u_oop": u_oop,
        "eu": eu, "ev": ev, "uy": uy,
        "v_idx": v_idx, "cand": cand, "post": post, "oop": oop,
        "base_unl_lnot": base_unl_lnot, "device": device,
    }


def eval_candidate(ctx: dict, q_post: np.ndarray, u_post: np.ndarray,
                   debias: float, exact_unl_bp: bool) -> dict:
    device = ctx["device"]
    if exact_unl_bp:
        _, _, unl_lnot0 = run_unlabeled_bp(
            ctx["u_cand"], u_post, ctx["eu"], ctx["ev"], ctx["uy"],
            ctx["table"], ctx["known_allowed"], device=device)
    else:
        unl_lnot0 = ctx["base_unl_lnot"]

    _, _, test_lnot, _, _ = target_cavity_predict(
        q_post, ctx["oop"], ctx["cand"], ctx["one"], ctx["two"],
        ctx["known_allowed"], unl_lnot0, ctx["table"], debias=debias, device=device)
    u_node, u_edge, unl_lnot = run_unlabeled_bp(
        ctx["u_cand"], u_post, ctx["eu"], ctx["ev"], ctx["uy"],
        ctx["table"], ctx["known_allowed"], external_log_avail=test_lnot, device=device)

    actual = ctx["actual"]
    split = ctx["split"]
    ta, tb = ctx["a"][ctx["pair_rows"]], ctx["b"][ctx["pair_rows"]]
    best = None
    trials = []
    for scale in SCALES:
        qs = power_scale_posteriors(q_post, scale)
        pred, top_p, _, _, _ = target_cavity_predict(
            qs, ctx["oop"], ctx["cand"], ctx["one"], ctx["two"],
            ctx["known_allowed"], unl_lnot, ctx["table"],
            debias=debias, cavity_iters=12, use_escape=False, device=device)
        both_ok = float(((top_p[:, 0] == ta) & (top_p[:, 1] == tb))[split:].mean())
        tune_r = rmse(pred[:split], actual[:split])
        test_r = rmse(pred[split:], actual[split:])
        item = {
            "scale": scale,
            "tune_rmse": tune_r,
            "test_rmse": test_r,
            "tune_score": score_fn(tune_r),
            "test_score": score_fn(test_r),
            "test_both_correct": both_ok,
        }
        trials.append(item)
        if best is None or item["tune_rmse"] < best["tune_rmse"]:
            best = item
    return {"selected": best, "trials": trials, "u_node": u_node, "u_edge": u_edge}


def save_pseudo_battles(path: Path, u_cand: np.ndarray, eu: np.ndarray,
                        ev: np.ndarray, uy: np.ndarray, u_edge: np.ndarray) -> None:
    k_u = u_cand.shape[1]
    flat = u_edge.reshape(len(eu), k_u * k_u)
    best_e = flat.argmax(1)
    pa = u_cand[eu, best_e // k_u]
    pb = u_cand[ev, best_e % k_u]
    w = flat[np.arange(len(eu)), best_e]
    # Enforce exact dataset invariant pa < pb (swap & invert score if needed)
    swap = pa > pb
    if np.any(swap):
        pa_s = np.where(swap, pb, pa)
        pb_s = np.where(swap, pa, pb)
        uy_s = np.where(swap, 10.0 - uy, uy)
        pa, pb, uy = pa_s, pb_s, uy_s
    valid = pa < pb
    np.savez_compressed(
        path,
        a=pa[valid].astype(np.uint16),
        b=pb[valid].astype(np.uint16),
        y=uy[valid].astype(np.float32),
        weight=w[valid].astype(np.float32),
    )
    print(f"saved pseudo-battles {path.name}: rows={int(valid.sum())} "
          f"mean_w={float(w[valid].mean()):.4f} w>=0.85={int((w[valid] >= 0.85).sum())}",
          flush=True)


def export_gen2_teacher_packs(alpha: float, temperature: float, dense_share: float,
                              unl_fused_path: Path, test_fused_path: Path,
                              unl_f0_fused_path: Path) -> None:
    """Build leak-free OOF teacher pack + external teacher packs for Gen-2 distillation."""
    n_labeled = len(np.load("labeled_ids.npy"))
    with np.load(L / "rx_ranker_oof.npz") as z:
        idx = z["indices"].astype(np.int64)
        cand_raw = z["candidates"].astype(np.uint16)
        post_raw = z["posteriors"].astype(np.float32)

    cand = np.empty((n_labeled, cand_raw.shape[1]), dtype=np.uint16)
    post = np.empty((n_labeled, post_raw.shape[1]), dtype=np.float32)
    cand[idx] = cand_raw
    post[idx] = post_raw

    if alpha > 0:
        rx_oof = gather_cand(np.load(L / "oof_resnext29_dist.npy", mmap_mode="r"), cand)
        dn_oof = gather_cand(np.load(L / "oof_dense_p2imp.npy", mmap_mode="r"), cand)
        stu = dense_share * dn_oof + (1.0 - dense_share) * rx_oof
        stu /= np.maximum(stu.sum(1, keepdims=True), 1e-8)
        post = fuse_posteriors(post, stu, alpha, "log", temperature)

    np.savez_compressed(
        L / "gen2_distill_labeled_oof.npz",
        candidates=cand,
        posteriors=post.astype(np.float16),
    )
    labels = np.load("labeled_ids.npy").astype(np.int64) + 1
    top1 = float((cand[np.arange(n_labeled), post.argmax(1)] == labels).mean())
    print(f"saved gen2_distill_labeled_oof.npz: top1={top1:.5f} "
          f"top1_mass={float(post.max(1).mean()):.4f}", flush=True)

    with np.load(unl_f0_fused_path) as u0:
        np.savez_compressed(
            L / "gen2_distill_unl_f0only.npz",
            candidates=u0["candidates"].astype(np.uint16),
            posteriors=u0["posteriors"].astype(np.float16),
        )

    with np.load(unl_fused_path) as u, np.load(test_fused_path) as t:
        cand_e = np.concatenate([u["candidates"], t["candidates"]], axis=0).astype(np.uint16)
        post_e = np.concatenate([u["posteriors"], t["posteriors"]], axis=0).astype(np.float16)
    ext_imgs = L / "distill_external_images.npy"
    if not ext_imgs.exists():
        unl_im = np.asarray(np.load("unl/test_32x32.npy"))
        tst_im = np.asarray(np.load("test_32x32.npy"))
        np.save(ext_imgs, np.concatenate([unl_im, tst_im], axis=0))
    np.savez_compressed(L / "gen2_distill_external_full.npz",
                        candidates=cand_e, posteriors=post_e)
    print(f"saved gen2_distill_external_full.npz: rows={len(cand_e)}", flush=True)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f0_dn_path, u0_dn_path = ensure_dense_f0_rot5()
    ctx = load_fold0_ctx(device)

    dn_f0 = gather_cand(np.load(f0_dn_path, mmap_mode="r"), ctx["cand"])
    rx_f0 = gather_cand(np.load(L / "oof_resnext29_dist.npy", mmap_mode="r"),
                        ctx["cand"], rows=ctx["v_idx"])
    dn_u0 = gather_cand(np.load(u0_dn_path, mmap_mode="r"), ctx["u_cand"])
    rx_u0 = gather_cand(np.load(L / "unl_f0_resnext29_dist.npy", mmap_mode="r"), ctx["u_cand"])

    coarse = []
    # Anchor: alpha = 0.0 (pure rx_ranker) across debias
    for debias in DEBIASES:
        res = eval_candidate(ctx, ctx["post"], ctx["u_pri"], debias=debias, exact_unl_bp=False)
        rec = {
            "alpha": 0.0, "temperature": 1.0, "dense_share": 0.0,
            "debias": debias, "selected": res["selected"],
        }
        coarse.append(rec)
        print(f"[anchor] alpha=0.00 debias={debias:.2f} "
              f"tune={rec['selected']['tune_score']:.4f} "
              f"test={rec['selected']['test_score']:.4f} "
              f"both={rec['selected']['test_both_correct']:.5f}", flush=True)

    grid = [(a, t, s, d)
            for d in DEBIASES
            for s in DENSE_SHARES
            for a in ALPHAS if a > 0.0
            for t in TEMPERATURES]
    for idx_g, (alpha, temp, share, debias) in enumerate(grid, start=1):
        stu_q = share * dn_f0 + (1.0 - share) * rx_f0
        stu_q /= np.maximum(stu_q.sum(1, keepdims=True), 1e-8)
        q_mix = fuse_posteriors(ctx["post"], stu_q, alpha, "log", temp)
        res = eval_candidate(ctx, q_mix, ctx["u_pri"], debias=debias, exact_unl_bp=False)
        rec = {
            "alpha": alpha, "temperature": temp, "dense_share": share,
            "debias": debias, "selected": res["selected"],
        }
        coarse.append(rec)
        if idx_g % 10 == 0 or idx_g == len(grid):
            print(f"[{idx_g:03d}/{len(grid)}] alpha={alpha:.2f} T={temp:.2f} "
                  f"dn={share:.2f} deb={debias:.2f} "
                  f"tune={rec['selected']['tune_score']:.4f} "
                  f"test={rec['selected']['test_score']:.4f}", flush=True)

    # Re-evaluate top 8 coarse tune winners (plus anchor) with exact coupled u_mix <-> q_mix BP
    coarse.sort(key=lambda r: (r["selected"]["tune_rmse"], r["alpha"]))
    shortlist = coarse[:8]
    anchors = [r for r in coarse if r["alpha"] == 0.0]
    for anc in anchors:
        if anc not in shortlist:
            shortlist.append(anc)

    exact_records = []
    best_bundle = None
    for item in shortlist:
        alpha = item["alpha"]
        temp = item["temperature"]
        share = item["dense_share"]
        debias = item["debias"]
        if alpha == 0.0:
            q_mix, u_mix = ctx["post"], ctx["u_pri"]
        else:
            stu_q = share * dn_f0 + (1.0 - share) * rx_f0
            stu_q /= np.maximum(stu_q.sum(1, keepdims=True), 1e-8)
            q_mix = fuse_posteriors(ctx["post"], stu_q, alpha, "log", temp)
            stu_u = share * dn_u0 + (1.0 - share) * rx_u0
            stu_u /= np.maximum(stu_u.sum(1, keepdims=True), 1e-8)
            u_mix = fuse_posteriors(ctx["u_pri"], stu_u, alpha, "log", temp)
        res = eval_candidate(ctx, q_mix, u_mix, debias=debias, exact_unl_bp=True)
        rec = {
            "alpha": alpha, "temperature": temp, "dense_share": share,
            "resnext_share": round(1.0 - share, 4),
            "debias": debias, "selected": res["selected"],
        }
        exact_records.append(rec)
        print(f"[exact] alpha={alpha:.2f} T={temp:.2f} dn={share:.2f} deb={debias:.2f} "
              f"tune={rec['selected']['tune_score']:.4f} "
              f"test={rec['selected']['test_score']:.4f} "
              f"both={rec['selected']['test_both_correct']:.5f}", flush=True)
        if best_bundle is None or (
            rec["selected"]["tune_rmse"], rec["alpha"]
        ) < (
            best_bundle[0]["selected"]["tune_rmse"], best_bundle[0]["alpha"]
        ):
            best_bundle = (rec, q_mix, u_mix, res["u_edge"])

    winner, win_q_f0, win_u_f0, win_u_edge_f0 = best_bundle
    print("MILESTONE 1 WINNER:", json.dumps(winner, indent=2), flush=True)

    # Save Fold-0 winning posteriors & Fold-0-clean pseudo-battles for Milestone 2/3
    np.savez_compressed(
        L / "rx_v1_fold0_scores.npz",
        valid_indices=ctx["v_idx"],
        candidates=ctx["cand"].astype(np.uint16),
        posteriors=win_q_f0.astype(np.float32),
        oop_mass=ctx["oop"].astype(np.float32),
    )
    np.savez_compressed(
        L / "rx_v1_unlabeled_f0only.npz",
        candidates=ctx["u_cand"].astype(np.uint16),
        posteriors=win_u_f0.astype(np.float32),
        oop_mass=ctx["u_oop"].astype(np.float32),
    )
    save_pseudo_battles(L / "unlabeled_pseudo_battles_f0clean.npz",
                        ctx["u_cand"], ctx["eu"], ctx["ev"], ctx["uy"], win_u_edge_f0)

    # Mix full unlabeled & test posteriors and generate submission_rx_v1.csv
    unl_out = L / "rx_v1_unlabeled.npz"
    test_out = L / "rx_v1_test.npz"
    alpha = winner["alpha"]
    temp = winner["temperature"]
    share = winner["dense_share"]
    debias = winner["debias"]
    scale = winner["selected"]["scale"]

    if alpha > 0.0:
        u_sources = [(L / "unl_resnext29_dist.npy", 1.0 - share)]
        t_sources = [(L / "test_resnext29_dist.npy", 1.0 - share)]
        if share > 0.0:
            u_sources.append((L / "unl_dense_distill.npy", share))
            t_sources.append((L / "test_dense_distill.npy", share))
        mix(L / "rx_ranker_unlabeled.npz", u_sources, unl_out, None, alpha, "log", temp)
        mix(L / "rx_ranker_test.npz", t_sources, test_out, None, alpha, "log", temp)
    else:
        for src, dst in ((L / "rx_ranker_unlabeled.npz", unl_out),
                         (L / "rx_ranker_test.npz", test_out)):
            with np.load(src) as z:
                np.savez_compressed(dst, **{k: z[k] for k in z.files})

    sub_csv = L / "submission_rx_v1.csv"
    sub_args = argparse.Namespace(
        battles=Path("data/known_battles.npz"),
        train_csv=Path("data/train.csv.gz"),
        test_csv=Path("data/test.csv.gz"),
        unlabeled_paths=Path("data/unlabeled_paths.txt"),
        data_dir=Path("."),
        unlabeled_priors=unl_out,
        test_npz=test_out,
        table=L / "pair_semi_v23_blend_table.npy",
        scale=scale,
        debias=debias,
        cavity_iters=12,
        use_escape=False,
        save_pseudo_battles=L / "unlabeled_pseudo_battles_gen2.npz",
        save_transductive_bank=L / "transductive_bank_gen2_full.npz",
        bank_min_conf=0.92,
        test_bank_min_conf=0.93,
        output=sub_csv,
    )
    run_submission(sub_args, device)

    # Enforce pa < pb on unlabeled_pseudo_battles_gen2.npz as well
    with np.load(L / "unlabeled_pseudo_battles_gen2.npz") as pbz:
        pa = pbz["a"].astype(np.int64)
        pb = pbz["b"].astype(np.int64)
        py_v = pbz["y"].astype(np.float32)
        pw = pbz["weight"].astype(np.float32)
    swap = pa > pb
    if np.any(swap):
        pa, pb = np.where(swap, pb, pa), np.where(swap, pa, pb)
        py_v = np.where(swap, 10.0 - py_v, py_v)
    valid = pa < pb
    np.savez_compressed(
        L / "unlabeled_pseudo_battles_gen2.npz",
        a=pa[valid].astype(np.uint16),
        b=pb[valid].astype(np.uint16),
        y=py_v[valid].astype(np.float32),
        weight=pw[valid].astype(np.float32),
    )

    export_gen2_teacher_packs(alpha, temp, share, unl_out, test_out,
                              L / "rx_v1_unlabeled_f0only.npz")

    report = {
        "winner": winner,
        "submission_csv": str(sub_csv),
        "exact_shortlist": exact_records,
    }
    (L / "milestone1_rx_fusion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Milestone 1 complete -> {sub_csv}", flush=True)


if __name__ == "__main__":
    main()
