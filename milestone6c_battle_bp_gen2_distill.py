"""Milestone 6C (Unified Grand-Slam of Discoveries #1 + #2 + #3):
Battle-BP-Disambiguated Full-Data Transductive Soft Distillation across
Orthogonal Architectures — including 64x64 7x7 `convnextv2_nano` (distilled for
the very first time!), `resnext29_8x64d`, and `densenet169_cifar` — fused with
`BilateralSpatialAppendageNet` (Discovery #3), `SE-PyramidNet` (Milestone 5),
and `PairTable v5` (Discovery #2).

Why Battle-BP Transductive Soft Distillation breaks the Twin-30 ceiling:
- Earlier `distill_external_full.npz` used raw pre-BP image posteriors from the
  79.9 ranker, so confused twins (`#770` vs `#993`) remained confused in soft
  targets.
- Here, on `68,675` `unl` images (2.31 battles/image + 67/class Sinkhorn) and
  `160,000` `test` images, Battle BP with our `~0.17` RMSE payoff table resolves
  twin ambiguities via battle outcomes (`0.00` vs `9.98`)!
- Training `--train-all` students on `160,925` labeled + `228,675` Battle-BP
  external soft targets gives 3.4x visual coverage with twin-disambiguated
  supervision!
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from degradation import clean_impulses_gpu
from evaluate_soft_score import DENOMINATOR, rmse
from export_probs_rot5 import score_model
from image_experiment import SEED, stratified_folds
from train_large_image import make_model
from joint_constrained_inference import (
    power_scale_posteriors,
    run_unlabeled_bp,
    target_cavity_predict,
)
from milestone1_rx_fusion_and_pack import gather_cand
from milestone3_inference_calibration import (
    adaptive_power_scale,
    run_unlabeled_bp_hetero,
    z_lap_from_gallery,
)
from milestone6b_twin_spatial_specialist import adaptive_specialist_fuse
from mix_distill_posteriors import fuse_posteriors

L = Path("outputs/large")


def score_fn(val: float) -> float:
    return 100.0 * (1.0 - val / DENOMINATOR)


def run_cmd(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN:", " ".join(cmd), f"-> {log_path}", flush=True)
    with log_path.open("w") as handle:
        subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=True)


def build_battle_bp_teacher_packs(nof0_tab: np.ndarray, full_tab: np.ndarray,
                                  debias: float, device: str) -> tuple[Path, Path]:
    """Build:
    1. `bp_v5_unl_f0only_pack.npz`: Fold-0-leak-free Battle-BP soft targets on
       `unl/test_32x32.npy` (for Fold-0 validation pilot gate).
    2. `bp_v5_external_full.npz`: Full Battle-BP soft targets on all 228,675
       `unl + test` images (`distill_external_images.npy`).
    """
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
    pair_f0 = (folds[f1] == 0) & (folds[f2] == 0)

    unk = rows.pokemon_id1.isna() & rows.pokemon_id2.isna()
    u_idx = pd.Index(Path("data/unlabeled_paths.txt").read_text().split())
    eu = u_idx.get_indexer(rows.noisy_image1[unk])
    ev = u_idx.get_indexer(rows.noisy_image2[unk])
    uy = rows.result[unk].to_numpy(np.float32)

    # 1. Leak-free Fold-0 unlabeled Battle-BP teacher pack
    f0_pack_out = L / "bp_v5_unl_f0only_pack.npz"
    if not f0_pack_out.exists():
        known_f0 = np.triu(np.ones((1026, 1026), dtype=np.float32), k=1)
        known_f0[0, :] = 0.0
        known_f0[:, 0] = 0.0
        known_f0[a[~pair_f0], b[~pair_f0]] = 0.0
        with np.load(L / "rx_v1_unlabeled_f0only.npz") as uz:
            u_cand = uz["candidates"].astype(np.int64)
            u_pri = uz["posteriors"].astype(np.float32)
        u_node, _, _ = run_unlabeled_bp(
            u_cand, u_pri, eu, ev, uy, nof0_tab, known_f0, device=device)
        # Blend 0.70 Battle-BP + 0.30 visual posterior for smooth dark knowledge
        u_teach = 0.70 * u_node + 0.30 * u_pri
        u_teach /= np.maximum(u_teach.sum(1, keepdims=True), 1e-8)
        # Convert 1..1025 -> 0..1024? Check build_distill_pack.py convention!
        # NOTE: ranker_unlabeled.npz stores 1-based Pokemon IDs (1..1025), and
        # soft_cross_entropy in train_large_image_external.py expects 1-based IDs!
        np.savez_compressed(f0_pack_out,
                            candidates=u_cand.astype(np.uint16),
                            posteriors=u_teach.astype(np.float16))
        print(f"Built {f0_pack_out.name}: top1_conf={float(u_teach.max(1).mean()):.4f}", flush=True)

    # 2. Full 228,675 external (`unl + test`) Battle-BP teacher pack
    full_pack_out = L / "bp_v5_external_full.npz"
    if not full_pack_out.exists():
        with np.load("data/known_battles.npz") as kz:
            ka_all, kb_all = kz["a"].astype(np.int64), kz["b"].astype(np.int64)
        known_full = np.triu(np.ones((1026, 1026), dtype=np.float32), k=1)
        known_full[0, :] = 0.0
        known_full[:, 0] = 0.0
        known_full[ka_all, kb_all] = 0.0

        # Pick strongest available unlabeled & test packs
        u_src = L / "rx_v6b_unlabeled.npz" if (L / "rx_v6b_unlabeled.npz").exists() else (
            L / "rx_v4_unlabeled.npz" if (L / "rx_v4_unlabeled.npz").exists() else L / "rx_v1_unlabeled.npz"
        )
        t_src = L / "rx_v6b_test.npz" if (L / "rx_v6b_test.npz").exists() else (
            L / "rx_v4_test.npz" if (L / "rx_v4_test.npz").exists() else L / "rx_v1_test.npz"
        )
        with np.load(u_src) as uz:
            tu_cand = uz["candidates"].astype(np.int64)
            tu_pri = uz["posteriors"].astype(np.float32)
        with np.load(t_src) as tz:
            t_cand = tz["candidates"].astype(np.int64)
            t_raw = tz["posteriors"].astype(np.float32)
            t_oop = tz["oop_mass"].astype(np.float32)

        test_paths = Path("test_paths.txt").read_text().splitlines()
        p_idx = pd.Index(test_paths)
        test_df = pd.read_csv("data/test.csv.gz")
        idx1 = p_idx.get_indexer(test_df.noisy_image1)
        idx2 = p_idx.get_indexer(test_df.noisy_image2)

        t_scaled = power_scale_posteriors(t_raw, 0.95)
        u_node, _, u_lnot = run_unlabeled_bp(
            tu_cand, tu_pri, eu, ev, uy, full_tab, known_full, device=device)
        for outer in range(2):
            _, _, test_lnot, m1, m2 = target_cavity_predict(
                t_scaled, t_oop, t_cand, idx1, idx2, known_full, u_lnot, full_tab,
                debias=debias, cavity_iters=12, use_escape=False, device=device)
            if outer == 0:
                u_node, _, u_lnot = run_unlabeled_bp(
                    tu_cand, tu_pri, eu, ev, uy, full_tab, known_full,
                    external_log_avail=test_lnot, device=device)

        t_marg = np.zeros_like(t_raw)
        t_marg[idx1] = m1
        t_marg[idx2] = m2

        # Unlabeled has 2.31 battles/img -> 0.75 BP + 0.25 visual
        # Test has 1.0 battle/img -> conservative 0.45 Cavity-BP + 0.55 visual
        u_soft = 0.75 * u_node + 0.25 * tu_pri
        u_soft /= np.maximum(u_soft.sum(1, keepdims=True), 1e-8)
        t_soft = 0.45 * t_marg + 0.55 * t_raw
        t_soft /= np.maximum(t_soft.sum(1, keepdims=True), 1e-8)

        cand_e = np.concatenate([tu_cand, t_cand], axis=0).astype(np.uint16)
        post_e = np.concatenate([u_soft, t_soft], axis=0).astype(np.float16)
        np.savez_compressed(full_pack_out, candidates=cand_e, posteriors=post_e)
        print(f"Built {full_pack_out.name}: rows={len(cand_e)} "
              f"unl_conf={float(u_soft.max(1).mean()):.4f} "
              f"test_conf={float(t_soft.max(1).mean()):.4f}", flush=True)

        if not (L / "distill_external_images.npy").exists():
            unl_im = np.asarray(np.load("unl/test_32x32.npy"))
            test_im = np.asarray(np.load("test_32x32.npy"))
            np.save(L / "distill_external_images.npy", np.concatenate([unl_im, test_im], axis=0))

    return f0_pack_out, full_pack_out


def train_pilot_student(model_name: str, init_pt: Path, lr: float,
                        batch_size: int, short: str, f0_ext_pack: Path) -> None:
    """Train leak-free Fold-0 pilot (`106,600` Fold-1+2 + `68,675` Battle-BP unl)."""
    py = sys.executable
    common = [
        "--training-seed", "202611",
        "--batch-size", str(batch_size),
        "--eval-batch-size", "256",
        "--clean-impulses",
        "--translate-pixels", "1.0",
        "--rotate-degrees", "8",
        "--scale-min", "0.95",
        "--scale-max", "1.05",
        "--epochs", "12",
        "--warmup-epochs", "1",
        "--min-epochs", "8",
        "--patience", "6",
        "--min-improvement", "0.0002",
        "--learning-rate", f"{lr}",
        "--min-learning-rate", "1e-6",
        "--distill-labeled", str(L / "gen2_distill_labeled_oof.npz"),
        "--distill-alpha", "0.55",
        "--distill-fraction", "0.42",
        "--data-dir", ".",
        "--output-dir", str(L),
    ]

    f0_run = f"{short}_bpv5_f0"
    if not (L / f"{f0_run}_best.pt").exists():
        run_cmd([
            py, "-u", "train_large_image_external.py", *common,
            "--model", model_name, "--fold", "0",
            "--initial-weights", str(init_pt),
            "--distill-external-images", "unl/test_32x32.npy",
            "--distill-external-pack", str(f0_ext_pack),
            "--run-name", f0_run,
        ], Path("logs") / f"{f0_run}.log")

    f0_out = L / f"f0_rot5_{short}_bpv5.npy"
    u0_out = L / f"unl_f0_{short}_bpv5.npy"
    if not f0_out.exists() or not u0_out.exists():
        model = make_model(model_name, pretrained=False).cuda().eval()
        model.load_state_dict(torch.load(L / f"{f0_run}_best.pt", map_location="cpu", weights_only=True))
        if not f0_out.exists():
            labels_np = np.load("labeled_ids.npy").astype(np.int64)
            f0_valid = np.flatnonzero(stratified_folds(labels_np, 3, SEED) == 0)
            imgs = clean_impulses_gpu(torch.as_tensor(np.asarray(np.load("labeled_32x32.npy", mmap_mode="r")).copy(), device="cuda"))
            probs = score_model(model, model_name, imgs, f0_valid, batch_size=256)
            np.save(f0_out, probs.astype(np.float16))
            top1 = float((probs.argmax(1) == labels_np[f0_valid]).mean())
            print(f"[{short}_bpv5] Fold-0 6-view TTA top1 = {top1:.5f}", flush=True)
            del imgs
        if not u0_out.exists():
            u_imgs = clean_impulses_gpu(torch.as_tensor(np.asarray(np.load("unl/test_32x32.npy", mmap_mode="r")).copy(), device="cuda"))
            u_probs = score_model(model, model_name, u_imgs, np.arange(len(u_imgs), dtype=np.int64), batch_size=256)
            np.save(u0_out, u_probs.astype(np.float16))
            del u_imgs
        del model
        torch.cuda.empty_cache()


def train_full_all_images_student(model_name: str, init_pt: Path, lr: float,
                                  batch_size: int, short: str, full_ext_pack: Path) -> None:
    """Train full `--train-all` student on ALL 389,600 images (`160,925` labeled +
    `228,675` Battle-BP external `unl + test` images) for Kaggle submission!"""
    py = sys.executable
    common = [
        "--training-seed", "202611",
        "--batch-size", str(batch_size),
        "--eval-batch-size", "256",
        "--clean-impulses",
        "--translate-pixels", "1.0",
        "--rotate-degrees", "8",
        "--scale-min", "0.95",
        "--scale-max", "1.05",
        "--epochs", "12",
        "--warmup-epochs", "1",
        "--min-epochs", "8",
        "--patience", "6",
        "--min-improvement", "0.0002",
        "--learning-rate", f"{lr}",
        "--min-learning-rate", "1e-6",
        "--distill-labeled", str(L / "gen2_distill_labeled_oof.npz"),
        "--distill-alpha", "0.55",
        "--distill-fraction", "0.42",
        "--data-dir", ".",
        "--output-dir", str(L),
    ]

    full_run = f"{short}_bpv5_full_f0"
    if not (L / f"{full_run}_best.pt").exists():
        run_cmd([
            py, "-u", "train_large_image_external.py", *common,
            "--train-all", "--model", model_name, "--fold", "0",
            "--initial-weights", str(init_pt),
            "--distill-external-images", str(L / "distill_external_images.npy"),
            "--distill-external-pack", str(full_ext_pack),
            "--run-name", full_run,
        ], Path("logs") / f"{full_run}.log")

    for qdir, out_name in [("unl", f"unl_{short}_bpv5_full.npy"), (".", f"test_{short}_bpv5_full.npy")]:
        if not (L / out_name).exists():
            run_cmd([
                py, "-u", "export_probs_rot5.py",
                "--model", model_name, "--prefixes", f"{short}_bpv5_full",
                "--folds", "0", "--split", "query", "--query-dir", qdir,
                "--clean-impulses", "--batch-size", "256",
                "--output", str(L / out_name),
            ], Path("logs") / f"export_{Path(out_name).stem}.log")


def main() -> None:
    device = "cuda"
    m6a = L / "milestone6a_table_v5.json"
    if m6a.exists() and json.loads(m6a.read_text()).get("passed"):
        nof0_tab = np.load(L / "pair_v5_nofold0_blend_table.npy")
        full_tab = np.load(L / "pair_v5_full_blend_table.npy")
    else:
        m2 = json.loads((L / "milestone2_pair_table_v4.json").read_text())
        nof0_tab = np.load(Path(m2["active_nof0_table"]))
        full_tab = np.load(Path(m2["active_full_table"]))

    m3 = json.loads((L / "milestone3_calibration.json").read_text())["active"]
    s0, s1 = float(m3["s0"]), float(m3["s1"])
    debias = float(m3["debias"])
    beta_lap = float(m3["beta_lap"])

    f0_ext_pack, full_ext_pack = build_battle_bp_teacher_packs(nof0_tab, full_tab, debias, device)

    # 1. Distill leak-free Fold-0 pilots first (`64x64 7x7 convnextv2_nano`, `resnext29_8x64d`, `densenet169_cifar`)
    specs = [
        ("convnextv2_nano", L / "convnextv2_imp_f0_best.pt", 1.4e-4, 128, "convnext"),
        ("resnext29_8x64d", L / "resnext29_f0_best.pt", 1.0e-4, 64, "resnext29"),
        ("densenet169_cifar", L / "densenet169_imp_f0_best.pt", 1.0e-4, 128, "densenet169"),
    ]
    for model_name, init_pt, lr, bs, short in specs:
        train_pilot_student(model_name, init_pt, lr=lr, batch_size=bs, short=short, f0_ext_pack=f0_ext_pack)

    # 2. Leak-free Fold-0 Grand-Slam Fusion & Selection across ALL available orthogonal models!
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

    with np.load(L / "rx_ranker_unlabeled_f0only.npz") as uz:
        u_cand = uz["candidates"].astype(np.int64)
        u_ranker = uz["posteriors"].astype(np.float32)
    with np.load(L / "rx_ranker_fold0_scores.npz") as fz:
        v_idx = fz["valid_indices"].astype(np.int64)
        cand = fz["candidates"].astype(np.int64)
        q_ranker = fz["posteriors"].astype(np.float32)
        oop = fz["oop_mass"].astype(np.float32)

    z_lap_f0 = z_lap_from_gallery(L / "gallery_rx_top40_f0.npz", v_idx)
    pos = np.full(len(paths), -1, dtype=np.int32)
    pos[v_idx] = np.arange(len(v_idx))
    one, two = pos[f1[pair_rows]], pos[f2[pair_rows]]
    actual = y[pair_rows]
    ta, tb = a[pair_rows], b[pair_rows]

    def norm_rows(m: np.ndarray) -> np.ndarray:
        return m / np.maximum(m.sum(1, keepdims=True), 1e-8)

    q_conv = norm_rows(gather_cand(np.load(L / "f0_rot5_convnext_bpv5.npy", mmap_mode="r"), cand))
    u_conv = norm_rows(gather_cand(np.load(L / "unl_f0_convnext_bpv5.npy", mmap_mode="r"), u_cand))

    # Match full-export blend: 0.65 Battle-BP student + 0.35 Gen-1 external student
    q_rx_g1 = norm_rows(gather_cand(np.load(L / "oof_resnext29_dist.npy", mmap_mode="r"), cand, rows=v_idx))
    u_rx_g1 = norm_rows(gather_cand(np.load(L / "unl_f0_resnext29_dist.npy", mmap_mode="r"), u_cand))
    q_dn_g1 = norm_rows(gather_cand(np.load(L / "densenet169_dist_f0_rot5.npy", mmap_mode="r"), cand))
    u_dn_g1 = norm_rows(gather_cand(np.load(L / "unl_f0_dense_dist.npy", mmap_mode="r"), u_cand))

    q_rx = norm_rows(0.65 * norm_rows(gather_cand(np.load(L / "f0_rot5_resnext29_bpv5.npy", mmap_mode="r"), cand)) + 0.35 * q_rx_g1)
    u_rx = norm_rows(0.65 * norm_rows(gather_cand(np.load(L / "unl_f0_resnext29_bpv5.npy", mmap_mode="r"), u_cand)) + 0.35 * u_rx_g1)

    q_dn = norm_rows(0.70 * norm_rows(gather_cand(np.load(L / "f0_rot5_densenet169_bpv5.npy", mmap_mode="r"), cand)) + 0.30 * q_dn_g1)
    u_dn = norm_rows(0.70 * norm_rows(gather_cand(np.load(L / "unl_f0_densenet169_bpv5.npy", mmap_mode="r"), u_cand)) + 0.30 * u_dn_g1)

    # Optional: include WRN40 Gen2, SE-PyramidNet, and Spatial Twin Specialist if available
    has_w40 = (L / "oof_wrn40_gen2.npy").exists() and (L / "unl_wrn40_gen2.npy").exists()
    has_pyr = (L / "sepyramid_f0_rot5.npy").exists() and (L / "unl_sepyramid.npy").exists()
    has_sp = (L / "spatial_twin_f0_rot5.npy").exists() and (L / "unl_spatial_twin.npy").exists()

    q_w40 = norm_rows(gather_cand(np.load(L / "oof_wrn40_gen2.npy", mmap_mode="r"), cand, rows=v_idx)) if has_w40 else None
    u_w40 = norm_rows(gather_cand(np.load(L / "unl_f0_wrn40_gen2.npy", mmap_mode="r"), u_cand)) if has_w40 else None
    q_pyr = norm_rows(gather_cand(np.load(L / "sepyramid_f0_rot5.npy", mmap_mode="r"), cand)) if has_pyr else None
    u_pyr = norm_rows(gather_cand(np.load(L / "unl_f0_sepyramid.npy", mmap_mode="r"), u_cand)) if has_pyr else None
    q_sp = norm_rows(gather_cand(np.load(L / "spatial_twin_f0_rot5.npy", mmap_mode="r"), cand)) if has_sp else None
    u_sp = norm_rows(gather_cand(np.load(L / "unl_f0_spatial_twin.npy", mmap_mode="r"), u_cand)) if has_sp else None

    # TRUE 80.92531 Anchor (`rx_v1_fold0_scores.npz` already contains Discovery #1 ConvNeXtV2 0.09 -> 79.9605!)
    with np.load(L / "rx_v1_fold0_scores.npz") as fz1, np.load(L / "rx_v1_unlabeled_f0only.npz") as uz1:
        q_anc = fz1["posteriors"].astype(np.float32)
        u_anc = uz1["posteriors"].astype(np.float32)
    in_c_f0 = (cand == true[v_idx, None]).astype(np.float32)
    anc_brier = float(((q_anc - in_c_f0) ** 2).sum(1).mean())
    anc_top1 = float((cand[np.arange(len(v_idx)), q_anc.argmax(1)] == true[v_idx]).mean())

    torch.cuda.empty_cache()
    free_b, _ = torch.cuda.mem_get_info()
    if free_b < 10 * 1024 ** 3:
        device = "cpu"
        print(f"[grand_slam] only {free_b / 2**30:.2f} GiB free, cavity BP on CPU", flush=True)

    _, _, ul0 = run_unlabeled_bp_hetero(u_cand, u_anc, eu, ev, uy, nof0_tab, known_allowed, s0=s0, s1=s1, device=device)
    _, _, tl0, _, _ = target_cavity_predict(q_anc, oop, cand, one, two, known_allowed, ul0, nof0_tab, debias=debias, device=device)
    _, _, ul1 = run_unlabeled_bp_hetero(u_cand, u_anc, eu, ev, uy, nof0_tab, known_allowed, s0=s0, s1=s1, external_log_avail=tl0, device=device)
    pred_anc, _, _, _, _ = target_cavity_predict(
        adaptive_power_scale(q_anc, np.clip(0.95 - beta_lap * z_lap_f0, 0.55, 1.45)),
        oop, cand, one, two, known_allowed, ul1, nof0_tab, debias=debias, cavity_iters=12, use_escape=False, device=device)
    anc_tune = score_fn(rmse(pred_anc[:split], actual[:split]))
    anc_test = score_fn(rmse(pred_anc[split:], actual[split:]))
    print(f"[grand_slam anchor 80.92531] tune={anc_tune:.4f} test={anc_test:.4f} "
          f"img_top1={anc_top1:.5f} img_brier={anc_brier:.5f}", flush=True)

    # Pure `--train-all` core Grand-Slam recipes (ZERO global non-train-all spatial dilution!)
    # Lesson from 6B LB (80.82792 vs 80.92531): global `sp_wg=0.12` worsened 160k Brier MSE (0.03907 -> 0.03916).
    # Only pure `--train-all` backbones (`sp_wg=0, sp_wa=0`) or strict zero-perturbation `sp_wg=0, sp_wa<=0.06` are permitted!
    recipes = [
        (0.50, 0.30, 0.20, 0.00, 0.00, 0.36, 0.78, 0.00, 0.00),
        (0.52, 0.30, 0.18, 0.00, 0.00, 0.35, 0.75, 0.00, 0.00),
        (0.48, 0.28, 0.24, 0.00, 0.00, 0.36, 0.80, 0.00, 0.00),
        (0.55, 0.27, 0.18, 0.00, 0.00, 0.38, 0.80, 0.00, 0.00),
        (0.46, 0.28, 0.18, 0.08, 0.00, 0.36, 0.78, 0.00, 0.00),
        (0.50, 0.30, 0.20, 0.00, 0.00, 0.34, 0.82, 0.00, 0.00),
        (0.50, 0.30, 0.20, 0.00, 0.00, 0.36, 0.78, 0.00, 0.05),
    ]

    trials = []
    for rx_w, dn_w, cv_w, w4_w, py_w, alpha, temp, sp_wg, sp_wa in recipes:
        eff_w4 = w4_w if has_w40 else 0.0
        eff_py = py_w if has_pyr else 0.0
        tot_w = rx_w + dn_w + cv_w + eff_w4 + eff_py
        stu_q = (rx_w * q_rx + dn_w * q_dn + cv_w * q_conv
                 + (eff_w4 * q_w40 if has_w40 else 0.0)
                 + (eff_py * q_pyr if has_pyr else 0.0)) / tot_w
        stu_u = (rx_w * u_rx + dn_w * u_dn + cv_w * u_conv
                 + (eff_w4 * u_w40 if has_w40 else 0.0)
                 + (eff_py * u_pyr if has_pyr else 0.0)) / tot_w
        stu_q = norm_rows(stu_q)
        stu_u = norm_rows(stu_u)

        # Base ranker + distilled ConvNeXt orthogonal boost
        qb = fuse_posteriors(q_ranker, q_conv, 0.12, "log", 0.85)
        ub = fuse_posteriors(u_ranker, u_conv, 0.12, "log", 0.85)
        qm = fuse_posteriors(qb, stu_q, alpha, "log", temp)
        um = fuse_posteriors(ub, stu_u, alpha, "log", temp)
        if has_sp and (sp_wg > 0.0 or sp_wa > 0.0):
            qm = adaptive_specialist_fuse(qm, q_sp, sp_wg, sp_wa, 0.85)
            um = adaptive_specialist_fuse(um, u_sp, sp_wg, sp_wa, 0.85)

        _, _, ul0 = run_unlabeled_bp_hetero(u_cand, um, eu, ev, uy, nof0_tab, known_allowed, s0=s0, s1=s1, device=device)
        _, _, tl0, _, _ = target_cavity_predict(qm, oop, cand, one, two, known_allowed, ul0, nof0_tab, debias=debias, device=device)
        _, _, ul1 = run_unlabeled_bp_hetero(u_cand, um, eu, ev, uy, nof0_tab, known_allowed, s0=s0, s1=s1, external_log_avail=tl0, device=device)

        best_s = None
        for sc in (0.90, 0.95, 1.00):
            qs = adaptive_power_scale(qm, np.clip(sc - beta_lap * z_lap_f0, 0.55, 1.45))
            pred, top_p, _, _, _ = target_cavity_predict(
                qs, oop, cand, one, two, known_allowed, ul1, nof0_tab,
                debias=debias, cavity_iters=12, use_escape=False, device=device)
            tr_r = rmse(pred[:split], actual[:split])
            te_r = rmse(pred[split:], actual[split:])
            both_ok = float(((top_p[:, 0] == ta) & (top_p[:, 1] == tb))[split:].mean())
            it = {
                "scale_base": sc, "tune_rmse": tr_r, "test_rmse": te_r,
                "tune_score": score_fn(tr_r), "test_score": score_fn(te_r),
                "test_both_correct": both_ok,
            }
            if best_s is None or it["tune_rmse"] < best_s["tune_rmse"]:
                best_s = it
        rec = {
            "recipe": [rx_w, dn_w, cv_w, eff_w4, eff_py, alpha, temp, sp_wg if has_sp else 0.0, sp_wa if has_sp else 0.0],
            "img_top1": float((cand[np.arange(len(v_idx)), qm.argmax(1)] == true[v_idx]).mean()),
            "img_brier": float(((qm - in_c_f0) ** 2).sum(1).mean()),
            "selected": best_s,
        }
        trials.append(rec)
        print(f"[grand_slam_f0] recipe={rec['recipe']} img_top1={rec['img_top1']:.5f} "
              f"brier={rec['img_brier']:.5f} tune={best_s['tune_score']:.4f} "
              f"test={best_s['test_score']:.4f} both={best_s['test_both_correct']:.5f}", flush=True)

    brier_safe = [r for r in trials if r["img_brier"] <= anc_brier + 1e-5 and r["img_top1"] >= anc_top1 - 1e-5]
    pool = brier_safe if brier_safe else trials
    chosen = min(pool, key=lambda r: r["selected"]["tune_rmse"])
    passed = (chosen["selected"]["test_score"] > anc_test + 0.005
              and chosen["img_brier"] <= anc_brier + 1e-5)
    print(f"Milestone 6C Grand-Slam gate: passed={passed} "
          f"(test {anc_test:.4f} -> {chosen['selected']['test_score']:.4f}, "
          f"brier {anc_brier:.5f} -> {chosen['img_brier']:.5f})", flush=True)

    sub_csv = None
    if passed:
        rx_w, dn_w, cv_w, eff_w4, eff_py, alpha, temp, sp_wg, sp_wa = chosen["recipe"]
        tot_w = rx_w + dn_w + cv_w + eff_w4 + eff_py
        stu_q = norm_rows((rx_w * q_rx + dn_w * q_dn + cv_w * q_conv
                           + (eff_w4 * q_w40 if has_w40 else 0.0)
                           + (eff_py * q_pyr if has_pyr else 0.0)) / tot_w)
        stu_u = norm_rows((rx_w * u_rx + dn_w * u_dn + cv_w * u_conv
                           + (eff_w4 * u_w40 if has_w40 else 0.0)
                           + (eff_py * u_pyr if has_pyr else 0.0)) / tot_w)
        win_q = fuse_posteriors(fuse_posteriors(q_ranker, q_conv, 0.12, "log", 0.85), stu_q, alpha, "log", temp)
        win_u = fuse_posteriors(fuse_posteriors(u_ranker, u_conv, 0.12, "log", 0.85), stu_u, alpha, "log", temp)
        if has_sp and (sp_wg > 0.0 or sp_wa > 0.0):
            win_q = adaptive_specialist_fuse(win_q, q_sp, sp_wg, sp_wa, 0.85)
            win_u = adaptive_specialist_fuse(win_u, u_sp, sp_wg, sp_wa, 0.85)
        np.savez_compressed(L / "rx_v6_grand_slam_f0_post.npz", q_post=win_q.astype(np.float32), u_post=win_u.astype(np.float32))

        for model_name, init_pt, lr, bs, short in specs:
            train_full_all_images_student(
                model_name, init_pt, lr=lr, batch_size=bs, short=short, full_ext_pack=full_ext_pack)
        sc_win = chosen["selected"]["scale_base"]

        with np.load(L / "rx_ranker_unlabeled.npz") as uz:
            tu_cand = uz["candidates"].astype(np.int64)
            tu_ranker = uz["posteriors"].astype(np.float32)
            tu_oop = uz["oop_mass"].astype(np.float32)
        with np.load(L / "rx_ranker_test.npz") as tz:
            t_cand = tz["candidates"].astype(np.int64)
            t_ranker = tz["posteriors"].astype(np.float32)
            t_oop = tz["oop_mass"].astype(np.float32)

        # Blend full `--train-all` Gen-1 & Gen-2 Battle-BP models on Unlabeled & Test!
        tu_rx = norm_rows(0.65 * gather_cand(np.load(L / "unl_resnext29_bpv5_full.npy", mmap_mode="r"), tu_cand)
                          + 0.35 * gather_cand(np.load(L / "unl_resnext29_distill.npy", mmap_mode="r"), tu_cand))
        t_rx = norm_rows(0.65 * gather_cand(np.load(L / "test_resnext29_bpv5_full.npy", mmap_mode="r"), t_cand)
                         + 0.35 * gather_cand(np.load(L / "test_resnext29_distill.npy", mmap_mode="r"), t_cand))

        tu_dn = norm_rows(0.70 * gather_cand(np.load(L / "unl_densenet169_bpv5_full.npy", mmap_mode="r"), tu_cand)
                          + 0.30 * gather_cand(np.load(L / "unl_dense_distill.npy", mmap_mode="r"), tu_cand))
        t_dn = norm_rows(0.70 * gather_cand(np.load(L / "test_densenet169_bpv5_full.npy", mmap_mode="r"), t_cand)
                         + 0.30 * gather_cand(np.load(L / "test_dense_distill.npy", mmap_mode="r"), t_cand))

        tu_cv = norm_rows(gather_cand(np.load(L / "unl_convnext_bpv5_full.npy", mmap_mode="r"), tu_cand))
        t_cv = norm_rows(gather_cand(np.load(L / "test_convnext_bpv5_full.npy", mmap_mode="r"), t_cand))

        tu_stu = rx_w * tu_rx + dn_w * tu_dn + cv_w * tu_cv
        t_stu = rx_w * t_rx + dn_w * t_dn + cv_w * t_cv
        if eff_w4 > 0.0:
            tu_stu += eff_w4 * norm_rows(gather_cand(np.load(L / "unl_wrn40_gen2.npy", mmap_mode="r"), tu_cand))
            t_stu += eff_w4 * norm_rows(gather_cand(np.load(L / "test_wrn40_gen2.npy", mmap_mode="r"), t_cand))
        if eff_py > 0.0:
            tu_stu += eff_py * norm_rows(gather_cand(np.load(L / "unl_sepyramid.npy", mmap_mode="r"), tu_cand))
            t_stu += eff_py * norm_rows(gather_cand(np.load(L / "test_sepyramid.npy", mmap_mode="r"), t_cand))
        tu_stu = norm_rows(tu_stu / tot_w)
        t_stu = norm_rows(t_stu / tot_w)

        tu_b = fuse_posteriors(tu_ranker, tu_cv, 0.12, "log", 0.85)
        t_b = fuse_posteriors(t_ranker, t_cv, 0.12, "log", 0.85)
        tu_pri = fuse_posteriors(tu_b, tu_stu, alpha, "log", temp)
        t_post_raw = fuse_posteriors(t_b, t_stu, alpha, "log", temp)

        if has_sp and (sp_wg > 0.0 or sp_wa > 0.0):
            tu_sp = norm_rows(gather_cand(np.load(L / "unl_spatial_twin.npy", mmap_mode="r"), tu_cand))
            t_sp = norm_rows(gather_cand(np.load(L / "test_spatial_twin.npy", mmap_mode="r"), t_cand))
            tu_pri = adaptive_specialist_fuse(tu_pri, tu_sp, sp_wg, sp_wa, 0.85)
            t_post_raw = adaptive_specialist_fuse(t_post_raw, t_sp, sp_wg, sp_wa, 0.85)

        np.savez_compressed(L / "rx_v6_grand_slam_unlabeled.npz",
                            candidates=tu_cand.astype(np.uint16),
                            posteriors=tu_pri.astype(np.float32),
                            oop_mass=tu_oop)
        np.savez_compressed(L / "rx_v6_grand_slam_test.npz",
                            candidates=t_cand.astype(np.uint16),
                            posteriors=t_post_raw.astype(np.float32),
                            oop_mass=t_oop)

        z_lap_test = z_lap_from_gallery(L / "gallery_rx_top40_test.npz", None)
        t_post = adaptive_power_scale(t_post_raw, np.clip(sc_win - beta_lap * z_lap_test, 0.55, 1.45))

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
            tu_cand, tu_pri, eu, ev, uy, full_tab, known_full, s0=s0, s1=s1, device=device)
        for outer in range(2):
            pred, _, test_lnot, _, _ = target_cavity_predict(
                t_post, t_oop, t_cand, idx1, idx2, known_full, u_lnot, full_tab,
                debias=debias, cavity_iters=12, use_escape=False, device=device)
            if outer == 0:
                _, _, u_lnot = run_unlabeled_bp_hetero(
                    tu_cand, tu_pri, eu, ev, uy, full_tab, known_full,
                    s0=s0, s1=s1, external_log_avail=test_lnot, device=device)

        pred = np.clip(pred, 0.0, 10.0)
        sub_csv = L / "submission_rx_v6_grand_slam.csv"
        with sub_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_id", "result"])
            for row_idx, val in enumerate(pred):
                writer.writerow([f"test-{row_idx:06d}", f"{float(val):.6f}"])
        (L / "submission_latest_best.csv").write_bytes(sub_csv.read_bytes())
        print(f"Milestone 6C Grand-Slam complete -> {sub_csv}", flush=True)

    report = {
        "anchor_tune_score": anc_tune,
        "anchor_test_score": anc_test,
        "chosen": chosen,
        "passed": passed,
        "submission_csv": str(sub_csv) if sub_csv else None,
        "trials": trials,
    }
    (L / "milestone6c_grand_slam.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
