"""Milestone 4: Gen-2 3-fold soft distillation of DenseNet-169 and WRN-40-10,
6-view TTA export, leak-free Fold-0 student-mix selection, and
submission_rx_v4_gen2distill.csv generation.
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

from evaluate_soft_score import DENOMINATOR, rmse
from image_experiment import SEED, stratified_folds
from joint_constrained_inference import target_cavity_predict
from milestone1_rx_fusion_and_pack import gather_cand
from milestone3_inference_calibration import (
    adaptive_power_scale,
    run_unlabeled_bp_hetero,
    z_lap_from_gallery,
)
from mix_distill_posteriors import fuse_posteriors

L = Path("outputs/large")
ALPHAS = (0.25, 0.35, 0.45, 0.55)
TEMPERATURES = (0.65, 0.75, 0.85, 1.00)
# (resnext_w, dense_gen2_w, wrn40_gen2_w)
STUDENT_RATIOS = (
    (1.00, 0.00, 0.00),
    (0.70, 0.30, 0.00),
    (0.60, 0.25, 0.15),
    (0.50, 0.30, 0.20),
    (0.55, 0.35, 0.10),
    (0.45, 0.35, 0.20),
)


def score_fn(val: float) -> float:
    return 100.0 * (1.0 - val / DENOMINATOR)


def run_cmd(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN:", " ".join(cmd), f"-> {log_path}", flush=True)
    with log_path.open("w") as handle:
        subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=True)


def distill_three_folds(model_name: str, init_prefix: str, out_prefix: str,
                        lr: float, batch_size: int) -> None:
    py = sys.executable
    for fold in (0, 1, 2):
        best_pt = L / f"{out_prefix}_f{fold}_best.pt"
        if best_pt.exists():
            print(f"skip existing {best_pt.name}", flush=True)
            continue
        cmd = [
            py, "-u", "train_large_image_external.py",
            "--model", model_name,
            "--fold", str(fold),
            "--training-seed", "314159",
            "--batch-size", str(batch_size),
            "--eval-batch-size", "256",
            "--clean-impulses",
            "--epochs", "12",
            "--warmup-epochs", "1",
            "--min-epochs", "8",
            "--patience", "6",
            "--min-improvement", "0.0002",
            "--learning-rate", f"{lr}",
            "--min-learning-rate", "1e-6",
            "--translate-pixels", "1.0",
            "--rotate-degrees", "8",
            "--scale-min", "0.95",
            "--scale-max", "1.05",
            "--initial-weights", str(L / f"{init_prefix}_f{fold}_best.pt"),
            "--distill-labeled", str(L / "gen2_distill_labeled_oof.npz"),
            "--distill-alpha", "0.55",
            "--data-dir", ".",
            "--output-dir", str(L),
            "--run-name", f"{out_prefix}_f{fold}",
        ]
        run_cmd(cmd, Path("logs") / f"{out_prefix}_f{fold}.log")
        meta = json.loads((L / f"{out_prefix}_f{fold}_best.json").read_text())
        print(f"{out_prefix}_f{fold}: best_ep={meta['epoch']} val_top1={meta['top1']:.5f}",
              flush=True)


def export_rot5_all(model_name: str, prefix: str, short: str) -> None:
    py = sys.executable
    jobs = [
        (["--split", "oof", "--folds", "0", "1", "2"], L / f"oof_{short}.npy"),
        (["--split", "query", "--folds", "0", "--query-dir", "unl"], L / f"unl_f0_{short}.npy"),
        (["--split", "query", "--folds", "0", "1", "2", "--query-dir", "unl"], L / f"unl_{short}.npy"),
        (["--split", "query", "--folds", "0", "1", "2", "--query-dir", "."], L / f"test_{short}.npy"),
    ]
    for extra, out_path in jobs:
        if out_path.exists():
            continue
        cmd = [
            py, "-u", "export_probs_rot5.py",
            "--model", model_name,
            "--prefixes", prefix,
            "--clean-impulses",
            "--batch-size", "256",
            *extra,
            "--output", str(out_path),
        ]
        run_cmd(cmd, Path("logs") / f"export_{out_path.stem}.log")


def blend_gathered(mats: list[np.ndarray], weights: tuple[float, ...]) -> np.ndarray:
    out = np.zeros_like(mats[0], dtype=np.float32)
    for m, w in zip(mats, weights):
        if w > 0.0:
            out += float(w) * m
    out /= np.maximum(out.sum(1, keepdims=True), 1e-8)
    return out


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Distill 3 folds of DenseNet-169 and 3 folds of WRN-40-10 from Gen-2 OOF teacher
    distill_three_folds("densenet169_cifar", "densenet169_imp", "densenet169_gen2",
                        lr=1.0e-4, batch_size=128)
    export_rot5_all("densenet169_cifar", "densenet169_gen2", "dense_gen2")

    distill_three_folds("wrn40_10", "wrn40_10_imp", "wrn40_gen2",
                        lr=8.0e-5, batch_size=128)
    export_rot5_all("wrn40_10", "wrn40_gen2", "wrn40_gen2")

    # 2. Load active table & calibration settings from Milestones 2 & 3
    m2 = json.loads((L / "milestone2_pair_table_v4.json").read_text())
    m3 = json.loads((L / "milestone3_calibration.json").read_text())
    nof0_table = np.load(Path(m2["active_nof0_table"]))
    full_table = np.load(Path(m2["active_full_table"]))
    cal = m3["active"]
    s0, s1 = float(cal["s0"]), float(cal["s1"])
    debias = float(cal["debias"])
    scale_base = float(cal["scale_base"])
    beta_lap = float(cal["beta_lap"])
    baseline_test_score = float(cal["test_score"])

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
        u_base_post = uz["posteriors"].astype(np.float32)
    with np.load(L / "rx_ranker_fold0_scores.npz") as fz:
        v_idx = fz["valid_indices"].astype(np.int64)
        cand = fz["candidates"].astype(np.int64)
        q_base_post = fz["posteriors"].astype(np.float32)
        oop = fz["oop_mass"].astype(np.float32)

    z_lap_f0 = z_lap_from_gallery(L / "gallery_rx_top40_f0.npz", v_idx)
    pos = np.full(len(paths), -1, dtype=np.int32)
    pos[v_idx] = np.arange(len(v_idx))
    one, two = pos[f1[pair_rows]], pos[f2[pair_rows]]
    actual = y[pair_rows]
    ta, tb = a[pair_rows], b[pair_rows]

    q_mats = [
        gather_cand(np.load(L / "oof_resnext29_dist.npy", mmap_mode="r"), cand, rows=v_idx),
        gather_cand(np.load(L / "oof_dense_gen2.npy", mmap_mode="r"), cand, rows=v_idx),
        gather_cand(np.load(L / "oof_wrn40_gen2.npy", mmap_mode="r"), cand, rows=v_idx),
    ]
    u_mats = [
        gather_cand(np.load(L / "unl_f0_resnext29_dist.npy", mmap_mode="r"), u_cand),
        gather_cand(np.load(L / "unl_f0_dense_gen2.npy", mmap_mode="r"), u_cand),
        gather_cand(np.load(L / "unl_f0_wrn40_gen2.npy", mmap_mode="r"), u_cand),
    ]

    # Fast coarse screen with fixed base_unl_lnot, then exact coupled BP on top 8
    _, _, base_unl_lnot = run_unlabeled_bp_hetero(
        u_cand, u_base_post, eu, ev, uy, nof0_table, known_allowed,
        s0=s0, s1=s1, device=device)

    coarse = []
    for ratio in STUDENT_RATIOS:
        stu_q = blend_gathered(q_mats, ratio)
        for alpha in ALPHAS:
            for temp in TEMPERATURES:
                q_mix = fuse_posteriors(q_base_post, stu_q, alpha, "log", temp)
                _, _, test_lnot, _, _ = target_cavity_predict(
                    q_mix, oop, cand, one, two, known_allowed, base_unl_lnot,
                    nof0_table, debias=debias, device=device)
                _, _, unl_lnot = run_unlabeled_bp_hetero(
                    u_cand, u_base_post, eu, ev, uy, nof0_table, known_allowed,
                    s0=s0, s1=s1, external_log_avail=test_lnot, device=device)
                best_s = None
                for sc in (scale_base - 0.05, scale_base, scale_base + 0.05):
                    s_vec = np.clip(sc - beta_lap * z_lap_f0, 0.55, 1.45)
                    qs = adaptive_power_scale(q_mix, s_vec)
                    pred, top_p, _, _, _ = target_cavity_predict(
                        qs, oop, cand, one, two, known_allowed, unl_lnot, nof0_table,
                        debias=debias, cavity_iters=12, use_escape=False, device=device)
                    tr_r = rmse(pred[:split], actual[:split])
                    te_r = rmse(pred[split:], actual[split:])
                    both_ok = float(((top_p[:, 0] == ta) & (top_p[:, 1] == tb))[split:].mean())
                    it = {
                        "scale_base": round(sc, 4), "tune_rmse": tr_r, "test_rmse": te_r,
                        "tune_score": score_fn(tr_r), "test_score": score_fn(te_r),
                        "test_both_correct": both_ok,
                    }
                    if best_s is None or it["tune_rmse"] < best_s["tune_rmse"]:
                        best_s = it
                coarse.append({
                    "ratio": list(ratio), "alpha": alpha, "temperature": temp,
                    "selected": best_s,
                })

    coarse.sort(key=lambda r: r["selected"]["tune_rmse"])
    exact = []
    for item in coarse[:8]:
        ratio = tuple(item["ratio"])
        alpha = item["alpha"]
        temp = item["temperature"]
        stu_q = blend_gathered(q_mats, ratio)
        stu_u = blend_gathered(u_mats, ratio)
        q_mix = fuse_posteriors(q_base_post, stu_q, alpha, "log", temp)
        u_mix = fuse_posteriors(u_base_post, stu_u, alpha, "log", temp)

        _, _, unl_lnot0 = run_unlabeled_bp_hetero(
            u_cand, u_mix, eu, ev, uy, nof0_table, known_allowed,
            s0=s0, s1=s1, device=device)
        _, _, test_lnot, _, _ = target_cavity_predict(
            q_mix, oop, cand, one, two, known_allowed, unl_lnot0, nof0_table,
            debias=debias, device=device)
        _, _, unl_lnot = run_unlabeled_bp_hetero(
            u_cand, u_mix, eu, ev, uy, nof0_table, known_allowed,
            s0=s0, s1=s1, external_log_avail=test_lnot, device=device)

        best_s = None
        for sc in (0.85, 0.90, 0.95, 1.00, 1.05):
            s_vec = np.clip(sc - beta_lap * z_lap_f0, 0.55, 1.45)
            qs = adaptive_power_scale(q_mix, s_vec)
            pred, top_p, _, _, _ = target_cavity_predict(
                qs, oop, cand, one, two, known_allowed, unl_lnot, nof0_table,
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
            "ratio": list(ratio), "alpha": alpha, "temperature": temp,
            "selected": best_s,
        }
        exact.append(rec)
        print(f"[gen2_exact] ratio={ratio} alpha={alpha:.2f} T={temp:.2f} "
              f"tune={best_s['tune_score']:.4f} test={best_s['test_score']:.4f} "
              f"both={best_s['test_both_correct']:.5f}", flush=True)

    winner = min(exact, key=lambda r: (r["selected"]["tune_rmse"], r["alpha"]))
    passed = winner["selected"]["test_score"] > baseline_test_score
    print(f"Milestone 4 gate: passed={passed} "
          f"(test {baseline_test_score:.4f} -> {winner['selected']['test_score']:.4f})",
          flush=True)

    sub_csv = None
    if passed:
        ratio = tuple(winner["ratio"])
        alpha = winner["alpha"]
        temp = winner["temperature"]
        sc_win = winner["selected"]["scale_base"]

        with np.load(L / "rx_ranker_unlabeled.npz") as uz:
            tu_cand = uz["candidates"].astype(np.int64)
            tu_base = uz["posteriors"].astype(np.float32)
            tu_oop = uz["oop_mass"].astype(np.float32)
        with np.load(L / "rx_ranker_test.npz") as tz:
            t_cand = tz["candidates"].astype(np.int64)
            t_base = tz["posteriors"].astype(np.float32)
            t_oop = tz["oop_mass"].astype(np.float32)

        tu_stu = blend_gathered([
            gather_cand(np.load(L / "unl_resnext29_dist.npy", mmap_mode="r"), tu_cand),
            gather_cand(np.load(L / "unl_dense_gen2.npy", mmap_mode="r"), tu_cand),
            gather_cand(np.load(L / "unl_wrn40_gen2.npy", mmap_mode="r"), tu_cand),
        ], ratio)
        t_stu = blend_gathered([
            gather_cand(np.load(L / "test_resnext29_dist.npy", mmap_mode="r"), t_cand),
            gather_cand(np.load(L / "test_dense_gen2.npy", mmap_mode="r"), t_cand),
            gather_cand(np.load(L / "test_wrn40_gen2.npy", mmap_mode="r"), t_cand),
        ], ratio)

        tu_pri = fuse_posteriors(tu_base, tu_stu, alpha, "log", temp)
        t_post_raw = fuse_posteriors(t_base, t_stu, alpha, "log", temp)

        np.savez_compressed(L / "rx_v4_unlabeled.npz",
                            candidates=tu_cand.astype(np.uint16),
                            posteriors=tu_pri.astype(np.float32),
                            oop_mass=tu_oop)
        np.savez_compressed(L / "rx_v4_test.npz",
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
            tu_cand, tu_pri, eu, ev, uy, full_table, known_full,
            s0=s0, s1=s1, device=device)
        for outer in range(2):
            pred, _, test_lnot, _, _ = target_cavity_predict(
                t_post, t_oop, t_cand, idx1, idx2, known_full, u_lnot, full_table,
                debias=debias, cavity_iters=12, use_escape=False, device=device)
            if outer == 0:
                _, _, u_lnot = run_unlabeled_bp_hetero(
                    tu_cand, tu_pri, eu, ev, uy, full_table, known_full,
                    s0=s0, s1=s1, external_log_avail=test_lnot, device=device)

        pred = np.clip(pred, 0.0, 10.0)
        sub_csv = L / "submission_rx_v4_gen2distill.csv"
        with sub_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_id", "result"])
            for row_idx, val in enumerate(pred):
                writer.writerow([f"test-{row_idx:06d}", f"{float(val):.6f}"])
        print(f"Milestone 4 complete -> {sub_csv}", flush=True)

    report = {
        "baseline_test_score": baseline_test_score,
        "winner": winner,
        "passed": passed,
        "submission_csv": str(sub_csv) if sub_csv else None,
        "exact_shortlist": exact,
    }
    (L / "milestone4_gen2_distill.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
