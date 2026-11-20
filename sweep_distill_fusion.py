"""Select distill fusion on the fold-0 half split, then submit only on a real gain.

The leak-free student on labeled fold 0 is the DenseNet pilot
(`distill_pilot_f0_probs.npy`), trained with that fold held out. The full-data
DenseNet and WRN-28 files are `--train-all` exports on unlabeled and test
images, so their internal ratio cannot be chosen on this split. Selection
follows `joint_constrained_inference.py --mode evaluate-fold0`: the posterior
scale is chosen on the tune half, and the paired test-half score is only read
after the fusion parameters are fixed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_soft_score import DENOMINATOR, rmse
from image_experiment import SEED, stratified_folds
from joint_constrained_inference import (
    power_scale_posteriors,
    run_submission,
    run_unlabeled_bp,
    target_cavity_predict,
)
from mix_distill_posteriors import fuse_posteriors, mix

L = Path("outputs/large")
ALPHAS = (0.25, 0.35, 0.45, 0.55, 0.65)
TEMPERATURES = (0.5, 0.75, 1.0, 1.5, 2.0)
MODES = ("linear", "log")
SCALES = (0.85, 0.90, 0.95, 1.00, 1.05)
GATE = 78.42
# Previous submission mix. Not identifiable on the fold-0 half split.
STUDENT_WEIGHTS = (0.6, 0.4)


def score_fn(value: float) -> float:
    return 100.0 * (1.0 - value / DENOMINATOR)


def pick_scale(trials: list[dict]) -> dict:
    selected = trials[0]
    for item in trials[1:]:
        if item["tune_rmse"] < selected["tune_rmse"]:
            selected = item
    return selected


def candidate_student(candidates: np.ndarray, probs: Path) -> np.ndarray:
    gathered = np.take_along_axis(
        np.asarray(np.load(probs, mmap_mode="r"), dtype=np.float32),
        candidates.astype(np.int64) - 1, axis=1)
    gathered /= np.maximum(gathered.sum(1, keepdims=True), 1e-8)
    return np.ascontiguousarray(gathered)


def load_context(device: str) -> dict:
    rows = pd.read_csv(Path("data/train.csv.gz"))
    identity = {}
    for side in (1, 2):
        for path, value in zip(rows[f"noisy_image{side}"], rows[f"pokemon_id{side}"]):
            if pd.notna(value):
                identity[path] = int(value)
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

    with np.load(L / "phase2_imp_ranker_unlabeled_f0only.npz") as uz:
        u_cand = uz["candidates"].astype(np.int64)
        u_pri = uz["posteriors"].astype(np.float32)
    with np.load(L / "phase2_imp_ranker_fold0_scores.npz") as fz:
        v_idx = fz["valid_indices"]
        cand = fz["candidates"].astype(np.int64)
        post = fz["posteriors"].astype(np.float32)
        oop = fz["oop_mass"].astype(np.float32)
    with np.load(L / "distill_pilot_fold0_scores.npz") as fz:
        if not np.array_equal(fz["valid_indices"], v_idx):
            raise SystemExit("pilot fold-0 rows do not match the ranker")
        if not np.array_equal(fz["candidates"].astype(np.int64), cand):
            raise SystemExit("pilot fold-0 candidates do not match the ranker")
        pilot_post = fz["posteriors"].astype(np.float32)

    pos = np.full(len(paths), -1, dtype=np.int32)
    pos[v_idx] = np.arange(len(v_idx))
    one, two = pos[f1[pair_rows]], pos[f2[pair_rows]]
    if (one < 0).any() or (two < 0).any():
        raise SystemExit("a fold-0 battle image is missing from the ranker rows")

    print("caching unlabeled messages that do not depend on the query mix", flush=True)
    _, _, base_unl_lnot = run_unlabeled_bp(
        u_cand, u_pri, eu, ev, uy, table, known_allowed, device=device)
    return {
        "a": a, "b": b, "pair_rows": pair_rows, "split": split, "actual": y[pair_rows],
        "one": one, "two": two,
        "table": table, "known_allowed": known_allowed,
        "u_cand": u_cand, "u_pri": u_pri, "eu": eu, "ev": ev, "uy": uy,
        "cand": cand, "post": post, "oop": oop, "pilot_post": pilot_post,
        "student": candidate_student(cand, L / "distill_pilot_f0_probs.npy"),
        "base_unl_lnot": base_unl_lnot, "device": device,
    }


def score_posteriors(ctx: dict, post: np.ndarray) -> list[dict]:
    device = ctx["device"]
    _, _, test_lnot, _, _ = target_cavity_predict(
        post, ctx["oop"], ctx["cand"], ctx["one"], ctx["two"],
        ctx["known_allowed"], ctx["base_unl_lnot"], ctx["table"], device=device)
    _, _, unl_lnot = run_unlabeled_bp(
        ctx["u_cand"], ctx["u_pri"], ctx["eu"], ctx["ev"], ctx["uy"],
        ctx["table"], ctx["known_allowed"], external_log_avail=test_lnot, device=device)
    actual = ctx["actual"]
    split = ctx["split"]
    truth = ctx["a"][ctx["pair_rows"]], ctx["b"][ctx["pair_rows"]]
    trials = []
    for scale in SCALES:
        qs = power_scale_posteriors(post, scale)
        pred, top_p, _, _, _ = target_cavity_predict(
            qs, ctx["oop"], ctx["cand"], ctx["one"], ctx["two"],
            ctx["known_allowed"], unl_lnot, ctx["table"],
            debias=1.0, cavity_iters=12, use_escape=False, device=device)
        both_ok = float(((top_p[:, 0] == truth[0]) & (top_p[:, 1] == truth[1]))[split:].mean())
        tune = rmse(pred[:split], actual[:split])
        test = rmse(pred[split:], actual[split:])
        trials.append({
            "scale": scale,
            "tune_rmse": tune,
            "test_rmse": test,
            "tune_score": score_fn(tune),
            "test_score": score_fn(test),
            "test_both_correct": both_ok,
        })
    return trials


def config_key(item: dict) -> tuple:
    chosen = item["selected"]
    return (chosen["tune_rmse"], item["alpha"], abs(item["temperature"] - 1.0),
            0 if item["mode"] == "linear" else 1, item["temperature"])


def write_submission(selected: dict) -> dict:
    alpha = selected["alpha"]
    mode = selected["mode"]
    temperature = selected["temperature"]
    scale = selected["selected"]["scale"]
    unlabeled = L / "distill_fusion_unlabeled.npz"
    test = L / "distill_fusion_test.npz"
    output = L / "submission_distill_fusion.csv"
    mix(L / "phase2_imp_ranker_unlabeled.npz",
        [(L / "unl_dense_distill.npy", STUDENT_WEIGHTS[0]),
         (L / "unl_wrn28_distill.npy", STUDENT_WEIGHTS[1])],
        unlabeled, None, alpha, mode, temperature)
    mix(L / "phase2_imp_ranker_test.npz",
        [(L / "test_dense_distill.npy", STUDENT_WEIGHTS[0]),
         (L / "test_wrn28_distill.npy", STUDENT_WEIGHTS[1])],
        test, None, alpha, mode, temperature)
    args = argparse.Namespace(
        battles=Path("data/known_battles.npz"),
        train_csv=Path("data/train.csv.gz"),
        test_csv=Path("data/test.csv.gz"),
        unlabeled_paths=Path("data/unlabeled_paths.txt"),
        data_dir=Path("."),
        unlabeled_priors=unlabeled,
        test_npz=test,
        table=L / "pair_semi_v23_blend_table.npy",
        scale=scale,
        debias=1.0,
        cavity_iters=12,
        use_escape=False,
        save_pseudo_battles=None,
        save_transductive_bank=None,
        bank_min_conf=0.92,
        test_bank_min_conf=0.93,
        output=output,
    )
    run_submission(args, "cuda" if torch.cuda.is_available() else "cpu")
    return {"unlabeled": str(unlabeled), "test": str(test), "csv": str(output),
            "scale": scale, "student_weights": list(STUDENT_WEIGHTS)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=L / "distill_fusion_sweep.json")
    parser.add_argument("--no-submit", action="store_true",
                        help="Score the grid and skip the test-set submission")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("fold-0 joint inference needs the GPU; CPU would not finish in minutes")
    started = time.time()
    ctx = load_context(device)

    reference = json.loads((L / "score_fold0_distill_pilot.json").read_text())["selected"]
    print("reproducing the saved linear alpha=0.25 pilot pack", flush=True)
    repro_trials = score_posteriors(ctx, ctx["pilot_post"])
    reproduced = pick_scale(repro_trials)
    score_gap = abs(reproduced["test_score"] - reference["test_score"])
    print(f"reproduction test {reproduced['test_score']:.6f} "
          f"saved {reference['test_score']:.6f} gap {score_gap:.6f}", flush=True)
    if score_gap > 0.02 or abs(reproduced["tune_score"] - reference["tune_score"]) > 0.02:
        raise SystemExit("fold-0 scorer does not reproduce score_fold0_distill_pilot.json")
    if reproduced["scale"] != reference["scale"]:
        raise SystemExit(f"reproduced scale {reproduced['scale']} != saved {reference['scale']}")

    rebuilt = fuse_posteriors(ctx["post"], ctx["student"], 0.25, "linear", 1.0)
    post_gap = float(np.max(np.abs(rebuilt - ctx["pilot_post"])))
    print(f"linear alpha=0.25 T=1 posterior max abs diff vs saved pack {post_gap:.3e}", flush=True)
    if post_gap > 1e-5:
        raise SystemExit("fusion does not reproduce distill_pilot_fold0_scores.npz")

    configs = []
    grid = [(mode, alpha, temperature)
            for mode in MODES for alpha in ALPHAS for temperature in TEMPERATURES]
    for i, (mode, alpha, temperature) in enumerate(grid, start=1):
        tick = time.time()
        if mode == "linear" and alpha == 0.25 and temperature == 1.0:
            chosen = reproduced
            scales = repro_trials
        else:
            post = fuse_posteriors(ctx["post"], ctx["student"], alpha, mode, temperature)
            scales = score_posteriors(ctx, post)
            chosen = pick_scale(scales)
        configs.append({"mode": mode, "alpha": alpha, "temperature": temperature,
                         "selected": chosen, "scales": scales})
        print(f"[{i:02d}/{len(grid)}] {mode} alpha={alpha:.2f} T={temperature:<4} "
              f"scale={chosen['scale']:.2f} tune={chosen['tune_score']:.4f} "
              f"test={chosen['test_score']:.4f} {time.time() - tick:.1f}s", flush=True)

    selected = min(configs, key=config_key)
    passed = selected["selected"]["test_score"] > GATE
    submission = None
    if passed and not args.no_submit:
        print("test half cleared 78.42; writing the submission with the tune-selected mix",
              flush=True)
        submission = write_submission(selected)
    elif passed:
        print("test half cleared 78.42; submission skipped by --no-submit", flush=True)
    else:
        print("test half did not clear 78.42; no submission written", flush=True)

    report = {
        "protocol": (
            "Fold-0 battles are split in half with seed 3780, matching "
            "joint_constrained_inference.evaluate. Fusion (mode, alpha, T) and "
            "the posterior scale are chosen by tune RMSE. The test-half score "
            "is read only for the selected setting. Unlabeled priors stay on "
            "phase2_imp_ranker_unlabeled_f0only.npz."
        ),
        "gate": GATE,
        "passed_gate": passed,
        "reproduction": {
            "saved": reference,
            "rerun": reproduced,
            "posterior_max_abs": post_gap,
        },
        "internal_ratio": {
            "swept": False,
            "submission_weights": {"densenet": STUDENT_WEIGHTS[0], "wrn28": STUDENT_WEIGHTS[1]},
            "reason": (
                "Fold-0 leak-free probabilities exist only for the DenseNet pilot "
                "(outputs/large/distill_pilot_f0_probs.npy, train_all=false). "
                "outputs/large/{unl,test}_{dense,wrn28}_distill.npy are full-data "
                "models trained with --train-all, so a DenseNet:WRN weight cannot "
                "be scored on this half split. The submission keeps 0.6:0.4."
            ),
        },
        "grid": {"modes": list(MODES), "alphas": list(ALPHAS),
                 "temperatures": list(TEMPERATURES), "scales": list(SCALES)},
        "selected_on_tune": selected,
        "submission": submission,
        "elapsed_sec": time.time() - started,
        "ranked_by_tune": [
            {"mode": item["mode"], "alpha": item["alpha"], "temperature": item["temperature"],
             **item["selected"]}
            for item in sorted(configs, key=config_key)
        ],
    }
    report["configs"] = configs
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    chosen = selected["selected"]
    print(f"selected {selected['mode']} alpha={selected['alpha']} "
          f"T={selected['temperature']} scale={chosen['scale']} "
          f"tune={chosen['tune_score']:.4f} test={chosen['test_score']:.4f} "
          f"gate={GATE} passed={passed}", flush=True)
    print(f"wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()
