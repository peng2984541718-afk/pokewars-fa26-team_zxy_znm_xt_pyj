"""Fold-0 selection of a log-space mix between the ranker, DenseNet, and ResNeXt.

Both student files are leak-free on fold 0. The winning weights are the ones
with the lowest tune RMSE. A new submission is justified only when that
choice also raises the untouched half above the previous 78.457.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from mix_distill_posteriors import fuse_posteriors
from sweep_distill_fusion import L, load_context, pick_scale, score_posteriors

ALPHAS = (0.25, 0.35, 0.45)
TEMPERATURES = (0.75, 1.0, 1.25)
# DenseNet share of the student blend. The rest is ResNeXt.
DENSE_SHARES = (1.0, 0.7, 0.5, 0.3, 0.0)
PREVIOUS_TEST = 78.457


def gather(path: Path, candidates: np.ndarray) -> np.ndarray:
    matrix = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    return np.take_along_axis(matrix, candidates.astype(np.int64) - 1, axis=1)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = load_context(device)
    dense = gather(L / "distill_pilot_f0_probs.npy", ctx["cand"])
    resnext = gather(L / "resnext29_dist_f0_probs.npy", ctx["cand"])
    if dense.shape != resnext.shape:
        raise SystemExit(f"student shapes differ: {dense.shape} vs {resnext.shape}")

    records = []
    grid = [(alpha, temperature, share)
            for share in DENSE_SHARES for alpha in ALPHAS for temperature in TEMPERATURES]
    for index, (alpha, temperature, share) in enumerate(grid, start=1):
        blended = share * dense + (1.0 - share) * resnext
        blended /= np.maximum(blended.sum(1, keepdims=True), 1e-8)
        mixed = fuse_posteriors(ctx["post"], blended, alpha, "log", temperature)
        chosen = pick_scale(score_posteriors(ctx, mixed))
        record = {"alpha": alpha, "temperature": temperature, "dense_share": share,
                  "resnext_share": 1.0 - share, "selected": chosen}
        records.append(record)
        print(f"[{index:02d}/{len(grid)}] log alpha={alpha:.2f} T={temperature:<4} "
              f"dense={share:.1f} tune={chosen['tune_score']:.4f} "
              f"test={chosen['test_score']:.4f} scale={chosen['scale']:.2f}", flush=True)

    selected = min(records, key=lambda item: (
        item["selected"]["tune_rmse"], item["alpha"], abs(item["temperature"] - 1.0),
        -item["dense_share"]))
    chosen = selected["selected"]
    passed = chosen["test_score"] > PREVIOUS_TEST
    report = {"selected_on_tune": selected, "passed_previous_test_half": passed,
              "previous_test_half": PREVIOUS_TEST, "ranked": records}
    path = L / "resnext_fusion_sweep.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"selected alpha={selected['alpha']} T={selected['temperature']} "
          f"dense={selected['dense_share']} test={chosen['test_score']:.4f} "
          f"passed={passed}", flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
