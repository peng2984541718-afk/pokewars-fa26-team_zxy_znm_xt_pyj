"""One step past the α=0.45 / T=0.75 corner of the ResNeXt fusion grid.

Same fold-0 protocol as select_resnext_fusion.py. The known winner is included
so the scorer can be checked against 78.6976 / 78.9263.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mix_distill_posteriors import fuse_posteriors
from sweep_distill_fusion import L, load_context, pick_scale, score_posteriors

ALPHAS = (0.45, 0.50, 0.55, 0.60, 0.65)
TEMPERATURES = (0.55, 0.60, 0.65, 0.70, 0.75)
DENSE_SHARES = (0.0, 0.15, 0.30)
CURRENT_TEST = 78.92630690294074


def gather(path: Path, candidates: np.ndarray) -> np.ndarray:
    matrix = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    return np.take_along_axis(matrix, candidates.astype(np.int64) - 1, axis=1)


def main() -> None:
    ctx = load_context("cuda")
    dense = gather(L / "distill_pilot_f0_probs.npy", ctx["cand"])
    resnext = gather(L / "resnext29_dist_f0_probs.npy", ctx["cand"])
    records = []
    grid = [(alpha, temperature, share)
            for share in DENSE_SHARES for alpha in ALPHAS for temperature in TEMPERATURES]
    for index, (alpha, temperature, share) in enumerate(grid, start=1):
        blended = share * dense + (1.0 - share) * resnext
        blended /= np.maximum(blended.sum(1, keepdims=True), 1e-8)
        mixed = fuse_posteriors(ctx["post"], blended, alpha, "log", temperature)
        chosen = pick_scale(score_posteriors(ctx, mixed))
        record = {"alpha": alpha, "temperature": temperature, "dense_share": share,
                  "resnext_share": round(1.0 - share, 2), "selected": chosen}
        records.append(record)
        print(f"[{index:02d}/{len(grid)}] log alpha={alpha:.2f} T={temperature:<4} "
              f"dense={share:.2f} tune={chosen['tune_score']:.4f} "
              f"test={chosen['test_score']:.4f} scale={chosen['scale']:.2f} "
              f"both={chosen['test_both_correct']:.5f}", flush=True)

    selected = min(records, key=lambda item: (
        item["selected"]["tune_rmse"], item["alpha"], abs(item["temperature"] - 1.0),
        -item["dense_share"]))
    chosen = selected["selected"]
    anchor = next(item for item in records
                  if item["alpha"] == 0.45 and item["temperature"] == 0.75
                  and item["dense_share"] == 0.30)
    report = {
        "anchor_alpha_0.45_T_0.75_dense_0.30": anchor,
        "selected_on_tune": selected,
        "beats_current_test_half": chosen["test_score"] > CURRENT_TEST,
        "current_test_half": CURRENT_TEST,
        "ranked": records,
    }
    path = L / "resnext_fusion_outer.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"anchor tune={anchor['selected']['tune_score']:.4f} "
          f"test={anchor['selected']['test_score']:.4f}", flush=True)
    print(f"selected alpha={selected['alpha']} T={selected['temperature']} "
          f"dense={selected['dense_share']} tune={chosen['tune_score']:.4f} "
          f"test={chosen['test_score']:.4f} beats={chosen['test_score'] > CURRENT_TEST}",
          flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
