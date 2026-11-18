"""Assign each public Dex image to one anonymous competition class.

Inputs are predictions from a fold-safe checkpoint; run independently for
each validation fold. Requires scipy for the Hungarian assignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
                        default=Path("outputs/large/external_probe"))
    parser.add_argument("--model-suffix", default="wrn_f0")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/large/external_probe/consensus_dex_to_class_f0.npy"))
    args = parser.parse_args()
    score_arrays = []
    source_ids = None
    for variant in ("home", "official-artwork"):
        file = args.input_dir / f"{variant}_{args.model_suffix}_mapping.npz"
        with np.load(file) as data:
            ids = data["source_ids"]
            if source_ids is not None and not np.array_equal(ids, source_ids):
                raise ValueError("Source IDs differ between sprite variants")
            source_ids = ids
            score_arrays.append(data["scores"].astype(np.float32))
    if source_ids is None or not np.array_equal(source_ids, np.arange(1, 1026)):
        raise ValueError("Expected all 1,025 National Dex numbers")
    scores = np.mean(score_arrays, axis=0)
    rows, classes = linear_sum_assignment(-scores)
    if not np.array_equal(rows, np.arange(1025)):
        raise ValueError("Assignment rows are not ordered by Dex number")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, (classes + 1).astype(np.uint16))
    independent = [array.argmax(axis=1) for array in score_arrays]
    report = {
        "variant_agreement": int(np.sum(independent[0] == independent[1])),
        "assignment_conflicts": (np.flatnonzero(classes != scores.argmax(axis=1)) + 1).tolist(),
        "median_assigned_confidence": float(np.median(scores[rows, classes])),
        "min_assigned_confidence": float(np.min(scores[rows, classes])),
        "mapping_file": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
