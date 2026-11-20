"""Assemble three disjoint fold predictions into a full OOF matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from image_experiment import NUM_CLASSES, SEED, stratified_folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    folds = stratified_folds(labels, 3, SEED)
    result = np.empty((len(labels), NUM_CLASSES), dtype=np.float16)
    seen = np.zeros(len(labels), dtype=bool)
    for fold in range(3):
        z = np.load(args.input_dir / f"{args.prefix}_f{fold}_oof.npz")
        indices = z["indices"]
        if (np.any(folds[indices] != fold) or np.any(seen[indices]) or
                not np.array_equal(z["labels"], labels[indices]) or
                z["probabilities"].shape != (len(indices), NUM_CLASSES)):
            raise ValueError(f"Fold {fold} prediction cache mismatch")
        result[indices] = z["probabilities"]
        seen[indices] = True
    if not np.all(seen):
        raise ValueError("OOF predictions do not cover every image")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result)
    print(f"Saved {len(labels)} OOF rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
