"""Tune probability sharpness on fold 0, then verify on untouched folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from image_experiment import SEED, stratified_folds


def normalize_power(p: np.ndarray, gamma: float) -> np.ndarray:
    powered = np.maximum(p.astype(np.float32), 1e-12)**gamma
    return powered / powered.sum(1, keepdims=True)


def evaluate(scores: np.ndarray, y: np.ndarray) -> dict:
    top10 = np.argpartition(scores, -10, axis=1)[:, -10:]
    return {"top1": float(np.mean(scores.argmax(1) == y)),
            "top10": float(np.mean((top10 == y[:, None]).any(1)))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-oof", type=Path, default=None,
                        help="Single-fold NPZ; omit for full assembled DenseNet OOF")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    all_labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    folds = stratified_folds(all_labels, 3, SEED)
    if args.dense_oof is None:
        indices = np.arange(len(all_labels))
        dense = np.load(args.input_dir / "multibackbone_densenet169_oof.npy",
                        mmap_mode="r")
    else:
        z = np.load(args.dense_oof)
        indices = z["indices"]
        dense = z["probabilities"]
        if not np.array_equal(z["labels"], all_labels[indices]):
            raise ValueError("DenseNet labels and training labels differ")
    y = all_labels[indices]
    wrn = np.asarray(np.load(args.input_dir / "multibackbone_wrn_rot5_oof.npy",
                             mmap_mode="r")[indices], dtype=np.float32)
    conv = np.asarray(np.load(args.input_dir / "multibackbone_convnextv2_oof.npy",
                              mmap_mode="r")[indices], dtype=np.float32)
    transformed = {name: {gamma: normalize_power(p, gamma)
                          for gamma in (.5, .75, 1.0, 1.25)}
                   for name, p in (("conv", conv), ("dense", dense))}
    output = {"count": len(y), "folds": sorted(set(map(int, folds[indices]))),
              "weights": {}}
    for conv_gamma in (.5, .75, 1.0, 1.25):
        for dense_gamma in (.5, .75, 1.0, 1.25):
            for conv_weight in (.1, .2):
                for dense_weight in (.2, .3, .4, .5):
                    scores = ((1-conv_weight-dense_weight)*wrn +
                              conv_weight*transformed["conv"][conv_gamma] +
                              dense_weight*transformed["dense"][dense_gamma])
                    key = (f"conv_g{conv_gamma}_dense_g{dense_gamma}"
                           f"_conv_w{conv_weight}_dense_w{dense_weight}")
                    outcome = evaluate(scores, y)
                    if args.dense_oof is None:
                        outcome["per_fold"] = [evaluate(scores[folds[indices] == f],
                                                        y[folds[indices] == f])
                                               for f in range(3)]
                    output["weights"][key] = outcome
    best_top1 = max(output["weights"], key=lambda k: output["weights"][k]["top1"])
    best_top10 = max(output["weights"], key=lambda k: output["weights"][k]["top10"])
    output["best_top1"] = best_top1
    output["best_top10"] = best_top10
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(best_top1, output["weights"][best_top1], flush=True)
    print(best_top10, output["weights"][best_top10], flush=True)


if __name__ == "__main__":
    main()
