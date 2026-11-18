"""Test fixed and confidence-gated ConvNeXt slots in a WRN top-10 list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from image_experiment import SEED, stratified_folds


def top_ids(probabilities: np.ndarray, k: int = 10) -> np.ndarray:
    chosen = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    order = np.argsort(-np.take_along_axis(probabilities, chosen, axis=1), axis=1)
    return np.take_along_axis(chosen, order, axis=1)


def reserved(wrn: np.ndarray, conv: np.ndarray, count: int) -> np.ndarray:
    if count == 0:
        return wrn
    first = conv[:, :count]
    available = ~(wrn[:, :, None] == first[:, None, :]).any(2)
    order = np.argsort(~available, axis=1, kind="stable")
    return np.concatenate((first, np.take_along_axis(wrn, order[:, :10-count], axis=1)),
                          axis=1)


def metrics(candidates: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> dict:
    hit = (candidates == labels[:, None]).any(1)
    return {"overall": float(hit.mean()),
            "folds": [float(hit[folds == f].mean()) for f in range(3)],
            "correct": int(hit.sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/large/candidate_union_eval.json"))
    args = parser.parse_args()
    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    folds = stratified_folds(labels, 3, SEED)
    wrn_probs = np.load(args.input_dir / "multibackbone_wrn_rot5_oof.npy",
                        mmap_mode="r")
    conv_probs = np.load(args.input_dir / "multibackbone_convnextv2_oof.npy",
                         mmap_mode="r")
    wrn = top_ids(wrn_probs)
    conv = top_ids(conv_probs)
    output = {"fixed_slots": {}, "confidence_gates": {}}
    for count in range(6):
        candidates = reserved(wrn, conv, count)
        output["fixed_slots"][str(count)] = metrics(candidates, labels, folds)
        print("slots", count, output["fixed_slots"][str(count)], flush=True)

    # Only switch the candidate list on lower-confidence WRN predictions.
    wrn_conf = np.max(wrn_probs, axis=1)
    conv_first = conv[:, 0]
    conv_conf = np.take_along_axis(conv_probs, conv_first[:, None], axis=1)[:, 0]
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        for count in (1, 2, 3):
            candidate = reserved(wrn, conv, count)
            choice = wrn.copy()
            use = wrn_conf < threshold
            choice[use] = candidate[use]
            key = f"wrn_conf_lt_{threshold}_conv_slots_{count}"
            output["confidence_gates"][key] = {
                **metrics(choice, labels, folds), "switched": int(use.sum())}
            print(key, output["confidence_gates"][key], flush=True)
    output["union_oracle"] = metrics(np.concatenate((wrn, conv), axis=1),
                                     labels, folds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
