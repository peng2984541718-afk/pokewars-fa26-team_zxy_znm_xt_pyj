"""Measure whether ConvNeXt V2 adds complementary OOF classes to WRN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from image_experiment import NUM_CLASSES, SEED, stratified_folds


def summary(scores: torch.Tensor, target: torch.Tensor) -> dict:
    top10 = scores.topk(10, dim=1).indices
    return {"count": len(target),
            "top1_correct": int((top10[:, 0] == target).sum()),
            "top5_correct": int((top10[:, :5] == target[:, None]).any(1).sum()),
            "top10_correct": int((top10 == target[:, None]).any(1).sum())}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/large/multibackbone_oof_eval.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required")
    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    folds = stratified_folds(labels, 3, SEED)
    arrays = {}
    for name in ("wrn_rot5", "convnextv2"):
        array = np.load(args.input_dir / f"multibackbone_{name}_oof.npy",
                        mmap_mode="r")
        if array.shape != (len(labels), NUM_CLASSES):
            raise ValueError(f"{name} probability shape mismatch")
        arrays[name] = torch.as_tensor(np.asarray(array).copy(), device="cuda")
    target = torch.as_tensor(labels, device="cuda")
    weights = (0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1)
    output = {"weights": {}, "folds": 3,
              "fusion": "(1-weight)*WRN_rot5_probability + weight*ConvNeXtV2_probability"}
    wrn_top10 = arrays["wrn_rot5"].topk(10, dim=1).indices
    conv_top10 = arrays["convnextv2"].topk(10, dim=1).indices
    union = ((wrn_top10 == target[:, None]).any(1) |
             (conv_top10 == target[:, None]).any(1))
    output["top10_union"] = {
        "overall_correct": int(union.sum()),
        "per_fold_correct": [int(union[torch.as_tensor(folds == f, device="cuda")].sum())
                             for f in range(3)]}
    for weight in weights:
        score = ((1 - weight) * arrays["wrn_rot5"].float() +
                 weight * arrays["convnextv2"].float())
        parts = [summary(score[torch.as_tensor(folds == f, device="cuda")],
                         target[torch.as_tensor(folds == f, device="cuda")])
                 for f in range(3)]
        total = {key: sum(x[key] for x in parts) for key in parts[0]}
        output["weights"][str(weight)] = {"per_fold": parts, "overall": total}
        print(weight, "top1", total["top1_correct"] / len(labels),
              "top10", total["top10_correct"] / len(labels), flush=True)
        del score
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
