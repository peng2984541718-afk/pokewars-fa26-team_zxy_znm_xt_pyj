"""Compare fixed probability blends of WRN, ConvNeXt V2 and DenseNet OOF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from image_experiment import NUM_CLASSES, SEED, stratified_folds


def counts(scores: torch.Tensor, target: torch.Tensor) -> dict:
    top = scores.topk(10, dim=1).indices
    return {"count": len(target), "top1": int((top[:, 0] == target).sum()),
            "top10": int((top == target[:, None]).any(1).sum())}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/large/three_backbone_oof_eval.json"))
    args = parser.parse_args()
    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    folds = stratified_folds(labels, 3, SEED)
    matrices = {}
    for name in ("wrn_rot5", "convnextv2", "densenet169"):
        a = np.load(args.input_dir / f"multibackbone_{name}_oof.npy", mmap_mode="r")
        if a.shape != (len(labels), NUM_CLASSES):
            raise ValueError(f"{name} shape mismatch")
        matrices[name] = torch.as_tensor(np.asarray(a).copy(), device="cuda").float()
    y = torch.as_tensor(labels, device="cuda")
    fold_masks = [torch.as_tensor(folds == fold, device="cuda") for fold in range(3)]
    results = {"formula": "(1-conv_weight-dense_weight)*WRN + conv_weight*ConvNeXt + dense_weight*DenseNet",
               "weights": {}}
    for conv_weight in (0, .1, .2, .3):
        for dense_weight in (0, .1, .2, .25, .3, .35, .4, .45, .5):
            if conv_weight + dense_weight >= 1:
                continue
            p = ((1-conv_weight-dense_weight)*matrices["wrn_rot5"] +
                 conv_weight*matrices["convnextv2"] +
                 dense_weight*matrices["densenet169"])
            parts = [counts(p[mask], y[mask]) for mask in fold_masks]
            key = f"conv_{conv_weight}_dense_{dense_weight}"
            results["weights"][key] = {"per_fold": parts,
                                       "overall": {field: sum(row[field] for row in parts)
                                                   for field in parts[0]}}
    # Fold 0 is the tuning fold; report held-out 1+2 separately for a fixed choice.
    best_top1_key = max(results["weights"], key=lambda k:
                        results["weights"][k]["per_fold"][0]["top1"])
    best_top10_key = max(results["weights"], key=lambda k:
                         results["weights"][k]["per_fold"][0]["top10"])
    results["fold0_selected"] = {"top1": best_top1_key, "top10": best_top10_key}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for key in (best_top1_key, best_top10_key, "conv_0.2_dense_0.3"):
        print(key, results["weights"][key], flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
