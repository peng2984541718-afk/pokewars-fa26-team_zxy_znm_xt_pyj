"""Summarize fixed four-backbone OOF probability blends after WRN-40 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from image_experiment import NUM_CLASSES, SEED, stratified_folds


def measure(scores: torch.Tensor, labels: torch.Tensor,
            folds: torch.Tensor) -> dict:
    top = scores.topk(10, dim=1).indices
    result = {"overall": {}, "folds": []}
    for fold in range(3):
        mask = folds == fold
        part = {"fold": fold, "count": int(mask.sum()),
                "top1_correct": int((top[mask, 0] == labels[mask]).sum()),
                "top10_correct": int((top[mask] == labels[mask][:, None]).any(1).sum())}
        result["folds"].append(part)
    result["overall"] = {key: sum(part[key] for part in result["folds"])
                         for key in ("count", "top1_correct", "top10_correct")}
    return result


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/large/four_backbone_oof_eval.json"))
    args = parser.parse_args()
    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    folds = stratified_folds(labels, 3, SEED)
    y = torch.as_tensor(labels, device="cuda")
    f = torch.as_tensor(folds, device="cuda")
    matrices = {}
    for name in ("wrn_rot5", "convnextv2", "densenet169", "wrn40_10"):
        array = np.load(args.input_dir / f"multibackbone_{name}_oof.npy", mmap_mode="r")
        if array.shape != (len(labels), NUM_CLASSES):
            raise ValueError(f"{name} OOF shape mismatch")
        matrices[name] = torch.as_tensor(np.asarray(array).copy(), device="cuda").float()
    base = (.5*matrices["wrn_rot5"] + .2*matrices["convnextv2"] +
            .3*matrices["densenet169"])
    output = {"fixed_base": "0.5 WRN-28 + 0.2 ConvNeXt V2 + 0.3 DenseNet-169",
              "wrn40_alone": measure(matrices["wrn40_10"], y, f),
              "blends": {}}
    for weight in (0, .1, .2, .3, .4, .5):
        output["blends"][str(weight)] = measure(
            (1-weight)*base + weight*matrices["wrn40_10"], y, f)
    wrn40_hit = (matrices["wrn40_10"].topk(10, dim=1).indices == y[:, None]).any(1)
    base_hit = (base.topk(10, dim=1).indices == y[:, None]).any(1)
    output["base_wrn40_top10_union_oracle"] = int((base_hit | wrn40_hit).sum())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for key, item in output["blends"].items():
        total = item["overall"]
        print(key, total["top1_correct"]/total["count"],
              total["top10_correct"]/total["count"], flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
