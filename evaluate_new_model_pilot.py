"""Measure the held-out value of one new backbone before training other folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def metrics(prob: torch.Tensor, target: torch.Tensor) -> dict:
    top10 = prob.topk(10, dim=1).indices
    return {"top1": float((top10[:, 0] == target).float().mean()),
            "top10": float((top10 == target[:, None]).any(1).float().mean())}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-oof", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    z = np.load(args.new_oof)
    indices = z["indices"]
    target = torch.as_tensor(z["labels"].astype(np.int64), device="cuda")
    new = torch.as_tensor(z["probabilities"], device="cuda").float()
    old = {}
    for name in ("wrn_rot5", "convnextv2"):
        old[name] = torch.as_tensor(np.asarray(np.load(
            args.input_dir / f"multibackbone_{name}_oof.npy", mmap_mode="r")
            [indices]).copy(), device="cuda").float()
    output = {"count": len(indices), "new_model": metrics(new, target),
              "wrn": metrics(old["wrn_rot5"], target),
              "convnextv2": metrics(old["convnextv2"], target),
              "mixtures": {}}
    for weight in (.05, .1, .15, .2, .25, .3, .4):
        for base_name, base in (("wrn", old["wrn_rot5"]),
                                ("wrn_conv20", .8*old["wrn_rot5"]+.2*old["convnextv2"])):
            key = f"{base_name}_plus_new_{weight}"
            output["mixtures"][key] = metrics((1-weight)*base+weight*new, target)
    wrn_top10 = old["wrn_rot5"].topk(10, dim=1).indices
    new_top10 = new.topk(10, dim=1).indices
    output["top10_union_oracle"] = float((((wrn_top10 == target[:, None]).any(1)) |
                                            ((new_top10 == target[:, None]).any(1))).float().mean())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
