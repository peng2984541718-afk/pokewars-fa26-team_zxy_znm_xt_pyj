"""Blend cross-fitted candidate rankers using fold-0 selection only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ranks(score: np.ndarray) -> np.ndarray:
    order = np.argsort(-score, axis=1)
    result = np.empty(order.shape, dtype=np.float32)
    result[np.arange(len(score))[:, None], order] = np.arange(1, score.shape[1]+1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--lambdarank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prob-dir", type=Path, default=None,
                        help="Include fixed three-architecture probability ranking")
    args = parser.parse_args()
    a = np.load(args.binary)
    b = np.load(args.lambdarank)
    for name in ("ids", "labels", "folds", "valid"):
        if not np.array_equal(a[name], b[name]):
            raise ValueError(f"OOF rankers differ in {name}")
    ids = a["ids"]
    y = a["labels"]
    folds = a["folds"]
    valid = a["valid"]
    ra = ranks(a["scores"])
    rb = ranks(b["scores"])
    rf = None
    if args.prob_dir is not None:
        fixed = np.zeros(ids.shape, dtype=np.float32)
        for name, weight in (("wrn_rot5", .5), ("convnextv2", .2),
                             ("densenet169", .3)):
            p = np.load(args.prob_dir / f"multibackbone_{name}_oof.npy",
                        mmap_mode="r")
            fixed += weight*np.take_along_axis(p, ids, axis=1).astype(np.float32)
        fixed[~valid] = -np.inf
        rf = ranks(fixed)
    output = {"form": "weighted reciprocal ranks of cross-fitted models and optional fixed blend",
              "parameters": {}}
    for offset in (0, 5, 20):
        sa = 1/(ra+offset)
        sb = 1/(rb+offset)
        fixed_weights = (0, .1, .2, .3, .4) if rf is not None else (0,)
        sf = 1/(rf+offset) if rf is not None else None
        for weight in (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1):
            for fixed_weight in fixed_weights:
                if weight+fixed_weight > 1:
                    continue
                combined = (1-weight-fixed_weight)*sa + weight*sb
                if sf is not None:
                    combined += fixed_weight*sf
                combined[~valid] = -1
                chosen = np.argsort(-combined, axis=1)[:, :10]
                top10 = np.take_along_axis(ids, chosen, axis=1)
                parts = []
                for fold in range(3):
                    sel = folds == fold
                    true = y[sel]
                    parts.append({"fold": fold, "count": len(true),
                                  "top1_correct": int((top10[sel, 0] == true).sum()),
                                  "top10_correct": int((top10[sel] == true[:, None]).any(1).sum())})
                key = (f"offset_{offset}_lambdarank_weight_{weight}"
                       f"_fixed_weight_{fixed_weight}")
                output["parameters"][key] = {"folds": parts,
                                             "overall": {field: sum(p[field] for p in parts)
                                                         for field in ("count", "top1_correct",
                                                                       "top10_correct")}}
    selected = max(output["parameters"], key=lambda k:
                   output["parameters"][k]["folds"][0]["top10_correct"])
    output["fold0_selected_top10"] = selected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("Selected", selected, output["parameters"][selected], flush=True)


if __name__ == "__main__":
    main()
