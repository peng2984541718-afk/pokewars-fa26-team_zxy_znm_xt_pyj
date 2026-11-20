"""Pilot rank-based three-backbone fusion on a held-out fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ranks(probabilities: np.ndarray) -> np.ndarray:
    order = np.argsort(-probabilities.astype(np.float32), axis=1)
    result = np.empty(order.shape, dtype=np.uint16)
    result[np.arange(len(order))[:, None], order] = np.arange(1, order.shape[1]+1,
                                                            dtype=np.uint16)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    z = np.load(args.dense_oof)
    indices = z["indices"]
    target = z["labels"].astype(np.int64)
    predictions = {}
    for name in ("wrn_rot5", "convnextv2"):
        predictions[name] = np.load(args.input_dir / f"multibackbone_{name}_oof.npy",
                                    mmap_mode="r")[indices]
    predictions["densenet169"] = z["probabilities"]
    ordered = {name: ranks(p) for name, p in predictions.items()}
    output = {}
    for offset in (1, 5, 20, 50):
        reverse = {name: 1/(ordered[name].astype(np.float32)+offset)
                   for name in ordered}
        for conv_weight in (.1, .2, .3):
            for dense_weight in (.2, .3, .4, .5):
                s = ((1-conv_weight-dense_weight)*reverse["wrn_rot5"] +
                     conv_weight*reverse["convnextv2"] +
                     dense_weight*reverse["densenet169"])
                top10 = np.argpartition(s, -10, axis=1)[:, -10:]
                key = f"offset_{offset}_conv_{conv_weight}_dense_{dense_weight}"
                output[key] = {"top1": float(np.mean(s.argmax(1) == target)),
                               "top10": float(np.mean((top10 == target[:, None]).any(1)))}
    best1 = max(output, key=lambda k: output[k]["top1"])
    best10 = max(output, key=lambda k: output[k]["top10"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(best1, output[best1], flush=True)
    print(best10, output[best10], flush=True)


if __name__ == "__main__":
    main()
