"""Weighted average of full OOF probability matrices, plus a compact top-k file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("labeled_ids.npy"))
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    if len(args.inputs) != len(args.weights):
        parser.error("--weights needs one value per input")
    weights = np.array(args.weights) / sum(args.weights)
    arrays = [np.load(path, mmap_mode="r") for path in args.inputs]
    labels = np.load(args.labels).astype(np.int64)
    combined = np.empty(arrays[0].shape, dtype=np.float16)
    candidates = np.empty((len(labels), args.top_k), dtype=np.uint16)
    probabilities = np.empty((len(labels), args.top_k), dtype=np.float16)
    for start in range(0, len(labels), 20000):
        end = min(start + 20000, len(labels))
        block = sum(w * np.asarray(a[start:end], dtype=np.float32)
                    for w, a in zip(weights, arrays))
        combined[start:end] = block
        top = np.argsort(-block, axis=1)[:, :args.top_k]
        candidates[start:end] = top + 1
        probabilities[start:end] = np.take_along_axis(block, top, axis=1)
    np.save(args.output, combined)
    np.savez_compressed(args.compact, candidates=candidates, probabilities=probabilities)
    hits = candidates == (labels + 1)[:, None]
    report = {"inputs": [str(p) for p in args.inputs], "weights": weights.tolist(),
              "top1": float(hits[:, 0].mean()), "top5": float(hits[:, :5].any(1).mean()),
              f"top{args.top_k}": float(hits.any(1).mean())}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
