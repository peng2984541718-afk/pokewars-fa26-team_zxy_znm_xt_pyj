"""Pseudo-label unlabeled-battle images for one fold and save a training bank.

The priors must come only from checkpoints whose held-out fold is the target
fold, so the bank carries no information from that fold's validation images.
The bank format matches train_large_image_external.py --external-bank.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from joint_identity_inference import belief_propagation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--table", type=Path, default=Path("outputs/large/pair_mlp_table.npy"))
    parser.add_argument("--images", type=Path, default=Path("data/unlabeled_32x32.npy"))
    parser.add_argument("--unlabeled-paths", type=Path, default=Path("data/unlabeled_paths.txt"))
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv.gz"))
    parser.add_argument("--min-confidence", type=float, default=.9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = lambda v, dtype=None: torch.as_tensor(v, device=device, dtype=dtype)

    paths = args.unlabeled_paths.read_text().split()
    frame = pd.read_csv(args.priors)
    if list(frame.image_path) != paths:
        raise ValueError("Prior file is not in unlabeled path order")
    candidates = np.array([json.loads(v) for v in frame.topk_pokemon_ids], dtype=np.int64)
    prior = np.array([json.loads(v) for v in frame.topk_probabilities], dtype=np.float32)
    prior /= prior.sum(1, keepdims=True)
    rows = pd.read_csv(args.train_csv)
    unknown = rows.pokemon_id1.isna() & rows.pokemon_id2.isna()
    index = pd.Index(paths)
    eu = index.get_indexer(rows.noisy_image1[unknown])
    ev = index.get_indexer(rows.noisy_image2[unknown])
    y = rows.result[unknown].to_numpy(np.float32)
    posterior = belief_propagation(t(candidates), t(prior), t(eu), t(ev), t(y),
                                   t(np.load(args.table), torch.float32), .5, .05, 1.).cpu().numpy()
    best = posterior.argmax(1)
    label = candidates[np.arange(len(paths)), best]
    confidence = posterior[np.arange(len(paths)), best]
    keep = confidence >= args.min_confidence
    images = np.load(args.images)
    if len(images) != len(paths):
        raise ValueError("Image cache and path list differ in length")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, images=images[keep], labels=(label[keep] - 1).astype(np.int64),
             confidence=confidence[keep])
    report = {"images": len(paths), "kept": int(keep.sum()),
              "min_confidence": args.min_confidence,
              "changed_from_image_top1": float(np.mean(label != candidates[:, 0])),
              "priors": str(args.priors), "table": str(args.table)}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
