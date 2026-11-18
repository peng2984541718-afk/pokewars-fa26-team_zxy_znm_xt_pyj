"""Compute fold-ensemble WRN gallery features for test top-20 candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from feature_rerank import candidate_scores, extract
from image_experiment import SEED, stratified_folds
from train_large_image import make_model


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-batch-size", type=int, default=512)
    parser.add_argument("--neighbor-batch-size", type=int, default=64)
    parser.add_argument("--prefixes", nargs="+", default=["wrn28_10", "wrn28_10_mild"],
                        help="Must match the prefixes used by feature_rerank.py")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    paths = (args.data_dir / "test_paths.txt").read_text().splitlines()
    test_cache = np.load(args.data_dir / "test_32x32.npy", mmap_mode="r")
    train_cache = np.load(args.data_dir / "labeled_32x32.npy", mmap_mode="r")
    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    if len(paths) != len(test_cache) or len(labels) != len(train_cache):
        raise ValueError("Image cache/label count mismatch")
    ids = np.empty((len(paths), 20), dtype=np.int64)
    wrn_probs = np.empty_like(ids, dtype=np.float32)
    with args.candidates.open(newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            if i >= len(paths) or row["image_path"] != paths[i]:
                raise ValueError(f"Test path mismatch at row {i}")
            ids[i] = json.loads(row["topk_pokemon_ids"])
            wrn_probs[i] = json.loads(row["topk_probabilities"])
    if i + 1 != len(paths) or len(paths) != 160000 or np.any(ids < 1) or np.any(ids > 1025):
        raise ValueError("Incomplete or invalid test candidate file")
    folds = stratified_folds(labels, 3, SEED)
    train_images = torch.as_tensor(np.asarray(train_cache).copy(), device="cuda")
    test_images = torch.as_tensor(np.asarray(test_cache).copy(), device="cuda")
    query_indices = np.arange(len(paths))
    scores = {name: np.zeros(ids.shape, dtype=np.float32)
              for name in ("knn1", "knn5", "prototype")}
    for fold in range(3):
        gallery_indices = np.flatnonzero(folds != fold)
        for prefix in args.prefixes:
            checkpoint = args.checkpoint_dir / f"{prefix}_f{fold}_stage2_best.pt"
            model = make_model("wrn28_10", False).cuda().eval()
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            gallery_features = extract(model, train_images, gallery_indices,
                                       args.image_batch_size)
            query_features = extract(model, test_images, query_indices,
                                     args.image_batch_size)
            one = candidate_scores(gallery_features, query_features,
                                   labels[gallery_indices], ids-1,
                                   args.neighbor_batch_size)
            for name in scores:
                scores[name] += one[name] / (3 * len(args.prefixes))
            del model, gallery_features, query_features, one
            torch.cuda.empty_cache()
            print(f"Scored {checkpoint.name}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, candidates=ids.astype(np.uint16),
                        cnn_probs=wrn_probs.astype(np.float32), **scores)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
