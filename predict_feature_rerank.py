"""Rerank full-test WRN top-10 identities using training-image feature neighbors."""

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


def load_candidates(path: Path, expected_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ids = np.empty((len(expected_paths), 10), dtype=np.int64)
    probs = np.empty((len(expected_paths), 10), dtype=np.float32)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            if i >= len(expected_paths) or row["image_path"] != expected_paths[i]:
                raise ValueError(f"Test path mismatch at row {i}")
            ids[i] = json.loads(row["topk_pokemon_ids"])
            probs[i] = json.loads(row["topk_probabilities"])
        if i + 1 != len(expected_paths):
            raise ValueError("Test top-10 file has the wrong number of rows")
    if np.any(ids < 1) or np.any(ids > 1025) or np.any(probs <= 0):
        raise ValueError("Invalid candidate IDs or probabilities")
    return ids, probs


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--cnn-top10", type=Path, default=Path(
        "outputs/large/test_wrn28_10_six_model_rot5_160000_top10.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/large/test_wrn28_10_rot5_knn1_reranked_160000_top10.csv"))
    parser.add_argument("--beta", type=float, default=32.0)
    parser.add_argument("--image-batch-size", type=int, default=512)
    parser.add_argument("--neighbor-batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required")

    paths = (args.data_dir / "test_paths.txt").read_text().splitlines()
    test_cache = np.load(args.data_dir / "test_32x32.npy", mmap_mode="r")
    train_cache = np.load(args.data_dir / "labeled_32x32.npy", mmap_mode="r")
    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    if (paths != sorted(set(paths)) or len(paths) != len(test_cache) or
            len(labels) != len(train_cache) or test_cache.shape[1:] != (32, 32)):
        raise ValueError("Image caches and path lists are inconsistent")
    candidate_ids, candidate_probs = load_candidates(args.cnn_top10, paths)
    folds = stratified_folds(labels, 3, SEED)
    train_images = torch.as_tensor(np.asarray(train_cache).copy(), device="cuda")
    test_images = torch.as_tensor(np.asarray(test_cache).copy(), device="cuda")
    test_indices = np.arange(len(paths))
    similarities = np.zeros(candidate_probs.shape, dtype=np.float32)
    checkpoints = []
    torch.backends.cudnn.benchmark = True
    for fold in range(3):
        gallery_indices = np.flatnonzero(folds != fold)
        for prefix in ("wrn28_10", "wrn28_10_mild"):
            checkpoint = args.checkpoint_dir / f"{prefix}_f{fold}_stage2_best.pt"
            model = make_model("wrn28_10", pretrained=False).cuda().eval()
            model.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                             weights_only=True))
            gallery_features = extract(model, train_images, gallery_indices,
                                       args.image_batch_size)
            query_features = extract(model, test_images, test_indices,
                                     args.image_batch_size)
            scores = candidate_scores(gallery_features, query_features,
                                      labels[gallery_indices], candidate_ids - 1,
                                      args.neighbor_batch_size)
            similarities += scores["knn1"] / 6
            checkpoints.append(str(checkpoint))
            del model, gallery_features, query_features, scores
            torch.cuda.empty_cache()
            print(f"Scored {checkpoint.name}", flush=True)

    fused = np.log(np.maximum(candidate_probs, 1e-12)) + args.beta * similarities
    order = np.argsort(-fused, axis=1)
    sorted_ids = np.take_along_axis(candidate_ids, order, axis=1)
    sorted_cnn_probs = np.take_along_axis(candidate_probs, order, axis=1)
    sorted_scores = np.take_along_axis(fused, order, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "reranked_top1_pokemon_id",
                         "original_cnn_top1_pokemon_id", "original_cnn_top1_probability",
                         "reranked_top1_cnn_probability", "reranked_top1_original_rank",
                         "reranked_top10_pokemon_ids", "reranked_top10_cnn_probabilities",
                         "reranked_top10_scores"])
        for i, path in enumerate(paths):
            writer.writerow([path, int(sorted_ids[i, 0]), int(candidate_ids[i, 0]),
                             float(candidate_probs[i, 0]), float(sorted_cnn_probs[i, 0]),
                             int(order[i, 0]) + 1,
                             json.dumps(sorted_ids[i].tolist()),
                             json.dumps(sorted_cnn_probs[i].astype(float).tolist()),
                             json.dumps(sorted_scores[i].astype(float).tolist())])
    metadata = {
        "images": len(paths), "candidate_source": str(args.cnn_top10),
        "checkpoints": checkpoints, "beta": args.beta,
        "rerank_formula": "log(cnn_probability) + beta * mean_six_model_knn1_cosine",
        "gallery": "Each checkpoint uses images outside its held-out fold",
        "calibrated": False,
        "cnn_probabilities": "Original CNN scores attached to the same IDs; reranked scores are not probabilities",
        "ordering": "sorted unique test paths", "kaggle_submission": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(paths):,} reranked rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
