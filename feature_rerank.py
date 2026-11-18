"""Fold-safe deep-feature prototype and nearest-neighbor top-10 reranking."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from image_experiment import NUM_CLASSES, SEED, stratified_folds
from train_large_image import make_model, preprocess


@torch.inference_mode()
def extract(model: torch.nn.Module, images: torch.Tensor,
            indices: np.ndarray, batch_size: int) -> torch.Tensor:
    features = torch.empty((len(indices), 640), device="cuda", dtype=torch.float16)
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        index = torch.as_tensor(indices[start:end], device="cuda")
        x = preprocess(images[index])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.stages(model.stem(x))
            hidden = F.relu(model.bn(hidden)).mean(dim=(-2, -1))
        features[start:end] = F.normalize(hidden.float(), dim=1).half()
    return features


@torch.inference_mode()
def candidate_scores(train_features: torch.Tensor, valid_features: torch.Tensor,
                     train_labels: np.ndarray, candidates: np.ndarray,
                     batch_size: int) -> dict[str, np.ndarray]:
    counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    if counts.min() != counts.max():
        raise ValueError("Expected equal training examples per class")
    per_class = int(counts[0])
    order = torch.as_tensor(np.argsort(train_labels, kind="stable"), device="cuda")
    gallery = train_features[order].reshape(NUM_CLASSES, per_class, 640)
    sums = gallery.float().sum(dim=1)
    prototypes = F.normalize(sums, dim=1).half()
    n = len(valid_features)
    result = {name: np.empty((n, candidates.shape[1]), dtype=np.float32)
              for name in ("prototype", "knn1", "knn5")}
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        candidate_index = torch.as_tensor(candidates[start:end], device="cuda")
        query = valid_features[start:end]
        result["prototype"][start:end] = (prototypes[candidate_index].float() *
                                             query.float()[:, None, :]).sum(-1).cpu().numpy()
        neighbors = gallery[candidate_index]
        similarity = (neighbors.float() * query.float()[:, None, None, :]).sum(-1)
        result["knn1"][start:end] = similarity.max(-1).values.cpu().numpy()
        result["knn5"][start:end] = similarity.topk(5, dim=-1).values.mean(-1).cpu().numpy()
    return result


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--diagnostics", type=Path,
                        default=Path("outputs/large/wrn_pair_oof_diagnostics.npz"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--run-name", default="feature_rerank")
    parser.add_argument("--full-probs", type=Path, default=None,
                        help="Optional full-class OOF matrix for a wider candidate pool")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--prefixes", nargs="+", default=["wrn28_10", "wrn28_10_mild"],
                        help="WRN checkpoint prefixes whose gallery scores are averaged")
    parser.add_argument("--image-batch-size", type=int, default=512)
    parser.add_argument("--neighbor-batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required")

    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    image_cache = np.load(args.data_dir / "labeled_32x32.npy", mmap_mode="r")
    diag = np.load(args.diagnostics)
    if (len(labels) != len(image_cache) or len(labels) != len(diag["labels"]) or
            not np.array_equal(labels + 1, diag["labels"])):
        raise ValueError("Image, label, and OOF candidate caches differ")
    folds = stratified_folds(labels, 3, SEED)
    if not np.array_equal(folds, diag["folds"]):
        raise ValueError("Fold assignment differs from OOF candidates")
    train_indices = np.flatnonzero(folds != args.fold)
    valid_indices = np.flatnonzero(folds == args.fold)
    if args.full_probs is None:
        if args.top_k != 10:
            parser.error("--top-k other than 10 requires --full-probs")
        candidates = diag["top10_ids"][valid_indices].astype(np.int64) - 1
        probabilities = diag["top10_probs"][valid_indices].astype(np.float32)
    else:
        full = np.load(args.full_probs, mmap_mode="r")
        if full.shape != (len(labels), NUM_CLASSES) or not 1 <= args.top_k <= NUM_CLASSES:
            raise ValueError("Full probability shape or top-k is invalid")
        fold_probs = np.asarray(full[valid_indices], dtype=np.float32)
        selected = np.argpartition(fold_probs, -args.top_k, axis=1)[:, -args.top_k:]
        selected_probs = np.take_along_axis(fold_probs, selected, axis=1)
        order = np.argsort(-selected_probs, axis=1)
        candidates = np.take_along_axis(selected, order, axis=1)
        probabilities = np.take_along_axis(selected_probs, order, axis=1)
    true_labels = labels[valid_indices]
    images = torch.as_tensor(np.asarray(image_cache).copy(), device="cuda")
    torch.backends.cudnn.benchmark = True

    scores = {name: np.zeros((len(valid_indices), args.top_k), dtype=np.float32)
              for name in ("prototype", "knn1", "knn5")}
    for prefix in args.prefixes:
        model = make_model("wrn28_10", pretrained=False).cuda().eval()
        checkpoint = args.checkpoint_dir / f"{prefix}_f{args.fold}_stage2_best.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                         weights_only=True))
        train_features = extract(model, images, train_indices, args.image_batch_size)
        valid_features = extract(model, images, valid_indices, args.image_batch_size)
        model_scores = candidate_scores(train_features, valid_features,
                                        labels[train_indices], candidates,
                                        args.neighbor_batch_size)
        for name in scores:
            scores[name] += model_scores[name] / len(args.prefixes)
        del model, train_features, valid_features
        torch.cuda.empty_cache()
        print(f"fold {args.fold}: scored {prefix}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.run_name}_f{args.fold}.npz"
    np.savez_compressed(output, valid_indices=valid_indices, labels=true_labels + 1,
                        candidates=candidates + 1, cnn_probs=probabilities,
                        **scores)
    n = len(valid_indices)
    print(f"fold {args.fold} CNN top1={np.mean(candidates[:, 0] == true_labels):.5f}",
          flush=True)
    for name, similarity in scores.items():
        predicted = candidates[np.arange(n), similarity.argmax(1)]
        print(f"fold {args.fold} {name}_only={np.mean(predicted == true_labels):.5f}",
              flush=True)
        for beta in (2, 4, 8, 16, 32, 64):
            fused = np.log(probabilities.clip(1e-12)) + beta * similarity
            predicted = candidates[np.arange(n), fused.argmax(1)]
            print(f"fold {args.fold} {name} beta={beta}: "
                  f"top1={np.mean(predicted == true_labels):.5f}", flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
