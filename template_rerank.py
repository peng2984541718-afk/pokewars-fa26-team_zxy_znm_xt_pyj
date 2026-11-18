"""Fold-safe class-template matching for CNN top-10 image candidates.

Each prototype uses labeled images from the two training folds only. This
script exports template similarities so fusion weights can be selected on a
different fold from the one used for final evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_wrn_tta import transform
from image_experiment import NUM_CLASSES, SEED, stratified_folds
from train_large_image import preprocess


SHIFTS = ("left", "right", "up", "down")
AFFINE = ("rotate_minus5", "rotate_plus5", "scale_095", "scale_105")


@torch.inference_mode()
def build_templates(images: torch.Tensor, labels: np.ndarray,
                    train_indices: np.ndarray, batch_size: int,
                    align_iterations: int, aggregation: str) -> torch.Tensor:
    sums = torch.zeros((NUM_CLASSES, 1024), device="cuda", dtype=torch.float32)
    counts = torch.zeros(NUM_CLASSES, device="cuda", dtype=torch.float32)
    for start in range(0, len(train_indices), batch_size):
        indices = train_indices[start:start + batch_size]
        index = torch.as_tensor(indices, device="cuda")
        class_index = torch.as_tensor(labels[indices], device="cuda")
        x = preprocess(images[index]).flatten(1)
        sums.index_add_(0, class_index, x)
        counts.index_add_(0, class_index, torch.ones(len(indices), device="cuda"))
    if counts.min().item() < 1:
        raise ValueError("Missing class in training folds")
    templates = sums / counts[:, None]
    print(f"class template training counts: {int(counts.min().item())}–{int(counts.max().item())}",
          flush=True)
    templates = F.normalize(templates, dim=1)

    for iteration in range(max(align_iterations, int(aggregation == "median"))):
        aligned = torch.empty((len(train_indices), 1024), device="cuda")
        for start in range(0, len(train_indices), batch_size):
            end = min(start + batch_size, len(train_indices))
            indices = train_indices[start:end]
            index = torch.as_tensor(indices, device="cuda")
            class_index = torch.as_tensor(labels[indices], device="cuda")
            x = preprocess(images[index])
            class_templates = templates[class_index]
            chosen = x
            best_score = (F.normalize(x.flatten(1), dim=1) * class_templates).sum(-1)
            if iteration < align_iterations:
                for view in (*SHIFTS, *AFFINE):
                    candidate = transform(x, view)
                    score = (F.normalize(candidate.flatten(1), dim=1) *
                             class_templates).sum(-1)
                    improve = score > best_score
                    chosen = torch.where(improve[:, None, None, None], candidate, chosen)
                    best_score = torch.maximum(best_score, score)
            aligned[start:end] = chosen.flatten(1)
        if aggregation == "mean":
            sums.zero_()
            for start in range(0, len(train_indices), batch_size):
                end = min(start + batch_size, len(train_indices))
                class_index = torch.as_tensor(labels[train_indices[start:end]],
                                              device="cuda")
                sums.index_add_(0, class_index, aligned[start:end])
            templates = sums / counts[:, None]
        else:
            if counts.min().item() != counts.max().item():
                raise ValueError("Median aggregation requires equal class counts")
            order = torch.as_tensor(np.argsort(labels[train_indices], kind="stable"),
                                    device="cuda")
            per_class = int(counts[0].item())
            templates = aligned[order].reshape(NUM_CLASSES, per_class, 1024).median(1).values
        templates = F.normalize(templates, dim=1)
        print(f"aligned template pass {iteration + 1}/{max(align_iterations, int(aggregation == 'median'))}",
              flush=True)
    return templates


def similarity(x: torch.Tensor, candidate_templates: torch.Tensor) -> torch.Tensor:
    query = F.normalize(x.flatten(1), dim=1)
    return (candidate_templates * query[:, None, :]).sum(-1)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--diagnostics", type=Path,
                        default=Path("outputs/large/wrn_pair_oof_diagnostics.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--align-iterations", type=int, default=0)
    parser.add_argument("--aggregation", choices=("mean", "median"), default="mean")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required")

    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    image_cache = np.load(args.data_dir / "labeled_32x32.npy", mmap_mode="r")
    diag = np.load(args.diagnostics)
    if len(labels) != len(image_cache) or len(labels) != len(diag["labels"]):
        raise ValueError("Image, label, and OOF candidate caches have different sizes")
    if not np.array_equal(labels + 1, diag["labels"]):
        raise ValueError("OOF candidates are not aligned to labels")
    folds = stratified_folds(labels, 3, SEED)
    if not np.array_equal(folds, diag["folds"]):
        raise ValueError("Fold assignment differs from OOF candidates")
    train_indices = np.flatnonzero(folds != args.fold)
    valid_indices = np.flatnonzero(folds == args.fold)
    images = torch.as_tensor(np.asarray(image_cache).copy(), device="cuda")
    templates = build_templates(images, labels, train_indices, args.batch_size,
                                args.align_iterations, args.aggregation)

    candidates = diag["top10_ids"][valid_indices].astype(np.int64) - 1
    probabilities = diag["top10_probs"][valid_indices].astype(np.float32)
    true_labels = labels[valid_indices]
    n = len(valid_indices)
    raw_scores = np.empty((n, 10), dtype=np.float32)
    shift_scores = np.empty((n, 10), dtype=np.float32)
    full_scores = np.empty((n, 10), dtype=np.float32)

    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        indices = torch.as_tensor(valid_indices[start:end], device="cuda")
        candidate_index = torch.as_tensor(candidates[start:end], device="cuda")
        reference = templates[candidate_index]
        x = preprocess(images[indices])
        baseline = similarity(x, reference)
        shifted = baseline
        for view in SHIFTS:
            shifted = torch.maximum(shifted, similarity(transform(x, view), reference))
        affine = shifted
        for view in AFFINE:
            affine = torch.maximum(affine, similarity(transform(x, view), reference))
        raw_scores[start:end] = baseline.cpu().numpy()
        shift_scores[start:end] = shifted.cpu().numpy()
        full_scores[start:end] = affine.cpu().numpy()
        if end % 10000 == 0 or end == n:
            print(f"fold {args.fold}: {end:,}/{n:,}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = (f"_{args.aggregation}_aligned{args.align_iterations}"
              if args.align_iterations or args.aggregation != "mean" else "")
    output = args.output_dir / f"template_rerank_f{args.fold}{suffix}.npz"
    np.savez_compressed(output, valid_indices=valid_indices, labels=true_labels + 1,
                        candidates=candidates + 1, cnn_probs=probabilities,
                        correlation_raw=raw_scores,
                        correlation_shift=shift_scores,
                        correlation_full=full_scores)
    hit = (candidates == true_labels[:, None]).any(1)
    print(f"fold {args.fold} CNN top1={np.mean(candidates[:, 0] == true_labels):.5f} "
          f"top10_oracle={np.mean(hit):.5f}", flush=True)
    for key, scores in (("raw", raw_scores), ("shift", shift_scores),
                        ("full", full_scores)):
        chosen = candidates[np.arange(n), scores.argmax(1)]
        print(f"fold {args.fold} template_{key}_only={np.mean(chosen == true_labels):.5f}",
              flush=True)
        for beta in (2, 4, 8, 16, 32, 64):
            fused = np.log(probabilities.clip(1e-12)) + beta * scores
            predicted = candidates[np.arange(n), fused.argmax(1)]
            print(f"fold {args.fold} {key} beta={beta}: "
                  f"top1={np.mean(predicted == true_labels):.5f}", flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
