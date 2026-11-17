"""Matched-fold deterministic test-time augmentation for six WRN checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from image_experiment import SEED, stratified_folds
from train_large_image import make_model, preprocess


VIEWS = {
    "left": (0.0, -1.0, 0.0, 1.0),
    "right": (0.0, 1.0, 0.0, 1.0),
    "up": (0.0, 0.0, -1.0, 1.0),
    "down": (0.0, 0.0, 1.0, 1.0),
    "rotate_minus5": (-5.0, 0.0, 0.0, 1.0),
    "rotate_plus5": (5.0, 0.0, 0.0, 1.0),
    "scale_095": (0.0, 0.0, 0.0, 0.95),
    "scale_105": (0.0, 0.0, 0.0, 1.05),
}
CONFIGS = {
    "base": [],
    "shift_1px": ["left", "right", "up", "down"],
    "rotate_5deg": ["rotate_minus5", "rotate_plus5"],
    "scale_5pct": ["scale_095", "scale_105"],
    "shift_rotate": ["left", "right", "up", "down",
                     "rotate_minus5", "rotate_plus5"],
}


def transform(x: torch.Tensor, view: str) -> torch.Tensor:
    degrees, shift_x, shift_y, scale = VIEWS[view]
    angle = math.radians(degrees)
    theta = torch.tensor([[math.cos(angle) / scale, -math.sin(angle) / scale,
                           2 * shift_x / x.shape[-1]],
                          [math.sin(angle) / scale, math.cos(angle) / scale,
                           2 * shift_y / x.shape[-2]]], device=x.device,
                         dtype=x.dtype).expand(len(x), -1, -1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


@torch.inference_mode()
def predict(models: list[torch.nn.Module], x: torch.Tensor) -> torch.Tensor:
    total = None
    for model in models:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        probs = torch.softmax(logits.float(), dim=1)
        total = probs if total is None else total + probs
    return total / len(models)


def blank_counts() -> dict:
    return {name: {"top1_correct": 0, "top5_correct": 0,
                   "top10_correct": 0, "gained": 0, "lost": 0}
            for name in CONFIGS}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/large/wrn_pair_tta_cv.json"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required")

    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    images = np.load(args.data_dir / "labeled_32x32.npy", mmap_mode="r")
    if len(labels) != len(images) or images.shape[1:] != (32, 32):
        raise ValueError("Image and label caches are not aligned")
    folds = stratified_folds(labels, 3, SEED)
    all_counts = blank_counts()
    fold_reports = []
    torch.backends.cudnn.benchmark = True

    for fold in range(3):
        models = []
        for prefix in ("wrn28_10", "wrn28_10_mild"):
            model = make_model("wrn28_10", pretrained=False).cuda().eval()
            model.load_state_dict(torch.load(
                args.checkpoint_dir / f"{prefix}_f{fold}_stage2_best.pt",
                map_location="cpu", weights_only=True))
            models.append(model)
        indices = np.flatnonzero(folds == fold)
        fold_counts = blank_counts()

        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            raw = torch.as_tensor(np.asarray(images[batch_indices]).copy(), device="cuda")
            x = preprocess(raw)
            target = torch.as_tensor(labels[batch_indices], device="cuda")
            base = predict(models, x)
            view_probs = {name: predict(models, transform(x, name)) for name in VIEWS}
            base_correct = base.topk(1, dim=1).indices[:, 0] == target

            for name, views in CONFIGS.items():
                probs = base if not views else (base + sum(view_probs[v] for v in views)) / (1 + len(views))
                top10 = probs.topk(10, dim=1).indices
                hits = top10.eq(target[:, None])
                counts = fold_counts[name]
                counts["top1_correct"] += int(hits[:, :1].any(1).sum().item())
                counts["top5_correct"] += int(hits[:, :5].any(1).sum().item())
                counts["top10_correct"] += int(hits.any(1).sum().item())
                if name != "base":
                    now_correct = top10[:, 0] == target
                    counts["gained"] += int((now_correct & ~base_correct).sum().item())
                    counts["lost"] += int((~now_correct & base_correct).sum().item())

        for name in CONFIGS:
            for key in all_counts[name]:
                all_counts[name][key] += fold_counts[name][key]
        fold_reports.append({"fold": fold, "images": len(indices),
                             "counts": fold_counts})
        print(f"fold {fold}: {len(indices):,} images", flush=True)
        del models
        torch.cuda.empty_cache()

    report = {"images": len(labels), "folds": fold_reports,
              "views": VIEWS, "configurations": {name: ["base", *views]
                                                  for name, views in CONFIGS.items()},
              "results": {
                  name: {**counts,
                         "top1": counts["top1_correct"] / len(labels),
                         "top5": counts["top5_correct"] / len(labels),
                         "top10": counts["top10_correct"] / len(labels),
                         "net_top1_images": counts["gained"] - counts["lost"]}
                  for name, counts in all_counts.items()},
              "validation": "same stratified three folds; two models for each image trained only on the other folds"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
