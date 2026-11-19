"""Export full-class (N, 1025) float16 probabilities with ±5° rotation TTA.

Supports fold-safe OOF export on labeled_32x32.npy (--split oof) and ensemble
query export on test_32x32.npy inside --query-dir (--split query) across any
backbone family (wrn28_10, wrn40_10, convnextv2_nano, densenet169_cifar).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from degradation import clean_impulses_gpu
from evaluate_wrn_tta import transform
from image_experiment import NUM_CLASSES, SEED, stratified_folds
from train_large_image import make_model, preprocess


def checkpoint_path(checkpoint_dir: Path, prefix: str, fold: int) -> Path:
    for suffix in ("_stage2_best.pt", "_best.pt", "_last.pt"):
        candidate = checkpoint_dir / f"{prefix}_f{fold}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found for {prefix}_f{fold} in {checkpoint_dir}")


@torch.inference_mode()
def score_model(model: torch.nn.Module, model_name: str, images: torch.Tensor,
                indices: np.ndarray, batch_size: int) -> np.ndarray:
    out = np.empty((len(indices), NUM_CLASSES), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        idx = torch.as_tensor(indices[start:end], device="cuda")
        x0 = preprocess(images[idx])
        total = 0
        for view in (None, "rotate_minus5", "rotate_plus5"):
            xv = x0 if view is None else transform(x0, view)
            if model_name == "convnextv2_nano":
                xv = F.interpolate(xv, size=64, mode="bilinear", align_corners=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xv)
            total = total + torch.softmax(logits.float(), dim=1)
        out[start:end] = (total / 3.0).cpu().numpy()
    return out


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("wrn28_10", "wrn40_10", "convnextv2_nano",
                                            "densenet169_cifar"), required=True)
    parser.add_argument("--prefixes", nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    parser.add_argument("--split", choices=("oof", "query"), required=True)
    parser.add_argument("--folds", type=int, nargs="+", choices=(0, 1, 2), default=[0, 1, 2])
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--query-dir", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--clean-impulses", action="store_true",
                        help="Repair exact 25/245 impulse pixels before inference")
    args = parser.parse_args()

    weights = np.array(args.weights if args.weights else [1.0] * len(args.prefixes), dtype=np.float64)
    if len(weights) != len(args.prefixes):
        parser.error("--weights must match --prefixes")
    weights /= weights.sum()
    torch.backends.cudnn.benchmark = True

    if args.split == "oof":
        labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
        cache = np.load(args.data_dir / "labeled_32x32.npy", mmap_mode="r")
        folds = stratified_folds(labels, 3, SEED)
        images = torch.as_tensor(np.asarray(cache).copy(), device="cuda")
        if args.clean_impulses:
            images = clean_impulses_gpu(images)
        output = np.zeros((len(labels), NUM_CLASSES), dtype=np.float16)
        for fold in range(3):
            valid_idx = np.flatnonzero(folds == fold)
            fold_acc = np.zeros((len(valid_idx), NUM_CLASSES), dtype=np.float32)
            for prefix, w in zip(args.prefixes, weights):
                ckpt = checkpoint_path(args.checkpoint_dir, prefix, fold)
                model = make_model(args.model, pretrained=False).cuda().eval()
                model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
                fold_acc += float(w) * score_model(model, args.model, images, valid_idx, args.batch_size)
                del model
                torch.cuda.empty_cache()
            output[valid_idx] = fold_acc.astype(np.float16)
            print(f"OOF fold {fold}: top1={np.mean(fold_acc.argmax(1) == labels[valid_idx]):.5f}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, output)
        print(f"Saved {args.output}: overall top1={np.mean(output.argmax(1) == labels):.5f}", flush=True)
    else:
        paths = (args.query_dir / "test_paths.txt").read_text().splitlines()
        cache = np.load(args.query_dir / "test_32x32.npy", mmap_mode="r")
        if len(paths) != len(cache):
            raise ValueError("Query paths and image cache mismatch")
        images = torch.as_tensor(np.asarray(cache).copy(), device="cuda")
        if args.clean_impulses:
            images = clean_impulses_gpu(images)
        all_idx = np.arange(len(paths))
        acc = np.zeros((len(paths), NUM_CLASSES), dtype=np.float32)
        for fold in args.folds:
            for prefix, w in zip(args.prefixes, weights):
                ckpt = checkpoint_path(args.checkpoint_dir, prefix, fold)
                model = make_model(args.model, pretrained=False).cuda().eval()
                model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
                acc += (float(w) / len(args.folds)) * score_model(
                    model, args.model, images, all_idx, args.batch_size)
                del model
                torch.cuda.empty_cache()
                print(f"Scored {ckpt.name} on {len(paths)} query images", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, acc.astype(np.float16))
        print(f"Saved {args.output} ({acc.shape})", flush=True)


if __name__ == "__main__":
    main()
