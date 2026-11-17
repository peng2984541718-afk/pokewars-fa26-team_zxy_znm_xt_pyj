"""Predict test-image identities with the three held-out WRN checkpoints.

This writes image-level probabilities only; it does not produce a Kaggle submission.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from image_experiment import NUM_CLASSES
from evaluate_wrn_tta import transform
from train_large_image import make_model, preprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/large/test_wrn28_10_ensemble_80000_top10.csv"))
    parser.add_argument("--limit", type=int, default=80000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--variant", choices=("original", "mild", "ensemble"),
                        default="original")
    parser.add_argument("--tta", choices=("none", "rotate_5deg"), default="none")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required")
    if (not 1 <= args.top_k <= NUM_CLASSES or args.limit < 1 or
            args.offset < 0 or args.batch_size < 1):
        parser.error("Invalid limit, offset, top-k, or batch-size")

    all_paths = (args.data_dir / "test_paths.txt").read_text().splitlines()
    images = np.load(args.data_dir / "test_32x32.npy", mmap_mode="r")
    if len(all_paths) != len(images) or images.shape[1:] != (32, 32):
        raise ValueError("Test paths and image cache are not aligned")
    if all_paths != sorted(set(all_paths)):
        raise ValueError("Test paths must be sorted and unique")
    if args.offset + args.limit > len(all_paths):
        parser.error("Offset plus limit exceeds the number of test images")

    paths = all_paths[args.offset:args.offset + args.limit]
    probabilities = np.zeros((args.limit, NUM_CLASSES), dtype=np.float32)
    prefixes = (["wrn28_10"] if args.variant == "original"
                else ["wrn28_10_mild"] if args.variant == "mild"
                else ["wrn28_10", "wrn28_10_mild"])
    checkpoint_paths = [args.checkpoint_dir / f"{prefix}_f{fold}_stage2_best.pt"
                        for prefix in prefixes for fold in range(3)]
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    for checkpoint_path in checkpoint_paths:
        model = make_model("wrn28_10", pretrained=False).cuda().eval()
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu",
                                         weights_only=True))
        with torch.inference_mode():
            for start in range(0, args.limit, args.batch_size):
                end = min(start + args.batch_size, args.limit)
                raw = torch.as_tensor(
                    np.asarray(images[args.offset + start:args.offset + end]).copy(),
                    device="cuda")
                inputs = preprocess(raw)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(inputs)
                batch_probs = torch.softmax(logits.float(), dim=1)
                if args.tta == "rotate_5deg":
                    for view in ("rotate_minus5", "rotate_plus5"):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            transformed_logits = model(transform(inputs, view))
                        batch_probs += torch.softmax(transformed_logits.float(), dim=1)
                    batch_probs /= 3
                probabilities[start:end] += (batch_probs.cpu().numpy() /
                                             len(checkpoint_paths))
                if end % 10000 == 0 or end == args.limit:
                    print(f"{checkpoint_path.name}: {end:,}/{args.limit:,}", flush=True)
        del model
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = np.argpartition(probabilities, -args.top_k, axis=1)[:, -args.top_k:]
    selected_probs = np.take_along_axis(probabilities, selected, axis=1)
    order = np.argsort(-selected_probs, axis=1)
    top_indices = np.take_along_axis(selected, order, axis=1) + 1
    top_probs = np.take_along_axis(selected_probs, order, axis=1)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "top1_pokemon_id", "top1_probability",
                         "topk_pokemon_ids", "topk_probabilities", "topk_probability_mass"])
        for path, ids, scores in zip(paths, top_indices, top_probs):
            writer.writerow([path, int(ids[0]), float(scores[0]),
                             json.dumps(ids.astype(int).tolist()),
                             json.dumps(scores.astype(float).tolist()),
                             float(scores.sum())])
    metadata = {"images": args.limit, "offset": args.offset, "top_k": args.top_k,
                "variant": args.variant, "tta": args.tta,
                "checkpoints": [str(path) for path in checkpoint_paths],
                "aggregation": "mean of per-checkpoint softmax distributions",
                "calibrated": False,
                "ordering": "slice of sorted unique paths in test_paths.txt"}
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {args.limit:,} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
