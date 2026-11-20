"""Export full held-out probabilities for a newly trained image backbone."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from image_experiment import NUM_CLASSES, SEED, stratified_folds
from train_large_image import make_model, preprocess


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("densenet169_cifar", "wrn40_10"), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    folds = stratified_folds(labels, 3, SEED)
    indices = np.flatnonzero(folds == args.fold)
    cache = np.load(args.data_dir / "labeled_32x32.npy", mmap_mode="r")
    if len(cache) != len(labels):
        raise ValueError("Image and label cache mismatch")
    images = torch.as_tensor(np.asarray(cache).copy(), device="cuda")
    model = make_model(args.model, pretrained=False).cuda().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    probs = np.empty((len(indices), NUM_CLASSES), dtype=np.float16)
    for start in range(0, len(indices), args.batch_size):
        batch_indices = indices[start:start+args.batch_size]
        x = preprocess(images[torch.as_tensor(batch_indices, device="cuda")])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        probs[start:start+len(batch_indices)] = torch.softmax(logits.float(), dim=1).half().cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, indices=indices, labels=labels[indices].astype(np.uint16),
                        probabilities=probs)
    print(f"Saved fold {args.fold}: {len(indices)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
