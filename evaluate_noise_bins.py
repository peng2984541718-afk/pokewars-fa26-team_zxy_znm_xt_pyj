"""Validation-fold accuracy of WRN checkpoints per image-noise decile.

Scores each checkpoint with the original view and ±5° rotations, the same TTA
as export_wrn_oof_probs.py. Noise deciles are computed over all labeled images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from degradation import laplacian_noise
from evaluate_wrn_tta import transform
from image_experiment import SEED, stratified_folds
from train_large_image import make_model, preprocess


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    cache = np.load(args.data_dir / "labeled_32x32.npy")
    noise = laplacian_noise(cache)
    edges = np.quantile(noise, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, noise, side="right") - 1, 0, 9)
    valid = np.flatnonzero(stratified_folds(labels, 3, SEED) == args.fold)
    images = torch.as_tensor(cache[valid], device="cuda")
    report = {"fold": args.fold, "noise_edges": edges.round(2).tolist(),
              "bin_counts": np.bincount(bins[valid], minlength=10).tolist(), "models": {}}
    probs = {}
    for path in args.checkpoints:
        model = make_model("wrn28_10", pretrained=False).cuda().eval()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        out = np.empty((len(valid), 1025), dtype=np.float32)
        for start in range(0, len(valid), args.batch_size):
            x = preprocess(images[start:start + args.batch_size])
            total = 0
            for view in (None, "rotate_minus5", "rotate_plus5"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x if view is None else transform(x, view))
                total = total + torch.softmax(logits.float(), dim=1)
            out[start:start + len(x)] = (total / 3).cpu().numpy()
        probs[path.stem] = out
        del model
    if len(probs) > 1:
        probs["mean_of_all"] = np.mean(list(probs.values()), axis=0)
    for name, p in probs.items():
        ok = p.argmax(1) == labels[valid]
        report["models"][name] = {
            "top1": float(ok.mean()),
            "top1_by_noise_decile": [float(ok[bins[valid] == b].mean()) for b in range(10)]}
        print(name, f"top1={ok.mean():.5f}",
              " ".join(f"{v:.4f}" for v in report["models"][name]["top1_by_noise_decile"]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
