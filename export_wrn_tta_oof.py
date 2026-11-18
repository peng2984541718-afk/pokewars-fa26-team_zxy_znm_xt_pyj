"""Export matched-fold six-WRN top-10 candidates with ±5° rotation TTA."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from evaluate_wrn_tta import transform
from image_experiment import SEED, stratified_folds
from train_large_image import make_model, preprocess


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/large/wrn_pair_rot5_oof_candidates.npz"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required")
    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    cache = np.load(args.data_dir / "labeled_32x32.npy", mmap_mode="r")
    if len(labels) != len(cache) or cache.shape[1:] != (32, 32):
        raise ValueError("Image and label cache mismatch")
    folds = stratified_folds(labels, 3, SEED)
    images = torch.as_tensor(np.asarray(cache).copy(), device="cuda")
    ids = np.empty((len(labels), 10), dtype=np.uint16)
    probs = np.empty((len(labels), 10), dtype=np.float32)
    torch.backends.cudnn.benchmark = True

    for fold in range(3):
        models = []
        for prefix in ("wrn28_10", "wrn28_10_mild"):
            model = make_model("wrn28_10", pretrained=False).cuda().eval()
            checkpoint = args.checkpoint_dir / f"{prefix}_f{fold}_stage2_best.pt"
            model.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                             weights_only=True))
            models.append(model)
        indices = np.flatnonzero(folds == fold)
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            index = torch.as_tensor(batch_indices, device="cuda")
            raw = images[index]
            x = preprocess(raw)
            total = None
            for model in models:
                for view in (None, "rotate_minus5", "rotate_plus5"):
                    inputs = x if view is None else transform(x, view)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(inputs)
                    p = torch.softmax(logits.float(), dim=1)
                    total = p if total is None else total + p
            values, chosen = (total / 6).topk(10, dim=1)
            ids[batch_indices] = (chosen + 1).cpu().numpy().astype(np.uint16)
            probs[batch_indices] = values.cpu().numpy()
        print(f"fold {fold}: {len(indices):,} images", flush=True)
        del models
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, labels=labels.astype(np.uint16) + 1,
                        folds=folds, top10_ids=ids, top10_probs=probs)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
