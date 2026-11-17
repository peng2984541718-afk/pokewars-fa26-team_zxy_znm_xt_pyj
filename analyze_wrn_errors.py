"""Export honest out-of-fold diagnostics for the six-checkpoint WRN ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from image_experiment import SEED, stratified_folds
from train_large_image import make_model, preprocess


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/large/wrn_pair_oof_diagnostics.npz"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required")

    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    images = np.load(args.data_dir / "labeled_32x32.npy", mmap_mode="r")
    if len(labels) != len(images) or images.shape[1:] != (32, 32):
        raise ValueError("Image and label caches are not aligned")
    folds = stratified_folds(labels, 3, SEED)
    n = len(labels)
    top10_ids = np.empty((n, 10), dtype=np.uint16)
    top10_probs = np.empty((n, 10), dtype=np.float32)
    true_rank = np.empty(n, dtype=np.uint16)
    true_prob = np.empty(n, dtype=np.float32)
    original_top1 = np.empty(n, dtype=np.uint16)
    mild_top1 = np.empty(n, dtype=np.uint16)
    original_confidence = np.empty(n, dtype=np.float32)
    mild_confidence = np.empty(n, dtype=np.float32)
    entropy = np.empty(n, dtype=np.float32)

    torch.backends.cudnn.benchmark = True
    for fold in range(3):
        models = []
        for prefix in ("wrn28_10", "wrn28_10_mild"):
            model = make_model("wrn28_10", pretrained=False).cuda().eval()
            state = torch.load(args.checkpoint_dir / f"{prefix}_f{fold}_stage2_best.pt",
                               map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            models.append(model)

        valid_indices = np.flatnonzero(folds == fold)
        for start in range(0, len(valid_indices), args.batch_size):
            indices = valid_indices[start:start + args.batch_size]
            raw = torch.as_tensor(np.asarray(images[indices]).copy(), device="cuda")
            x = preprocess(raw)
            per_model = []
            for model in models:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                per_model.append(torch.softmax(logits.float(), dim=1))
            probs = (per_model[0] + per_model[1]) / 2
            target = torch.as_tensor(labels[indices], device="cuda")
            values, candidates = probs.topk(10, dim=1)
            target_probability = probs.gather(1, target[:, None]).squeeze(1)
            top10_ids[indices] = (candidates + 1).cpu().numpy().astype(np.uint16)
            top10_probs[indices] = values.cpu().numpy()
            true_rank[indices] = ((probs > target_probability[:, None]).sum(1) + 1
                                  ).cpu().numpy().astype(np.uint16)
            true_prob[indices] = target_probability.cpu().numpy()
            entropy[indices] = (-(probs * probs.clamp_min(1e-20).log()).sum(1)
                                ).cpu().numpy()
            for p, output_ids, output_confidence in zip(
                    per_model, (original_top1, mild_top1),
                    (original_confidence, mild_confidence)):
                confidence, predicted = p.max(1)
                output_ids[indices] = (predicted + 1).cpu().numpy().astype(np.uint16)
                output_confidence[indices] = confidence.cpu().numpy()
        print(f"fold {fold}: {len(valid_indices):,} held-out images", flush=True)
        del models
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, labels=labels.astype(np.uint16) + 1,
                        folds=folds, top10_ids=top10_ids, top10_probs=top10_probs,
                        true_rank=true_rank, true_prob=true_prob,
                        original_top1=original_top1, mild_top1=mild_top1,
                        original_confidence=original_confidence,
                        mild_confidence=mild_confidence, entropy=entropy)
    print(f"Saved {n:,} diagnostics to {args.output}", flush=True)


if __name__ == "__main__":
    main()
