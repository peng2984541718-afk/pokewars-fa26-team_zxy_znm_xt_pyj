"""Measure held-out top-k accuracy of the three WRN continuation checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from image_experiment import SEED, stratified_folds
from train_large_image import make_model, preprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--variant", choices=("original", "mild", "ensemble"),
                        default="original")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/large/wrn28_10_3fold_topk.json"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required")

    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    images = np.load(args.data_dir / "labeled_32x32.npy", mmap_mode="r")
    if len(labels) != len(images) or images.shape[1:] != (32, 32):
        raise ValueError("Labeled paths and image cache are not aligned")
    folds = stratified_folds(labels, 3, SEED)
    totals = {"top1": 0, "top5": 0, "top10": 0}
    results = []

    torch.backends.cudnn.benchmark = True
    for fold in range(3):
        names = ([f"wrn28_10_f{fold}_stage2_best.pt"] if args.variant == "original"
                 else [f"wrn28_10_mild_f{fold}_stage2_best.pt"] if args.variant == "mild"
                 else [f"wrn28_10_f{fold}_stage2_best.pt",
                       f"wrn28_10_mild_f{fold}_stage2_best.pt"])
        models = []
        for name in names:
            model = make_model("wrn28_10", pretrained=False).cuda().eval()
            model.load_state_dict(torch.load(args.checkpoint_dir / name,
                                             map_location="cpu", weights_only=True))
            models.append(model)
        valid_indices = np.flatnonzero(folds == fold)
        correct = {"top1": 0, "top5": 0, "top10": 0}
        with torch.inference_mode():
            for start in range(0, len(valid_indices), args.batch_size):
                batch_indices = valid_indices[start:start + args.batch_size]
                raw = torch.as_tensor(np.asarray(images[batch_indices]).copy(), device="cuda")
                inputs = preprocess(raw)
                target = torch.as_tensor(labels[batch_indices], device="cuda")
                if len(models) == 1:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        scores = models[0](inputs)
                else:
                    scores = None
                    for model in models:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(inputs)
                        p = torch.softmax(logits.float(), dim=1)
                        scores = p if scores is None else scores + p
                top10 = scores.topk(10, dim=1).indices
                hits = top10.eq(target[:, None])
                correct["top1"] += int(hits[:, :1].any(1).sum().item())
                correct["top5"] += int(hits[:, :5].any(1).sum().item())
                correct["top10"] += int(hits.any(1).sum().item())
        for key in totals:
            totals[key] += correct[key]
        result = {"fold": fold, "count": len(valid_indices),
                  **{key: correct[key] / len(valid_indices) for key in correct}}
        results.append(result)
        print(json.dumps(result), flush=True)
        del models
        torch.cuda.empty_cache()

    summary = {"count": len(labels),
               **{key: totals[key] / len(labels) for key in totals},
               "folds": results, "variant": args.variant,
               "checkpoints": "original/mild best stage2 weights for each held-out fold",
               "validation": "stratified 3-fold held-out unique images"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("count", "top1", "top5", "top10")}))


if __name__ == "__main__":
    main()
