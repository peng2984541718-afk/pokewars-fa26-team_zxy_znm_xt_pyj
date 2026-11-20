"""Export fold-safe full-class OOF probabilities for one WRN checkpoint prefix.

Each labeled image is scored by the fold model that did not train on it,
averaged over the original view and ±5° rotations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from evaluate_wrn_tta import transform
from image_experiment import NUM_CLASSES, SEED, stratified_folds
from train_large_image import make_model, preprocess


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True,
                        help="Checkpoints are <prefix>_f{fold}_stage2_best.pt")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    cache = np.load(args.data_dir / "labeled_32x32.npy", mmap_mode="r")
    if len(cache) != len(labels):
        raise ValueError("Image and label cache mismatch")
    folds = stratified_folds(labels, 3, SEED)
    images = torch.as_tensor(np.asarray(cache).copy(), device="cuda")
    torch.backends.cudnn.benchmark = True
    output = np.empty((len(labels), NUM_CLASSES), dtype=np.float16)
    for fold in range(3):
        model = make_model("wrn28_10", pretrained=False).cuda().eval()
        model.load_state_dict(torch.load(args.checkpoint_dir / f"{args.prefix}_f{fold}_stage2_best.pt",
                                         map_location="cpu", weights_only=True))
        indices = np.flatnonzero(folds == fold)
        for start in range(0, len(indices), args.batch_size):
            selected = indices[start:start + args.batch_size]
            x = preprocess(images[torch.as_tensor(selected, device="cuda")])
            total = 0
            for view in (None, "rotate_minus5", "rotate_plus5"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x if view is None else transform(x, view))
                total = total + torch.softmax(logits.float(), dim=1)
            output[selected] = (total / 3).cpu().numpy()
        accuracy = np.mean(output[indices].argmax(1) == labels[indices])
        print(f"{args.prefix} fold {fold}: top1={accuracy:.5f}", flush=True)
        del model
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output)
    print(f"{args.prefix} overall top1={np.mean(output.argmax(1) == labels):.5f}", flush=True)


if __name__ == "__main__":
    main()
