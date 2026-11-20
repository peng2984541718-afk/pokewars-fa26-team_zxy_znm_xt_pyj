"""Export fold-safe full-class OOF probabilities for WRN and ConvNeXt V2."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_wrn_tta import transform
from image_experiment import NUM_CLASSES, SEED, stratified_folds
from train_large_image import make_model, preprocess


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required")

    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    cache = np.load(args.data_dir / "labeled_32x32.npy", mmap_mode="r")
    if len(cache) != len(labels) or cache.shape[1:] != (32, 32):
        raise ValueError("Image and label cache mismatch")
    folds = stratified_folds(labels, 3, SEED)
    images = torch.as_tensor(np.asarray(cache).copy(), device="cuda")
    torch.backends.cudnn.benchmark = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shape = (len(labels), NUM_CLASSES)
    outputs = {}
    for name in ("wrn_rot5", "convnextv2"):
        outputs[name] = np.lib.format.open_memmap(
            args.output_dir / f"multibackbone_{name}_oof.tmp.npy", mode="w+",
            dtype=np.float16, shape=shape)

    for fold in range(3):
        indices = np.flatnonzero(folds == fold)
        for name in outputs:
            prefixes = (("wrn28_10", "wrn28_10_mild") if name == "wrn_rot5"
                        else ("convnextv2_scratch64",))
            for prefix in prefixes:
                model_name = "wrn28_10" if name == "wrn_rot5" else "convnextv2_nano"
                model = make_model(model_name, pretrained=False).cuda().eval()
                checkpoint = args.checkpoint_dir / f"{prefix}_f{fold}_stage2_best.pt"
                model.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                                 weights_only=True))
                for start in range(0, len(indices), args.batch_size):
                    selected = indices[start:start + args.batch_size]
                    x = preprocess(images[torch.as_tensor(selected, device="cuda")])
                    if name == "convnextv2":
                        x = F.interpolate(x, size=64, mode="bilinear",
                                          align_corners=False)
                    total = None
                    views = ((None, "rotate_minus5", "rotate_plus5")
                             if name == "wrn_rot5" else (None,))
                    for view in views:
                        inputs = x if view is None else transform(x, view)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(inputs)
                        probs = torch.softmax(logits.float(), dim=1)
                        total = probs if total is None else total + probs
                    contribution = (total / (len(views) * len(prefixes))).cpu().numpy()
                    if len(prefixes) == 1 or prefix == prefixes[0]:
                        outputs[name][selected] = contribution
                    else:
                        outputs[name][selected] = (outputs[name][selected].astype(np.float32)
                                                   + contribution)
                print(f"fold {fold}: exported {prefix}", flush=True)
                del model
                torch.cuda.empty_cache()
        for output in outputs.values():
            output.flush()

    for name, output in outputs.items():
        output.flush()
        del output
        source = args.output_dir / f"multibackbone_{name}_oof.tmp.npy"
        destination = args.output_dir / f"multibackbone_{name}_oof.npy"
        source.replace(destination)
        print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
