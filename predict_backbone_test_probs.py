"""Average three fold checkpoints on the 160k test images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from image_experiment import NUM_CLASSES
from train_large_image import make_model, preprocess


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("convnextv2_nano", "densenet169_cifar"),
                        required=True)
    parser.add_argument("--checkpoint-prefix", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    paths = (args.data_dir / "test_paths.txt").read_text().splitlines()
    images = np.load(args.data_dir / "test_32x32.npy", mmap_mode="r")
    if len(paths) != len(images) or paths != sorted(set(paths)):
        raise ValueError("Test cache/path mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npy")
    probs = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16,
                                       shape=(len(paths), NUM_CLASSES))
    for fold in range(3):
        suffix = "_best.pt" if args.model == "densenet169_cifar" else "_stage2_best.pt"
        checkpoint = args.checkpoint_dir / f"{args.checkpoint_prefix}_f{fold}{suffix}"
        model = make_model(args.model, False).cuda().eval()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        for start in range(0, len(paths), args.batch_size):
            end = min(start+args.batch_size, len(paths))
            raw = torch.as_tensor(np.asarray(images[start:end]).copy(), device="cuda")
            x = preprocess(raw)
            if args.model == "convnextv2_nano":
                x = F.interpolate(x, size=64, mode="bilinear", align_corners=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            value = torch.softmax(logits.float(), dim=1).cpu().numpy() / 3
            probs[start:end] = value if fold == 0 else probs[start:end].astype(np.float32) + value
            if end % 20000 == 0 or end == len(paths):
                print(f"{checkpoint.name}: {end}/{len(paths)}", flush=True)
        probs.flush()
        del model
        torch.cuda.empty_cache()
    del probs
    temporary.replace(args.output)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
