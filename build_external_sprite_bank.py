"""Render mapped PokéAPI artwork as noisy 32x32 training images.

Run with a mapping produced by a model that never saw the validation fold.
This program creates a bank only; it does not train or evaluate a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def render(source: Image.Image, rng: np.random.Generator, sample: int) -> np.ndarray:
    rgba = source.convert("RGBA")
    if sample % 2:
        bounds = rgba.getchannel("A").getbbox()
        if bounds:
            rgba = rgba.crop(bounds)
    # Compose at twice the target resolution, then downsample once.
    target = int(rng.uniform(41, 62))
    ratio = min(target / rgba.width, target / rgba.height)
    rgba = rgba.resize((max(1, round(rgba.width * ratio)),
                        max(1, round(rgba.height * ratio))), Image.Resampling.LANCZOS)
    rgba = rgba.rotate(float(rng.uniform(-18, 18)),
                       resample=Image.Resampling.BICUBIC, expand=True)
    canvas = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    shift = rng.integers(-5, 6, size=2)
    canvas.alpha_composite(rgba, ((64-rgba.width)//2 + int(shift[0]),
                                  (64-rgba.height)//2 + int(shift[1])))
    canvas = canvas.resize((32, 32), Image.Resampling.LANCZOS)
    array = np.asarray(canvas, dtype=np.float32)
    opacity = array[..., 3] / 255
    luma = .299*array[..., 0] + .587*array[..., 1] + .114*array[..., 2]
    background = float(rng.uniform(185, 215))
    foreground = (float(rng.uniform(55, 110)) if sample % 2
                  else 25 + .48*luma)
    gray = (1-opacity)*background + opacity*foreground
    gray += rng.normal(0, rng.uniform(24, 48), (32, 32))
    return np.clip(gray, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path,
                        default=Path("outputs/large/external_probe/consensus_dex_to_class_f0.npy"))
    parser.add_argument("--source-dir", type=Path,
                        default=Path("data/external/pokeapi"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/external/pokeapi/train_bank_fold0.npz"))
    parser.add_argument("--samples-per-variant", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3780)
    args = parser.parse_args()
    mapping = np.load(args.mapping).astype(np.int64) - 1
    if len(mapping) != 1025 or sorted(mapping.tolist()) != list(range(1025)):
        raise ValueError("Expected a one-to-one mapping of 1,025 classes")
    rng = np.random.default_rng(args.seed)
    images, labels = [], []
    for dex_id in range(1, 1026):
        for variant in ("home", "official-artwork"):
            with Image.open(args.source_dir / variant / f"{dex_id}.png") as source:
                for sample in range(args.samples_per_variant):
                    images.append(render(source, rng, sample))
                    labels.append(mapping[dex_id-1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, images=np.stack(images),
                        labels=np.asarray(labels, dtype=np.uint16))
    report = {"images": len(images), "classes": len(set(labels)),
              "samples_per_variant": args.samples_per_variant,
              "sources": ["home", "official-artwork"],
              "mapping": str(args.mapping), "seed": args.seed}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
