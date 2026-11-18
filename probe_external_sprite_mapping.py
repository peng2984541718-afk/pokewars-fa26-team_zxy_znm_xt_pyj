"""Ask a fold-safe image model whether public sprites map to competition IDs.

This is a feasibility probe. It does not use external sprites to train a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_large_image import make_model, preprocess


def render_sprite(path: Path, pokemon_id: int) -> np.ndarray:
    source = Image.open(path).convert("RGBA")
    alpha = source.getchannel("A")
    layouts = []
    layouts.append(source.resize((32, 32), Image.Resampling.BILINEAR))
    bounds = alpha.getbbox()
    if bounds is None:
        raise ValueError(f"No opaque pixels in {path}")
    cropped = source.crop(bounds)
    cropped.thumbnail((28, 28), Image.Resampling.BILINEAR)
    centered = Image.new("RGBA", (32, 32), (255, 255, 255, 0))
    centered.alpha_composite(cropped, ((32-cropped.width)//2,
                                       (32-cropped.height)//2))
    layouts.append(centered)
    views = []
    rng = np.random.default_rng(pokemon_id)
    for layout in layouts:
        rgba = np.asarray(layout, dtype=np.float32)
        opacity = rgba[:, :, 3] / 255.0
        luma = (.299*rgba[:, :, 0] + .587*rgba[:, :, 1] +
                .114*rgba[:, :, 2])
        for foreground in (np.full((32, 32), 75, dtype=np.float32),
                           30 + .4*luma):
            gray = 200*(1-opacity) + foreground*opacity
            noise = rng.normal(0, 35, size=(32, 32))
            views.append(np.clip(gray + noise, 0, 255).astype(np.uint8))
    return np.stack(views)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("default", "home", "official-artwork"),
                        required=True)
    parser.add_argument("--source-dir", type=Path,
                        default=Path("data/external/pokeapi"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/large/wrn28_10_f0_stage2_best.pt"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/large/external_probe"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    model = make_model("wrn28_10", pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu",
                                     weights_only=True))
    ids = []
    images = []
    for pokemon_id in range(1, 1026):
        path = args.source_dir / args.variant / f"{pokemon_id}.png"
        if path.is_file():
            ids.append(pokemon_id)
            images.append(render_sprite(path, pokemon_id))
    if not ids:
        raise ValueError("No sprite images found")
    raw = np.concatenate(images, axis=0)
    predictions = np.empty((len(raw), 1025), dtype=np.float32)
    for start in range(0, len(raw), args.batch_size):
        batch = torch.as_tensor(raw[start:start+args.batch_size].copy(), device=device)
        x = preprocess(batch)
        logits = model(x).float()
        predictions[start:start+len(batch)] = torch.softmax(logits, dim=1).cpu().numpy()
        if start and start % 1024 == 0:
            print(f"{args.variant}: {start}/{len(raw)} views", flush=True)
    per_source = predictions.reshape(len(ids), 4, 1025)
    scores = per_source.mean(axis=1)
    assigned = scores.argmax(axis=1) + 1
    confidence = scores.max(axis=1)
    sorted_scores = np.sort(scores, axis=1)
    output = {
        "variant": args.variant,
        "source_count": len(ids),
        "unique_assigned_competition_classes": len(set(assigned)),
        "median_top1_score": float(np.median(confidence)),
        "fraction_top1_above_0.5": float(np.mean(confidence > .5)),
        "median_top1_margin": float(np.median(sorted_scores[:, -1]-sorted_scores[:, -2])),
        "same_numeric_id_top1": int(np.sum(assigned == ids)),
        "views": "full/cropped x silhouette/luma; background 200, Gaussian noise std 35",
        "checkpoint": str(args.checkpoint),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / f"{args.variant}_wrn_f0_mapping.npz",
                        source_ids=np.asarray(ids, dtype=np.uint16),
                        scores=scores.astype(np.float16),
                        per_view_top1=per_source.argmax(axis=2).astype(np.uint16) + 1)
    (args.output_dir / f"{args.variant}_wrn_f0_mapping.json").write_text(
        json.dumps(output, indent=2) + "\n")
    print(output, flush=True)
    for pokemon_id in (1, 25, 150, 770, 993):
        if pokemon_id in ids:
            j = ids.index(pokemon_id)
            print(f"source {pokemon_id}: class {assigned[j]} "
                  f"score={confidence[j]:.3f}", flush=True)


if __name__ == "__main__":
    main()
