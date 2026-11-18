"""Fold-0 pilot: compare external artwork prototypes with held-out images.

The Dex-to-class map and checkpoint must both come from the model trained on
folds 1+2, so fold 0 remains unseen when building the external prototypes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from image_experiment import SEED, stratified_folds
from probe_external_sprite_mapping import render_sprite
from train_large_image import make_model, preprocess


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/large/wrn28_10_f0_stage2_best.pt"))
    parser.add_argument("--mapping", type=Path,
                        default=Path("outputs/large/external_probe/consensus_dex_to_class_f0.npy"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/large/external_probe/prototype_fold0.json"))
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    model = make_model("wrn28_10", False).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu",
                                     weights_only=True))
    captured: list[torch.Tensor] = []

    def capture(_module, inputs):
        captured.append(inputs[0].detach())

    hook = model.head.register_forward_pre_hook(capture)
    mapping = np.load(args.mapping).astype(np.int64) - 1
    if len(mapping) != 1025 or sorted(mapping.tolist()) != list(range(1025)):
        raise ValueError("Mapping must be a one-to-one 1025-class assignment")
    external = []
    for pokemon_id in range(1, 1026):
        views = []
        for variant in ("home", "official-artwork"):
            path = args.data_dir / "external" / "pokeapi" / variant / f"{pokemon_id}.png"
            views.append(render_sprite(path, pokemon_id))
        external.append(np.concatenate(views))
    external = np.concatenate(external)
    features = []
    for start in range(0, len(external), 128):
        x = torch.as_tensor(external[start:start+128].copy(), device=device)
        model(preprocess(x))
        features.append(F.normalize(captured.pop(), dim=1).cpu().numpy())
    prototypes_dex = np.concatenate(features).reshape(1025, 8, -1).mean(axis=1)
    prototypes = np.empty_like(prototypes_dex)
    prototypes[mapping] = prototypes_dex
    prototypes = F.normalize(torch.as_tensor(prototypes, device=device), dim=1)
    del external
    print("External prototypes ready", flush=True)

    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    folds = stratified_folds(labels, 3, SEED)
    indices = np.flatnonzero(folds == 0)
    images = np.load(args.data_dir / "labeled_32x32.npy", mmap_mode="r")
    results = {str(weight): {"top1": 0, "top10": 0} for weight in
               (0, .1, .2, .3, .5, .7, 1)}
    results["count"] = len(indices)
    for offset in range(0, len(indices), 128):
        idx = indices[offset:offset+128]
        x = torch.as_tensor(np.asarray(images[idx]).copy(), device=device)
        logits = model(preprocess(x)).float()
        feature = F.normalize(captured.pop(), dim=1)
        # Compare each image with the eight rendered external views per class.
        similarity = feature @ prototypes.T
        target = torch.as_tensor(labels[idx], device=device)
        base = F.softmax(logits, dim=1)
        reference = F.softmax(similarity / .05, dim=1)
        for weight in (0, .1, .2, .3, .5, .7, 1):
            score = (1-weight)*base + weight*reference
            results[str(weight)]["top1"] += int((score.argmax(1) == target).sum())
            results[str(weight)]["top10"] += int((score.topk(10).indices ==
                                                  target[:, None]).any(1).sum())
        if offset and offset % 5120 == 0:
            print(f"{offset}/{len(indices)}", flush=True)
    for key, value in results.items():
        if key != "count":
            value["top1"] /= len(indices)
            value["top10"] /= len(indices)
    hook.remove()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
