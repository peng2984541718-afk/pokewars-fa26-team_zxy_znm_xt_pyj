"""Train an image-ID model on all labeled images and predict test-image IDs.

This creates an image-level artifact for the team, not a Kaggle score submission.
Choose the epoch count using image_experiment.py cross-validation first.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from image_experiment import (
    EmbeddingCNN, batch, class_balanced_batch, contrastive_loss, image_cache,
    labeled_images, learning_rate_at_step, normalize, random_translation,
)


def get_test_paths(archive: ZipFile):
    with archive.open("data/data/test.csv") as source:
        rows = pd.read_csv(source, usecols=["noisy_image1", "noisy_image2"])
    return sorted(set(rows.noisy_image1) | set(rows.noisy_image2))


@torch.no_grad()
def predict(model, images, paths, device, destination, batch_size=512):
    model.eval()
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "predicted_pokemon_id", "top1_softmax",
                         "top5_pokemon_ids"])
        for start in range(0, len(paths), batch_size):
            raw = torch.from_numpy(np.asarray(images[start:start + batch_size]).copy())
            inputs = normalize(raw, device)
            _, logits = model(inputs)
            top_values, top_indices = torch.topk(logits, 5, dim=1)
            top_score = torch.softmax(logits, dim=1).gather(1, top_indices[:, :1])
            ids = (top_indices + 1).cpu().numpy()
            scores = top_score[:, 0].cpu().numpy()
            for path, row, score in zip(paths[start:start + batch_size], ids, scores):
                writer.writerow([path, int(row[0]), float(score),
                                 json.dumps([int(value) for value in row])])
    print(f"Saved {len(paths):,} test-image identities to {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("data/competition.zip"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--translate-pixels", type=int, default=1)
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3780)
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()
    if args.epochs <= args.warmup_epochs or args.epochs < 1:
        parser.error("Require epochs > warmup-epochs >= 0")
    if args.batch_size != 128:
        parser.error("Current balanced sampler requires batch size 128")

    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = EmbeddingCNN("raw").to(device)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "image_final_all_labeled.pt"

    if not args.predict_only:
        with ZipFile(args.zip) as archive:
            paths, labels = labeled_images(archive)
            images = image_cache(archive, paths, Path("data/labeled_32x32.npy"))
        rng = np.random.default_rng(args.seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                      weight_decay=1e-3)
        indices = np.arange(len(paths))
        sampler = class_balanced_batch(indices, labels, rng)
        steps = math.ceil(len(paths) / args.batch_size)
        total_steps = args.epochs * steps
        warmup_steps = args.warmup_epochs * steps
        history = []
        start = time.monotonic()
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            correct = 0
            for batch_number in range(steps):
                step = (epoch - 1) * steps + batch_number
                lr = learning_rate_at_step(step, total_steps, warmup_steps,
                                           args.learning_rate, args.min_learning_rate)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                chosen = next(sampler)
                inputs, targets = batch(images, labels, chosen, device)
                inputs = random_translation(inputs, args.translate_pixels)
                embedding, logits = model(inputs)
                loss = F.cross_entropy(logits, targets) + args.contrastive_weight * \
                    contrastive_loss(embedding, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == targets).sum().item()
            history.append({"epoch": epoch, "lr": lr,
                            "sampled_train_loss": total_loss / steps,
                            "sampled_train_top1": correct / (steps * args.batch_size)})
            (output_dir / "image_final_history.json").write_text(
                json.dumps(history, indent=2) + "\n")
            print(f"Final epoch {epoch}/{args.epochs}: loss={total_loss / steps:.3f}, "
                  f"train top1={history[-1]['sampled_train_top1']:.4f}, "
                  f"lr={lr:.2e}, elapsed={time.monotonic() - start:.0f}s", flush=True)
            if epoch % 5 == 0 or epoch == args.epochs:
                torch.save({name: value.detach().cpu()
                            for name, value in model.state_dict().items()}, model_path)
        (output_dir / "image_final_config.json").write_text(json.dumps({
            "epochs": args.epochs, "seed": args.seed,
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "warmup_epochs": args.warmup_epochs,
            "translate_pixels": args.translate_pixels,
            "contrastive_weight": args.contrastive_weight,
            "label_offset": 1,
            "head_input": "raw",
        }, indent=2) + "\n")
    else:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(device)

    with ZipFile(args.zip) as archive:
        test_paths = get_test_paths(archive)
        test_images = image_cache(archive, test_paths, Path("data/test_32x32.npy"))
    predict(model, test_images, test_paths, device,
            output_dir / "test_image_identity_predictions.csv")


if __name__ == "__main__":
    main()
