"""Local supervised-contrastive image recognition experiment for Pokéwars.

Only labeled training images are used. No battle scores or Kaggle submissions.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


SEED = 3780
NUM_CLASSES = 1025


def labeled_images(archive: ZipFile):
    with archive.open("data/data/train.csv") as source:
        rows = pd.read_csv(source)
    image_to_id = {}
    for side in (1, 2):
        paths = rows[f"noisy_image{side}"]
        ids = rows[f"pokemon_id{side}"]
        for path, identity in zip(paths[ids.notna()], ids[ids.notna()].astype(int)):
            previous = image_to_id.setdefault(path, identity)
            if previous != identity:
                raise ValueError(f"Conflicting labels for {path}")
    paths = sorted(image_to_id)
    labels = np.array([image_to_id[path] - 1 for path in paths], dtype=np.int64)
    if labels.min() != 0 or labels.max() != NUM_CLASSES - 1:
        raise ValueError("Expected IDs 1 through 1025")
    return paths, labels


def image_cache(archive: ZipFile, paths: list[str], cache_path: Path):
    expected = (len(paths), 32, 32)
    if cache_path.exists():
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape == expected and cached.dtype == np.uint8:
            print(f"Loaded {len(paths):,} cached labeled images", flush=True)
            return cached
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.uint8,
                                        shape=expected)
    start = time.monotonic()
    for i, path in enumerate(paths):
        member = "data/data/" + path.removeprefix("images/")
        with Image.open(io.BytesIO(archive.read(member))) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.uint8)
        if pixels.shape != (32, 32):
            raise ValueError(f"Unexpected image size: {path}")
        cached[i] = pixels
        if (i + 1) % 40000 == 0:
            print(f"Decoded {i + 1:,}/{len(paths):,} images in "
                  f"{time.monotonic() - start:.0f}s", flush=True)
    cached.flush()
    return cached


def stratified_folds(labels: np.ndarray, count: int, seed: int):
    rng = np.random.default_rng(seed)
    folds = np.empty(len(labels), dtype=np.int8)
    for identity in range(NUM_CLASSES):
        indices = np.flatnonzero(labels == identity)
        rng.shuffle(indices)
        if len(indices) < count:
            raise ValueError(f"Class {identity + 1} has fewer than {count} images")
        folds[indices] = np.arange(len(indices)) % count
    return folds


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, images):
        return self.layers(images)


class EmbeddingCNN(nn.Module):
    def __init__(self, head_input="raw"):
        super().__init__()
        self.head_input = head_input
        self.encoder = nn.Sequential(
            ConvBlock(1, 32), ConvBlock(32, 64), ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
        )
        self.classifier = nn.Linear(128, NUM_CLASSES)

    def forward(self, images):
        raw_embedding = self.encoder(images)
        embedding = F.normalize(raw_embedding, dim=1)
        head_features = raw_embedding if self.head_input == "raw" else embedding
        return embedding, self.classifier(head_features)


def normalize(raw: torch.Tensor, device: torch.device):
    image = raw.to(device=device, dtype=torch.float32).unsqueeze(1) / 255.0
    mean = image.mean(dim=(2, 3), keepdim=True)
    std = image.std(dim=(2, 3), keepdim=True).clamp_min(0.08)
    return (image - mean) / std


def batch(images: np.ndarray, labels: np.ndarray, indices: np.ndarray, device):
    raw = torch.from_numpy(np.asarray(images[indices]).copy())
    target = torch.from_numpy(labels[indices]).to(device)
    return normalize(raw, device), target


def random_translation(images: torch.Tensor, pixels: int):
    """Shift each image while padding with zero (its normalized mean gray)."""
    if pixels <= 0:
        return images
    n = len(images)
    offsets = torch.randint(-pixels, pixels + 1, (n, 2), device=images.device)
    transform = torch.zeros((n, 2, 3), device=images.device)
    transform[:, 0, 0] = 1
    transform[:, 1, 1] = 1
    transform[:, :, 2] = offsets * (2.0 / images.shape[-1])
    grid = F.affine_grid(transform, images.shape, align_corners=False)
    return F.grid_sample(images, grid, padding_mode="zeros", align_corners=False)


def learning_rate_at_step(step, total_steps, warmup_steps, peak_lr, min_lr):
    if step < warmup_steps:
        return peak_lr * (0.1 + 0.9 * (step + 1) / max(warmup_steps, 1))
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
    return min_lr + (peak_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2


def contrastive_loss(embedding: torch.Tensor, targets: torch.Tensor, temperature=0.1):
    similarities = embedding @ embedding.T / temperature
    eye = torch.eye(len(targets), device=targets.device, dtype=torch.bool)
    positive = (targets[:, None] == targets[None, :]) & ~eye
    logits = similarities.masked_fill(eye, -1e4)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -((log_prob * positive).sum(dim=1) /
             positive.sum(dim=1).clamp_min(1)).mean()


def class_balanced_batch(indices, labels, rng, classes_per_batch=32,
                         images_per_class=4):
    class_indices = [indices[labels[indices] == identity]
                     for identity in range(NUM_CLASSES)]
    available = np.array([identity for identity, members in enumerate(class_indices)
                          if len(members) >= 2])
    while True:
        selected_classes = rng.choice(available, classes_per_batch, replace=False)
        chosen = np.concatenate([
            rng.choice(class_indices[identity], images_per_class,
                       replace=len(class_indices[identity]) < images_per_class)
            for identity in selected_classes
        ])
        rng.shuffle(chosen)
        yield chosen


@torch.no_grad()
def encode(model, images, labels, indices, device, batch_size=512):
    model.eval()
    embeddings = np.empty((len(indices), 128), dtype=np.float32)
    logits = np.empty((len(indices), NUM_CLASSES), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        sample = indices[start:start + batch_size]
        inputs, _ = batch(images, labels, sample, device)
        vectors, scores = model(inputs)
        embeddings[start:start + len(sample)] = vectors.cpu().numpy()
        logits[start:start + len(sample)] = scores.cpu().numpy()
    return embeddings, logits


def topk_accuracy(scores, labels, k):
    top = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top == labels[:, None], axis=1)))


def cross_entropy_from_logits(logits, labels):
    maxima = logits.max(axis=1)
    shifted = logits - maxima[:, None]
    return float(np.mean(maxima + np.log(np.exp(shifted).sum(axis=1))
                         - logits[np.arange(len(labels)), labels]))


def evaluate_prototypes(model, images, labels, training, validation, device):
    train_vectors, _ = encode(model, images, labels, training, device)
    valid_vectors, valid_logits = encode(model, images, labels, validation, device)
    prototypes = np.zeros((NUM_CLASSES, 128), dtype=np.float32)
    np.add.at(prototypes, labels[training], train_vectors)
    counts = np.bincount(labels[training], minlength=NUM_CLASSES)
    prototypes /= np.maximum(counts[:, None], 1)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8)
    similarities = valid_vectors @ prototypes.T
    expected = labels[validation]
    predictions = np.argmax(similarities, axis=1)
    class_count = np.bincount(expected, minlength=NUM_CLASSES)
    class_correct = np.bincount(expected[predictions == expected], minlength=NUM_CLASSES)
    return {
        "prototype_top1": topk_accuracy(similarities, expected, 1),
        "prototype_top5": topk_accuracy(similarities, expected, 5),
        "prototype_macro_accuracy": float(np.mean(class_correct / np.maximum(class_count, 1))),
        "linear_head_top1": topk_accuracy(valid_logits, expected, 1),
        "linear_head_top5": topk_accuracy(valid_logits, expected, 5),
    }


def train_fold(images, labels, folds, fold, args, device):
    rng = np.random.default_rng(args.seed + fold)
    torch.manual_seed(args.seed + fold)
    training = np.flatnonzero(folds != fold)
    validation = np.flatnonzero(folds == fold)
    model = EmbeddingCNN(args.head_input).to(device)
    if args.initial_weights is not None:
        state = torch.load(args.initial_weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-3)
    steps = args.steps_per_epoch or math.ceil(len(training) / args.batch_size)
    total_steps = args.epochs * steps
    warmup_steps = args.warmup_epochs * steps
    sampler = class_balanced_batch(training, labels, rng,
                                   args.classes_per_batch, args.images_per_class)
    best = {"accuracy": -1.0, "state": None, "epoch": 0}
    history = []
    checkpoint = Path(f"outputs/image_{args.run_name}_fold{fold + 1}.pt")
    start = time.monotonic()
    if args.initial_weights is not None:
        _, initial_logits = encode(model, images, labels, validation, device)
        initial_accuracy = topk_accuracy(initial_logits, labels[validation], 1)
        best = {"accuracy": initial_accuracy,
                "state": {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()},
                "epoch": 0}
        torch.save(best["state"], checkpoint)
        history.append({"epoch": 0, "lr": None, "train_loss": None,
                        "sampled_train_top1": None,
                        "validation_loss": cross_entropy_from_logits(
                            initial_logits, labels[validation]),
                        "validation_top1": initial_accuracy,
                        "validation_top5": topk_accuracy(initial_logits,
                                                          labels[validation], 5)})
        print(f"Initial validation top1={initial_accuracy:.4f}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        for batch_number in range(steps):
            step = (epoch - 1) * steps + batch_number
            current_lr = learning_rate_at_step(step, total_steps, warmup_steps,
                                               args.learning_rate, args.min_learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            chosen = next(sampler)
            inputs, targets = batch(images, labels, chosen, device)
            inputs = random_translation(inputs, args.translate_pixels)
            vectors, logits = model(inputs)
            loss = F.cross_entropy(logits, targets) + args.contrastive_weight * \
                contrastive_loss(vectors, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item()
            correct += (logits.argmax(dim=1) == targets).sum().item()
        _, valid_logits = encode(model, images, labels, validation, device)
        accuracy = topk_accuracy(valid_logits, labels[validation], 1)
        top5 = topk_accuracy(valid_logits, labels[validation], 5)
        valid_loss = cross_entropy_from_logits(valid_logits, labels[validation])
        history.append({"epoch": epoch, "lr": current_lr,
                        "train_loss": loss_sum / steps,
                        "sampled_train_top1": correct / (steps * args.batch_size),
                        "validation_loss": valid_loss,
                        "validation_top1": accuracy, "validation_top5": top5})
        Path(f"outputs/image_{args.run_name}_fold{fold + 1}_history.json").write_text(
            json.dumps(history, indent=2) + "\n")
        print(f"Fold {fold + 1}/{args.folds} epoch {epoch}/{args.epochs}: "
              f"lr={current_lr:.2e}, loss={loss_sum / steps:.3f}, "
              f"train top1={history[-1]['sampled_train_top1']:.4f}, "
              f"val top1={accuracy:.4f}, top5={top5:.4f}, "
              f"elapsed={time.monotonic() - start:.0f}s", flush=True)
        if accuracy > best["accuracy"] + args.min_improvement:
            best = {"accuracy": accuracy,
                    "state": {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()},
                    "epoch": epoch}
            torch.save(best["state"], checkpoint)
        if epoch >= args.min_epochs and epoch - best["epoch"] >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best['epoch']}", flush=True)
            break
    model.load_state_dict(best["state"])
    metrics = evaluate_prototypes(model, images, labels, training, validation, device)
    metrics.update({"fold": fold + 1, "train_images": len(training),
                    "validation_images": len(validation), "best_epoch": best["epoch"],
                    "epochs_run": epoch,
                    "elapsed_seconds": round(time.monotonic() - start, 1)})
    return metrics, best["state"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("data/competition.zip"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold", type=int, default=None, help="Run only this zero-based fold")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Maximum epochs; early stopping may finish sooner")
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-improvement", type=float, default=0.001,
                        help="Minimum absolute top-1 improvement to save a new best")
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--translate-pixels", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="Use a short smoke test; omit for full training")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--classes-per-batch", type=int, default=32)
    parser.add_argument("--images-per-class", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run-name", default="contrastive",
                        help="Output name; use a distinct name for ablations")
    parser.add_argument("--head-input", choices=("raw", "normalized"), default="raw")
    parser.add_argument("--initial-weights", type=Path, default=None,
                        help="Fine-tune an existing checkpoint; epoch 0 is evaluated")
    args = parser.parse_args()
    if args.folds < 2 or args.classes_per_batch * args.images_per_class != args.batch_size:
        parser.error("Require at least 2 folds and batch size = classes × images per class")
    if args.fold is not None and not 0 <= args.fold < args.folds:
        parser.error("--fold must be in [0, folds)")
    if args.epochs < 1 or args.warmup_epochs >= args.epochs:
        parser.error("Require epochs > warmup-epochs >= 0")
    if args.min_epochs < 1 or args.patience < 1 or args.translate_pixels < 0:
        parser.error("Invalid early-stopping or translation settings")

    with ZipFile(args.zip) as archive:
        paths, labels = labeled_images(archive)
        images = image_cache(archive, paths, Path("data/labeled_32x32.npy"))
    folds = stratified_folds(labels, args.folds, args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device={device}, labeled images={len(paths):,}, "
          f"classes={len(np.unique(labels))}, parameters="
          f"{sum(p.numel() for p in EmbeddingCNN().parameters()):,}", flush=True)
    Path("outputs").mkdir(exist_ok=True)
    requested = [args.fold] if args.fold is not None else range(args.folds)
    results = []
    for fold in requested:
        metrics, state = train_fold(images, labels, folds, fold, args, device)
        results.append(metrics)
        torch.save(state, f"outputs/image_{args.run_name}_fold{fold + 1}.pt")
        total_validation = sum(item["validation_images"] for item in results)
        summary = {
            key: sum(item[key] * item["validation_images"] for item in results)
                 / total_validation
            for key in ("linear_head_top1", "linear_head_top5",
                        "prototype_top1", "prototype_top5",
                        "prototype_macro_accuracy")
        }
        Path(f"outputs/image_{args.run_name}_cv.json").write_text(json.dumps({
            "seed": args.seed, "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "contrastive_weight": args.contrastive_weight,
            "head_input": args.head_input,
            "warmup_epochs": args.warmup_epochs,
            "min_learning_rate": args.min_learning_rate,
            "translate_pixels": args.translate_pixels,
            "patience": args.patience,
            "min_improvement": args.min_improvement,
            "initial_weights": str(args.initial_weights) if args.initial_weights else None,
            "results": results,
            "summary": summary,
        }, indent=2) + "\n")
        print(f"Fold {fold + 1} prototype top1={metrics['prototype_top1']:.4f}, "
              f"top5={metrics['prototype_top5']:.4f}, "
              f"linear top1={metrics['linear_head_top1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
