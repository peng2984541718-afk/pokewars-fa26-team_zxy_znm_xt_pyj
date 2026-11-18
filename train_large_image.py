"""GPU image-only experiments for Pokewars; no Kaggle submission code.

Run one stratified validation fold per process. Augmentations are sampled on
the fly, so no transformed validation image enters training.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from image_experiment import NUM_CLASSES, SEED, labeled_images, stratified_folds


class WideBasic(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, drop: float):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.drop = nn.Dropout2d(drop) if drop else nn.Identity()
        self.shortcut = (nn.Identity() if in_channels == out_channels and stride == 1
                         else nn.Conv2d(in_channels, out_channels, 1, stride, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(self.drop(F.relu(self.bn2(out))))
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    """WRN with a one-channel input for native 32x32 grayscale images."""

    def __init__(self, width: int = 10, drop: float = 0.2, depth: int = 28):
        super().__init__()
        if (depth - 4) % 6:
            raise ValueError("WideResNet depth must equal 6n + 4")
        channels = [16, 16 * width, 32 * width, 64 * width]
        self.stem = nn.Conv2d(1, channels[0], 3, padding=1, bias=False)
        blocks = (depth - 4) // 6
        self.stages = nn.Sequential(*[
            self._stage(channels[i], channels[i + 1], blocks,
                        1 if i == 0 else 2, drop)
            for i in range(3)
        ])
        self.bn = nn.BatchNorm2d(channels[-1])
        self.head = nn.Linear(channels[-1], NUM_CLASSES)

    @staticmethod
    def _stage(in_channels: int, out_channels: int, blocks: int,
               stride: int, drop: float) -> nn.Sequential:
        return nn.Sequential(WideBasic(in_channels, out_channels, stride, drop),
                             *[WideBasic(out_channels, out_channels, 1, drop)
                               for _ in range(blocks - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stem(x))
        x = F.relu(self.bn(x))
        return self.head(x.mean(dim=(-2, -1)))


def make_model(name: str, pretrained: bool) -> nn.Module:
    if name in ("wrn28_10", "wrn40_10"):
        if pretrained:
            raise ValueError("WRN experiment is trained from scratch")
        return WideResNet(depth=28 if name == "wrn28_10" else 40)
    if name == "convnextv2_nano":
        import timm
        # The input is expanded to at least 64x64 below. This retains spatial
        # features at the final stride-32 stage.
        return timm.create_model("convnextv2_nano.fcmae_ft_in1k",
                                 pretrained=pretrained, in_chans=1,
                                 num_classes=NUM_CLASSES, drop_path_rate=0.1)
    if name == "densenet169_cifar":
        if pretrained:
            raise ValueError("CIFAR-stem DenseNet experiment is trained from scratch")
        import timm
        model = timm.create_model("densenet169", pretrained=False, in_chans=1,
                                  num_classes=NUM_CLASSES)
        model.features.conv0 = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
        model.features.pool0 = nn.Identity()
        return model
    raise ValueError(name)


def preprocess(raw: torch.Tensor) -> torch.Tensor:
    x = raw.to(torch.float32).unsqueeze(1) / 255.0
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True).clamp_min(0.08)
    return (x - mean) / std


def augment(x: torch.Tensor, translation: float, degrees: float,
            scale_min: float, scale_max: float) -> torch.Tensor:
    """Independent translation/rotation/scale per image, with mean-gray padding."""
    n = len(x)
    angle = (torch.rand(n, device=x.device) * 2 - 1) * math.radians(degrees)
    scale = torch.empty(n, device=x.device).uniform_(scale_min, scale_max)
    cosine = torch.cos(angle) / scale
    sine = torch.sin(angle) / scale
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    theta[:, :, 2] = (torch.rand(n, 2, device=x.device) * 2 - 1) * \
        (2 * translation / x.shape[-1])
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def lr_at_step(step: int, steps: int, warmup: int,
               peak: float, minimum: float) -> float:
    if step < warmup:
        return peak * (0.1 + 0.9 * (step + 1) / max(warmup, 1))
    progress = min((step - warmup) / max(steps - warmup, 1), 1)
    return minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2


@torch.inference_mode()
def evaluate(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
             indices: np.ndarray, model_name: str, batch_size: int,
             convnext_size: int) -> dict:
    model.eval()
    correct1 = correct5 = 0
    loss = 0.0
    for offset in range(0, len(indices), batch_size):
        index = torch.as_tensor(indices[offset:offset + batch_size], device=images.device)
        x = preprocess(images[index])
        if model_name == "convnextv2_nano":
            x = F.interpolate(x, size=convnext_size, mode="bilinear", align_corners=False)
        target = labels[index]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        logits = logits.float()
        correct1 += int((logits.argmax(1) == target).sum().item())
        correct5 += int((logits.topk(5, dim=1).indices == target[:, None]).any(1).sum().item())
        loss += float(F.cross_entropy(logits, target, reduction="sum").item())
    return {"top1": correct1 / len(indices), "top5": correct5 / len(indices),
            "cross_entropy": loss / len(indices)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("wrn28_10", "wrn40_10", "convnextv2_nano",
                                            "densenet169_cifar"), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--translate-pixels", type=float, default=1.5)
    parser.add_argument("--rotate-degrees", type=float, default=12)
    parser.add_argument("--scale-min", type=float, default=0.9)
    parser.add_argument("--scale-max", type=float, default=1.1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=0,
                        help="Do not early-stop before this many continuation epochs")
    parser.add_argument("--min-improvement", type=float, default=0.0005)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--convnext-size", type=int, default=64)
    parser.add_argument("--initial-weights", type=Path, default=None,
                        help="Continue from an existing best checkpoint with a new LR schedule")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--training-seed", type=int, default=None,
                        help="Initialization and batch-order seed; keeps the validation split fixed")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--external-bank", type=Path, default=None,
                        help="Fold-safe mapped sprite bank produced by build_external_sprite_bank.py")
    parser.add_argument("--external-fraction", type=float, default=0.0,
                        help="Fraction of each training batch drawn from the external bank")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    if (args.warmup_epochs >= args.epochs or args.min_epochs > args.epochs or
            args.scale_min <= 0 or args.scale_max < args.scale_min):
        parser.error("Invalid schedule or scale range")
    if args.convnext_size < 64 or args.convnext_size % 32:
        parser.error("--convnext-size must be a multiple of 32 and at least 64")
    if not 0 <= args.external_fraction < 1:
        parser.error("--external-fraction must be in [0, 1)")
    if args.external_fraction and args.external_bank is None:
        parser.error("--external-bank is required for a positive --external-fraction")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required")
    training_seed = args.seed if args.training_seed is None else args.training_seed
    torch.manual_seed(training_seed + args.fold)
    np.random.seed(training_seed + args.fold)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"{args.model}_fold{args.fold + 1}"
    label_file = args.data_dir / "labeled_ids.npy"
    if label_file.exists():
        labels_np = np.load(label_file).astype(np.int64)
    else:
        with ZipFile(args.data_dir / "competition.zip") as archive:
            _, labels_np = labeled_images(archive)
    images_np = np.load(args.data_dir / "labeled_32x32.npy", mmap_mode="r")
    if len(images_np) != len(labels_np):
        raise ValueError("Image cache does not match label count")
    folds = stratified_folds(labels_np, 3, args.seed)
    train_idx = np.flatnonzero(folds != args.fold)
    valid_idx = np.flatnonzero(folds == args.fold)
    images = torch.as_tensor(np.asarray(images_np).copy(), device="cuda")
    labels = torch.as_tensor(labels_np, dtype=torch.long, device="cuda")
    external_images = external_labels = None
    if args.external_fraction:
        with np.load(args.external_bank) as bank:
            external_images_np = bank["images"]
            external_labels_np = bank["labels"].astype(np.int64)
        if (external_images_np.ndim != 3 or external_images_np.shape[1:] != (32, 32)
                or len(external_images_np) != len(external_labels_np)
                or external_labels_np.min() < 0
                or external_labels_np.max() >= NUM_CLASSES):
            raise ValueError("Invalid external bank image shape or labels")
        external_images = torch.as_tensor(external_images_np, device="cuda")
        external_labels = torch.as_tensor(external_labels_np, device="cuda")
    model = make_model(args.model, args.pretrained).cuda()
    if args.initial_weights is not None:
        model.load_state_dict(torch.load(args.initial_weights, map_location="cpu",
                                         weights_only=True))
    peak_lr = args.learning_rate or (2e-4 if args.pretrained else 8e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                                  weight_decay=args.weight_decay, fused=True)
    steps_per_epoch = math.ceil(len(train_idx) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    rng = np.random.default_rng(training_seed + args.fold)
    history: list[dict] = []
    best = {"top1": -1.0, "epoch": 0}
    start = time.monotonic()
    config = vars(args).copy()
    config.update({"data_dir": str(args.data_dir), "output_dir": str(args.output_dir),
                   "initial_weights": str(args.initial_weights) if args.initial_weights else None,
                   "run_name": run_name, "effective_learning_rate": peak_lr,
                   "train_images": len(train_idx), "validation_images": len(valid_idx),
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "external_images": len(external_images) if external_images is not None else 0,
                   "external_bank": str(args.external_bank) if args.external_bank else None})
    (args.output_dir / f"{run_name}_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"{run_name}: {len(train_idx)} train, {len(valid_idx)} validation, "
          f"{config['parameters']:,} parameters, GPU={torch.cuda.get_device_name()}", flush=True)
    if args.initial_weights is not None:
        initial_metrics = evaluate(model, images, labels, valid_idx, args.model,
                                   args.eval_batch_size, args.convnext_size)
        best = {"top1": initial_metrics["top1"], "top5": initial_metrics["top5"],
                "epoch": 0, "cross_entropy": initial_metrics["cross_entropy"]}
        torch.save(model.state_dict(), args.output_dir / f"{run_name}_best.pt")
        (args.output_dir / f"{run_name}_best.json").write_text(json.dumps(best, indent=2) + "\n")
        history.append({"epoch": 0, "learning_rate": None, "training_loss": None,
                        **initial_metrics, "elapsed_seconds": round(time.monotonic() - start, 1)})
        print(f"initial checkpoint val_top1={initial_metrics['top1']:.4f} "
              f"top5={initial_metrics['top5']:.4f}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        train_loss = 0.0
        for batch_id, offset in enumerate(range(0, len(order), args.batch_size)):
            index = torch.as_tensor(order[offset:offset + args.batch_size], device="cuda")
            raw = images[index]
            target = labels[index]
            if external_images is not None:
                n_external = round(len(index) * args.external_fraction)
                if n_external:
                    external_index = torch.as_tensor(
                        rng.integers(0, len(external_images), size=n_external), device="cuda")
                    raw = torch.cat((raw[:-n_external], external_images[external_index]))
                    target = torch.cat((target[:-n_external], external_labels[external_index]))
            x = preprocess(raw)
            x = augment(x, args.translate_pixels, args.rotate_degrees,
                        args.scale_min, args.scale_max)
            if args.model == "convnextv2_nano":
                x = F.interpolate(x, size=args.convnext_size, mode="bilinear", align_corners=False)
            global_step = (epoch - 1) * steps_per_epoch + batch_id
            lr = lr_at_step(global_step, total_steps, warmup_steps,
                            peak_lr, args.min_learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.float(), target,
                                       label_smoothing=args.label_smoothing)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.detach().item()) * len(index)
        metrics = evaluate(model, images, labels, valid_idx, args.model,
                           args.eval_batch_size, args.convnext_size)
        record = {"epoch": epoch, "learning_rate": lr,
                  "training_loss": train_loss / len(train_idx), **metrics,
                  "elapsed_seconds": round(time.monotonic() - start, 1)}
        history.append(record)
        (args.output_dir / f"{run_name}_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"epoch {epoch:02d} lr={lr:.2e} loss={record['training_loss']:.3f} "
              f"val_top1={metrics['top1']:.4f} top5={metrics['top5']:.4f} "
              f"elapsed={record['elapsed_seconds']}s", flush=True)
        if metrics["top1"] > best["top1"] + args.min_improvement:
            best = {"top1": metrics["top1"], "top5": metrics["top5"],
                    "epoch": epoch, "cross_entropy": metrics["cross_entropy"]}
            torch.save(model.state_dict(), args.output_dir / f"{run_name}_best.pt")
            (args.output_dir / f"{run_name}_best.json").write_text(json.dumps(best, indent=2) + "\n")
        if epoch >= args.min_epochs and epoch - best["epoch"] >= args.patience:
            print(f"early stop: best epoch {best['epoch']}", flush=True)
            break
    print("BEST", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
