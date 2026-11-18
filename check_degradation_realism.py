"""Compare synthetic noisy images with real noisy images.

Each synthetic image takes the estimated parameters of one real noisy image, so
both sets have the same noise-level mix. Reports a parameter round trip, the
noise-estimate distribution, and the held-out AUC of a small CNN trained to
tell real from synthetic (0.5 = indistinguishable).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import median_filter
from torch import nn
from torch.nn import functional as F

from degradation import (PARAM_NAMES, estimate_params, laplacian_noise, synthesize,
                         transform_silhouettes)
from train_large_image import preprocess


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def residual_stats(images: np.ndarray) -> dict:
    """Spatial correlation and foreground/background spread of the high-pass residual."""
    x = images.astype(np.float32)
    smooth = median_filter(x, size=(1, 5, 5), mode="nearest")
    r = np.where((images == 25) | (images == 245) | (images == 0) | (images == 255), np.nan, x - smooth)
    r = r - np.nanmean(r, axis=(1, 2), keepdims=True)
    var = np.nanmean(r * r)
    corr = lambda a, b: float(np.nanmean(a * b) / var)
    t = smooth.reshape(len(x), -1)
    cut = ((t.max(1) + t.min(1)) / 2)[:, None, None]
    return {"lag1_corr": (corr(r[:, :, 1:], r[:, :, :-1]) + corr(r[:, 1:], r[:, :-1])) / 2,
            "lag2_corr": (corr(r[:, :, 2:], r[:, :, :-2]) + corr(r[:, 2:], r[:, :-2])) / 2,
            "foreground_sd": float(np.sqrt(np.nanmean(np.where(smooth < cut, r, np.nan) ** 2))),
            "background_sd": float(np.sqrt(np.nanmean(np.where(smooth >= cut, r, np.nan) ** 2)))}


def shape_stats(images: np.ndarray) -> dict:
    """Area, centroid spread and second-moment extent of a denoised silhouette mask."""
    m = median_filter(images.astype(np.float32), size=(1, 5, 5), mode="nearest")
    f = m.reshape(len(m), -1)
    s = m < ((np.percentile(f, 95, 1) + np.percentile(f, 5, 1)) / 2)[:, None, None]
    yy, xx = np.mgrid[:32, :32]
    k = np.maximum(s.sum((1, 2)), 1)
    cy, cx = (s * yy).sum((1, 2)) / k, (s * xx).sum((1, 2)) / k
    extent = np.sqrt(((s * (yy - cy[:, None, None]) ** 2).sum((1, 2))
                      + (s * (xx - cx[:, None, None]) ** 2).sum((1, 2))) / k)
    return {"area": float(s.mean()), "centroid_sd": float((cy.std() + cx.std()) / 2),
            "extent": float(extent.mean())}


def discriminator() -> nn.Module:
    block = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
    return nn.Sequential(block(1, 32), block(32, 32), nn.MaxPool2d(2), block(32, 64),
                         block(64, 64), nn.MaxPool2d(2), block(64, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("outputs/large/degradation_f0.npz"))
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--count", type=int, default=40000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--scale", type=float, nargs=2, default=[1.0, 1.0])
    parser.add_argument("--shift-sd", type=float, default=0.0)
    parser.add_argument("--degrees", type=float, default=0.0)
    args = parser.parse_args()
    output = args.output or args.data.with_name(args.data.stem + "_realism.json")
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    with np.load(args.data) as z:
        s = torch.as_tensor(z["silhouettes"], device="cuda")
        residual = torch.as_tensor(z["residual"], device="cuda")
        pool, pool_index = z["params"], z["params_index"]
    images = np.load(args.data_dir / "labeled_32x32.npy", mmap_mode="r")
    pick = np.sort(rng.choice(len(pool), size=min(args.count, len(pool)), replace=False))
    real = np.asarray(images[pool_index[pick]])
    target = pool[pick]
    src = torch.as_tensor(rng.integers(0, len(s), len(real)), device="cuda")
    maps = s[src]
    if args.shift_sd or args.degrees or args.scale != [1.0, 1.0]:
        maps = transform_silhouettes(maps, tuple(args.scale), args.shift_sd, args.degrees)
    fake = synthesize(maps, residual[src], torch.as_tensor(target, device="cuda")).cpu().numpy()

    est_real, est_fake = estimate_params(real), estimate_params(fake)
    quant = lambda v: np.quantile(v, [.1, .5, .9]).round(4).tolist()
    round_trip = {name: {"real_q10_50_90": quant(est_real[:, i]),
                         "synthetic_q10_50_90": quant(est_fake[:, i])}
                  for i, name in enumerate(PARAM_NAMES)}
    round_trip["laplacian_noise"] = {"real_q10_50_90": quant(laplacian_noise(real)),
                                     "synthetic_q10_50_90": quant(laplacian_noise(fake))}
    round_trip["residual_structure"] = {"real": residual_stats(real),
                                        "synthetic": residual_stats(fake)}
    round_trip["shape"] = {"real": shape_stats(real), "synthetic": shape_stats(fake)}
    print(json.dumps(round_trip["shape"]), flush=True)
    hist = lambda v: (np.histogram(v, bins=16, range=(0, 256))[0] / v.size).round(4).tolist()
    histograms = {"real": hist(real), "synthetic": hist(fake)}

    x = torch.as_tensor(np.concatenate([real, fake]), device="cuda")
    y = torch.as_tensor(np.r_[np.ones(len(real)), np.zeros(len(fake))], device="cuda").float()
    order = torch.as_tensor(rng.permutation(len(x)), device="cuda")
    cut = int(.75 * len(x))
    train_idx, test_idx = order[:cut], order[cut:]
    model = discriminator().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), device="cuda")]
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = F.binary_cross_entropy_with_logits(model(preprocess(x[b])).squeeze(1), y[b])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            scores = torch.cat([model(preprocess(x[test_idx[i:i + 1024]])).squeeze(1)
                                for i in range(0, len(test_idx), 1024)]).cpu().numpy()
        history.append(auc(scores, y[test_idx].cpu().numpy()))
        print(f"epoch {epoch + 1}: held-out AUC={history[-1]:.4f}", flush=True)

    rows = []
    for r in range(4):
        rows.append(np.concatenate(list(real[r * 16:(r + 1) * 16]), 1))
        rows.append(np.concatenate(list(fake[r * 16:(r + 1) * 16]), 1))
    montage = np.concatenate(rows, 0)
    Image.fromarray(montage).resize((montage.shape[1] * 3, montage.shape[0] * 3),
                                    Image.NEAREST).save(output.with_suffix(".png"))
    report = {"pairs": len(real), "geometry": {"scale": args.scale, "shift_sd": args.shift_sd,
                                               "degrees": args.degrees},
              "discriminator_auc_by_epoch": history,
              "round_trip": round_trip, "pixel_histograms": histograms,
              "montage": str(output.with_suffix(".png")) + " (odd rows real, even rows synthetic)"}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"final AUC={history[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
