"""Parametric degradation model for the noisy 32x32 sprites.

An observed image is a dark silhouette on a light background whose contrast is
reduced, plus additive Gaussian noise clipped to [0, 255], followed by impulse
noise that sets pixels to exactly 25 or 245. Per-image parameters are
estimated from real images; clean images are reduced to a silhouette map and
re-degraded with parameters drawn jointly from real noisy images.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import median_filter

PARAM_NAMES = ("background", "foreground", "sigma", "impulse_low", "impulse_high")
IMPULSE_LOW, IMPULSE_HIGH = 25, 245


def _impulse_mask(flat: np.ndarray) -> np.ndarray:
    return (flat == IMPULSE_LOW) | (flat == IMPULSE_HIGH) | (flat <= 0) | (flat >= 255)


def estimate_params(images: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """Return an (n, 5) float32 array of PARAM_NAMES for uint8 (n, 32, 32) images."""
    out = np.empty((len(images), 5), dtype=np.float32)
    for start in range(0, len(images), chunk):
        x = np.asarray(images[start:start + chunk], dtype=np.float32)
        flat = x.reshape(len(x), -1)
        v = np.where(_impulse_mask(flat), np.nan, flat)
        t = np.nanmedian(v, axis=1, keepdims=True)
        for _ in range(10):
            lo = np.nanmean(np.where(v < t, v, np.nan), axis=1, keepdims=True)
            hi = np.nanmean(np.where(v >= t, v, np.nan), axis=1, keepdims=True)
            t = (lo + hi) / 2
        # Neighbour differences are dominated by noise; edges are a small minority.
        img = v.reshape(x.shape)
        d = np.concatenate([(img[:, :, 1:] - img[:, :, :-1]).reshape(len(x), -1),
                            (img[:, 1:, :] - img[:, :-1, :]).reshape(len(x), -1)], axis=1)
        sigma = 1.4826 * np.nanmedian(np.abs(d), axis=1) / np.sqrt(2)
        # Impulse rate = spike at the exact value minus the continuous density there.
        excess = lambda c: np.clip((flat == c).mean(1) - ((flat == c - 1).mean(1)
                                                          + (flat == c + 1).mean(1)) / 2, 0, 1)
        out[start:start + len(x)] = np.stack(
            [hi[:, 0], lo[:, 0], sigma, excess(IMPULSE_LOW), excess(IMPULSE_HIGH)], axis=1)
    return out


def silhouettes(images: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map clean uint8 images to silhouette strength s (about 0 = background, 1 = foreground).

    Impulse pixels are replaced by their 3x3 median. Also returns the residual
    noise of each map, sigma / (background - foreground), so synthesis adds
    only the noise that is still missing.
    """
    x = np.asarray(images, dtype=np.float32)
    med = median_filter(x, size=(1, 3, 3), mode="nearest")
    x = np.where(_impulse_mask(x), med, x)
    gap = np.maximum(params[:, 0] - params[:, 1], 20.0)
    s = np.clip((params[:, 0, None, None] - x) / gap[:, None, None], -0.25, 1.25)
    return s.astype(np.float16), (params[:, 2] / gap).astype(np.float32)


def transform_silhouettes(s: torch.Tensor, scale: tuple[float, float], shift_sd: float,
                          degrees: float, flip: bool = True) -> torch.Tensor:
    """Random zoom, translation, rotation and horizontal flip; background padding.

    Noisier real images are larger and more displaced than clean ones, so clean
    sources need this extra geometry before they are degraded.
    """
    n = len(s)
    zoom = torch.empty(n, device=s.device).uniform_(*scale)
    angle = (torch.rand(n, device=s.device) * 2 - 1) * np.deg2rad(degrees)
    sign = torch.where(torch.rand(n, device=s.device) < .5, -1.0, 1.0) if flip else 1.0
    theta = torch.zeros(n, 2, 3, device=s.device)
    theta[:, 0, 0] = torch.cos(angle) / zoom * sign
    theta[:, 0, 1] = -torch.sin(angle) / zoom
    theta[:, 1, 0] = torch.sin(angle) / zoom * sign
    theta[:, 1, 1] = torch.cos(angle) / zoom
    theta[:, :, 2] = torch.randn(n, 2, device=s.device) * shift_sd * 2 / s.shape[-1]
    x = s.float().unsqueeze(1)
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    return torch.nn.functional.grid_sample(x, grid, align_corners=False,
                                           padding_mode="zeros").squeeze(1)


def synthesize(s: torch.Tensor, residual: torch.Tensor, params: torch.Tensor,
               generator: torch.Generator | None = None) -> torch.Tensor:
    """Degrade silhouette maps s (n, 32, 32) with target params (n, 5); returns uint8."""
    s = s.float()
    bg, fg, sigma, low, high = params.float().unbind(1)
    gap = (bg - fg).clamp_min(5.0)
    present = residual.float() * gap
    extra = (sigma.square() - present.square()).clamp_min(0).sqrt()
    noise = torch.randn(s.shape, device=s.device, generator=generator)
    x = (bg[:, None, None] - gap[:, None, None] * s
         + extra[:, None, None] * noise).clamp(0, 255).round()
    u = torch.rand(s.shape, device=s.device, generator=generator)
    x = torch.where(u < low[:, None, None], torch.full_like(x, IMPULSE_LOW), x)
    x = torch.where(u > 1 - high[:, None, None], torch.full_like(x, IMPULSE_HIGH), x)
    return x.to(torch.uint8)


def clean_impulses_gpu(images: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Replace exact 25/245 impulse pixels with their 3x3 non-impulse median on GPU.

    Uses replicate padding at borders and falls back to the full 3x3 median if all
    neighbors are impulses. Operates strictly per-image with no dataset statistics.
    """
    out = images.clone()
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        x = images[start:end].to(torch.float32)
        mask = (x == IMPULSE_LOW) | (x == IMPULSE_HIGH)
        if not mask.any():
            continue
        padded = torch.nn.functional.pad(x.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
        patches = padded.unfold(1, 3, 1).unfold(2, 3, 1).reshape(-1, 32, 32, 9)
        full_med = patches.median(dim=-1).values
        # Exclude impulse neighbors via nanmedian so adjacent impulses do not corrupt each other
        imp_p = (patches == IMPULSE_LOW) | (patches == IMPULSE_HIGH)
        clean_p = torch.where(imp_p, torch.full_like(patches, float("nan")), patches)
        valid_med = torch.nanmedian(clean_p, dim=-1).values
        repaired = torch.where(torch.isnan(valid_med), full_med, valid_med)
        out[start:end] = torch.where(mask, repaired, x).round().clamp(0, 255).to(images.dtype)
    return out


def laplacian_noise(images: np.ndarray) -> np.ndarray:
    """Immerkaer noise estimate used for the noise bins in the evaluation."""
    from scipy.ndimage import convolve
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    x = np.asarray(images, dtype=np.float32)
    lap = np.abs(convolve(x, k[None], mode="reflect"))[:, 1:-1, 1:-1]
    return np.sqrt(np.pi / 2) / (6 * 30 * 30) * lap.sum((1, 2))
