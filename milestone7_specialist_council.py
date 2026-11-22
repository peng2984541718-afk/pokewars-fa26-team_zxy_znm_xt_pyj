"""Milestone 7: Orthogonal Specialist Council & Physical-Regime MoE Fusion.

Motivated by our Fold-0 forensic error stratification after Milestone 6B (80.1295):
  - Clean images (low 20% FFT high-freq): error rate is ALREADY 0.42% (99.58% acc).
  - High-freq / impulse grain (top 20%): error rate jumps 16.8x -> 6.72%!
  - Border-cropped / rotated / zoomed (top 20%): error rate jumps 11.4x -> 5.49%!
  - 96.4% of ALL remaining errors are ALREADY inside Top-20 (43.1% are #1 vs #2)!

We build THREE genuinely orthogonal specialists to form a Specialist Council:
  7A. `CentroidLogPolarNet`:
      Foreground-centroid-aligned (theta, log r) Polar CNN with circular angular
      convolutions + Siamese exact D2 flip invariance + No-GAP polar grid head.
      Mathematically converts 2D rotations -> angular shifts and zoom -> radial shifts!
  7B. `SpectralMultiBandSilhouetteNet`:
      6-channel grain-immune physical stem (3x3 median, 5x5 median, G_1.0, G_2.2,
      DoG contour band-pass, and exact 2D FFT radial low-pass r<=7.5) + No-GAP
      exact D2 bilateral spatial head, gated dynamically by per-image FFT noise!
  7C. `ForensicTemplateDiffReranker`:
      Canonical denoised class templates {mu_c, std_c} evaluated across multi-angle
      & mirror alignments ONLY on the discriminative pixels where an image's Top-20
      candidates actually differ (`Var_{c in Top20}[mu_c(u, v)]` forensic mask)!
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from degradation import clean_impulses_gpu
from evaluate_soft_score import DENOMINATOR, rmse
from export_probs_rot5 import score_model
from image_experiment import NUM_CLASSES, SEED, stratified_folds
from joint_constrained_inference import target_cavity_predict
from milestone1_rx_fusion_and_pack import gather_cand
from milestone3_inference_calibration import (
    adaptive_power_scale,
    run_unlabeled_bp_hetero,
    z_lap_from_gallery,
)
from milestone6b_twin_spatial_specialist import (
    compute_train_twin_weights,
    lr_at_step,
)
from mix_distill_posteriors import fuse_posteriors
from train_large_image import augment, preprocess
from train_large_image_external import soft_cross_entropy

L = Path("outputs/large")


def score_fn(val: float) -> float:
    return 100.0 * (1.0 - val / DENOMINATOR)


# =============================================================================
# Specialist 7A: Centroid Log-Polar Rotation & Scale Equivariant Network
# =============================================================================

class CircularThetaResBlock(nn.Module):
    """2D residual block on (Theta, LogR) where Theta (height, H) wraps 2*pi
    periodically via exact circular padding and LogR (width, W) uses border/zero."""
    def __init__(self, in_ch: int, out_ch: int, stride_theta: int = 1, stride_r: int = 1):
        super().__init__()
        self.stride_t = stride_theta
        self.stride_r = stride_r
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=(stride_theta, stride_r), padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=0, bias=False)
        self.short = (
            nn.Conv2d(in_ch, out_ch, 1, stride=(stride_theta, stride_r), bias=False)
            if (in_ch != out_ch or stride_theta != 1 or stride_r != 1)
            else nn.Identity()
        )

    @staticmethod
    def _pad_polar(x: torch.Tensor) -> torch.Tensor:
        # H is Theta (periodic 2*pi -> circular pad 1), W is LogR (replicate pad 1)
        x = torch.cat([x[:, :, -1:, :], x, x[:, :, :1, :]], dim=2)
        x = F.pad(x, (1, 1, 0, 0), mode="replicate")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.bn1(x))
        s = self.short(h if not isinstance(self.short, nn.Identity) else x)
        h = self.conv1(self._pad_polar(h))
        h = self.conv2(self._pad_polar(F.silu(self.bn2(h))))
        return h + s


class CentroidLogPolarNet(nn.Module):
    """Warps 32x32 Cartesian sprites around their soft foreground centroid into a
    48 x 32 (theta, log r) polar manifold, processes with periodic-theta convs,
    and pools with an exact D2 flip-invariant No-GAP polar + angular harmonic head."""
    def __init__(self, n_theta: int = 48, n_rad: int = 32, widths=(80, 160, 288)):
        super().__init__()
        self.n_theta = n_theta
        self.n_rad = n_rad

        # Precompute base polar unit vectors (n_theta, n_rad)
        theta = torch.linspace(-math.pi, math.pi * (1.0 - 2.0 / n_theta), n_theta)
        log_r = torch.linspace(math.log(0.055), math.log(1.12), n_rad)
        r = torch.exp(log_r)
        cos_t = torch.cos(theta).view(1, n_theta, 1)
        sin_t = torch.sin(theta).view(1, n_theta, 1)
        rr = r.view(1, 1, n_rad)
        self.register_buffer("dx", cos_t * rr, persistent=False)   # (1, T, R)
        self.register_buffer("dy", sin_t * rr, persistent=False)   # (1, T, R)
        self.register_buffer("r_chan", ((log_r - log_r.mean()) / log_r.std()).view(1, 1, 1, n_rad).expand(1, 1, n_theta, n_rad), persistent=False)

        ys = torch.linspace(-1.0, 1.0, 32).view(1, 1, 32, 1)
        xs = torch.linspace(-1.0, 1.0, 32).view(1, 1, 1, 32)
        self.register_buffer("grid_y", ys, persistent=False)
        self.register_buffer("grid_x", xs, persistent=False)

        # Input: 1 normalized luminance channel + 1 radial coordinate channel = 2
        self.stem = nn.Conv2d(2, widths[0], 3, padding=0, bias=False)
        self.stage1 = nn.Sequential(*[CircularThetaResBlock(widths[0], widths[0]) for _ in range(3)])
        # 48x32 -> 24x16
        self.stage2 = nn.Sequential(
            CircularThetaResBlock(widths[0], widths[1], stride_theta=2, stride_r=2),
            *[CircularThetaResBlock(widths[1], widths[1]) for _ in range(4)],
        )
        # 24x16 -> 12x8
        self.stage3 = nn.Sequential(
            CircularThetaResBlock(widths[1], widths[2], stride_theta=2, stride_r=2),
            *[CircularThetaResBlock(widths[2], widths[2]) for _ in range(4)],
        )
        self.norm = nn.BatchNorm2d(widths[2])

        # Radial profile head (rotation-invariant mean & std over theta across 8 radial rings!)
        # + Full 12x8 polar grid projection for asymmetric angular structure!
        self.polar_proj = nn.Sequential(
            nn.Conv2d(widths[2], 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        # Features:
        #  - radial ring mean (288 * 8 = 2304) [invariant to ANY rotation!]
        #  - radial ring std  (288 * 8 = 2304) [angular contrast per ring, rotation-invariant!]
        #  - symmetric + |asymmetric| 12x8 polar cells (2 * 32 * 12 * 8 = 6144)
        # Wait: keep compact bottleneck so generalization matches SpatialSpecialist!
        self.ring_proj = nn.Sequential(
            nn.Conv1d(widths[2] * 2, 192, 1, bias=False),
            nn.BatchNorm1d(192),
            nn.SiLU(inplace=True),
        )
        feat_dim = 192 * 8 + 2 * (32 * 12 * 8) // 4  # pool polar grid to 6x8 = 48 cells -> 2*32*48 = 3072 + 1536 = 4608
        self.classifier = nn.Sequential(
            nn.LayerNorm(4608),
            nn.Dropout(0.25),
            nn.Linear(4608, 1536, bias=False),
            nn.BatchNorm1d(1536),
            nn.SiLU(inplace=True),
            nn.Dropout(0.18),
            nn.Linear(1536, NUM_CLASSES),
        )

    def _warp_polar(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        # Estimate foreground energy map smoothly (clamped to [-0.22, 0.22] for stability)
        lum = x.mean(dim=1, keepdim=True)
        bg = torch.cat([lum[:, :, 0, :], lum[:, :, -1, :], lum[:, :, :, 0], lum[:, :, :, -1]], dim=-1).median(dim=-1).values.view(b, 1, 1, 1)
        sal = (lum - bg).abs().pow(1.5) + 1e-3
        sal = sal / sal.sum(dim=(2, 3), keepdim=True)
        cx = (sal * self.grid_x).sum(dim=(2, 3)).clamp(-0.22, 0.22)  # (B, 1)
        cy = (sal * self.grid_y).sum(dim=(2, 3)).clamp(-0.22, 0.22)  # (B, 1)

        gx = cx.view(b, 1, 1) + self.dx
        gy = cy.view(b, 1, 1) + self.dy
        grid = torch.stack([gx, gy], dim=-1)  # (B, T, R, 2)
        polar = F.grid_sample(lum, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return torch.cat([polar, self.r_chan.expand(b, -1, -1, -1).to(lum.dtype)], dim=1)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = self._warp_polar(x)
        h = self.stem(CircularThetaResBlock._pad_polar(p))
        h = self.stage1(h)
        h = self.stage2(h)
        h = F.silu(self.norm(self.stage3(h)))  # (B, C, 12, 8)
        # Rotation-invariant radial ring descriptors along R=8 (pooling over Theta=12)
        r_mean = h.mean(dim=2)
        r_std = (h.var(dim=2, unbiased=False) + 1e-5).sqrt()
        rings = self.ring_proj(torch.cat([r_mean, r_std], dim=1)).flatten(1)  # (B, 1536)
        # 6x8 coarse angular-radial map
        grid6x8 = F.adaptive_avg_pool2d(self.polar_proj(h), (6, 8)).flatten(1)  # (B, 1536)
        return rings, grid6x8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ra, ga = self._encode(x)
        rb, gb = self._encode(torch.flip(x, dims=[-1]))
        r_sym = 0.5 * (ra + rb)
        g_sym = 0.5 * (ga + gb)
        g_asym = (ga - gb).abs()
        return self.classifier(torch.cat([r_sym, g_sym, g_asym], dim=1))


# =============================================================================
# Specialist 7B: Spectral Multi-Bandwidth Grain-Immune Silhouette Network
# =============================================================================

class ConvResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.short = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False) if (in_ch != out_ch or stride != 1) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.bn1(x))
        s = self.short(h if not isinstance(self.short, nn.Identity) else x)
        h = self.conv1(h)
        h = self.conv2(F.silu(self.bn2(h)))
        return h + s


class SpectralMultiBandSilhouetteNet(nn.Module):
    """Explicit 6-band physical grain-immune decomposition on GPU before any
    learnable conv sees the image, eliminating the 16.8x high-freq grain blindspot:
      1. 3x3 exact spatial median filter (kills isolated salt/pepper impulses)
      2. 5x5 exact spatial median filter (kills 2-pixel impulse clusters)
      3. Gaussian low-pass sigma=1.0 (smooths Gaussian pixel noise)
      4. Gaussian silhouette low-pass sigma=2.2 (pure body/limb mass envelope)
      5. Difference-of-Gaussians band-pass (G_0.9 - G_2.2: clean internal contours)
      6. Exact 2D FFT radial low-pass r<=7.5 (zero high-frequency Fourier energy!)
    Plus CoordConv (y, |x|, r) + exact D2 bilateral No-GAP 8x8 + global head!
    """
    def __init__(self, widths=(96, 192, 320)):
        super().__init__()
        # Register fixed Gaussian kernels & Fourier radial mask
        self.register_buffer("g10", self._gauss_kernel(5, 1.0), persistent=False)
        self.register_buffer("g22", self._gauss_kernel(9, 2.2), persistent=False)
        self.register_buffer("g09", self._gauss_kernel(5, 0.85), persistent=False)

        fy = torch.fft.fftfreq(32).view(32, 1) * 32.0
        fx = torch.fft.rfftfreq(32).view(1, 17) * 32.0
        fr = torch.sqrt(fy * fy + fx * fx)
        # Smooth Butterworth-style radial rolloff centered at r=7.5 (avoids Gibbs ringing!)
        fft_mask = 1.0 / (1.0 + (fr / 7.5).pow(6))
        self.register_buffer("fft_mask", fft_mask.view(1, 1, 32, 17), persistent=False)

        ys = torch.linspace(-1.0, 1.0, 32).view(1, 1, 32, 1).expand(1, 1, 32, 32)
        xs = torch.linspace(-1.0, 1.0, 32).abs().view(1, 1, 1, 32).expand(1, 1, 32, 32)
        rr = torch.sqrt(ys * ys + xs * xs)
        self.register_buffer("coords", torch.cat([ys, xs, rr], dim=1), persistent=False)

        # 6 spectral/order-statistic bands + 3 coords = 9 input channels
        self.stem = nn.Conv2d(9, widths[0], 3, padding=1, bias=False)
        self.stage1 = nn.Sequential(*[ConvResBlock(widths[0], widths[0]) for _ in range(4)])
        self.stage2 = nn.Sequential(
            ConvResBlock(widths[0], widths[1], stride=2),
            *[ConvResBlock(widths[1], widths[1]) for _ in range(4)],
        )
        self.stage3 = nn.Sequential(
            ConvResBlock(widths[1], widths[2], stride=2),
            *[ConvResBlock(widths[2], widths[2]) for _ in range(4)],
        )
        self.norm = nn.BatchNorm2d(widths[2])
        self.grid_proj = nn.Sequential(
            nn.Conv2d(widths[2], 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        # 2 * (32 * 8 * 8 = 2048) + 2 * 320 = 4736
        self.classifier = nn.Sequential(
            nn.LayerNorm(4736),
            nn.Dropout(0.25),
            nn.Linear(4736, 1536, bias=False),
            nn.BatchNorm1d(1536),
            nn.SiLU(inplace=True),
            nn.Dropout(0.18),
            nn.Linear(1536, NUM_CLASSES),
        )

    @staticmethod
    def _gauss_kernel(ksize: int, sigma: float) -> torch.Tensor:
        ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        k = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        k = k / k.sum()
        return k.view(1, 1, ksize, ksize)

    @staticmethod
    def _median2d(u: torch.Tensor, k: int) -> torch.Tensor:
        pad = k // 2
        p = F.pad(u, (pad, pad, pad, pad), mode="reflect")
        uf = p.unfold(2, k, 1).unfold(3, k, 1).contiguous().view(u.shape[0], 1, 32, 32, k * k)
        return uf.median(dim=-1).values

    def _spectral_decompose(self, x: torch.Tensor) -> torch.Tensor:
        # Grayscale luminance channel (all dataset sprites are grayscale!)
        u = x.mean(dim=1, keepdim=True)
        m3 = self._median2d(u, 3)
        m5 = self._median2d(u, 5)
        g10 = F.conv2d(F.pad(m3, (2, 2, 2, 2), mode="reflect"), self.g10)
        g22 = F.conv2d(F.pad(m3, (4, 4, 4, 4), mode="reflect"), self.g22)
        g09 = F.conv2d(F.pad(m3, (2, 2, 2, 2), mode="reflect"), self.g09)
        dog = (g09 - g22) * 2.5
        # FFT radial low-pass in float32 (cuFFT requires fp32)
        u32 = m3.float()
        mu = u32.mean(dim=(2, 3), keepdim=True)
        spec = torch.fft.rfft2(u32 - mu) * self.fft_mask
        fft_lp = (torch.fft.irfft2(spec, s=(32, 32)) + mu).to(x.dtype)
        coords = self.coords.expand(x.shape[0], -1, -1, -1).to(x.dtype)
        return torch.cat([m3, m5, g10, g22, dog, fft_lp, coords], dim=1)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self._spectral_decompose(x)
        h = self.stage1(self.stem(s))
        h = self.stage2(h)
        h = F.silu(self.norm(self.stage3(h)))  # (B, 320, 8, 8)
        g = self.grid_proj(h).flatten(1)       # (B, 2048)
        ctx = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)  # (B, 640)
        return g, ctx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ga, ca = self._encode(x)
        gb, cb = self._encode(torch.flip(x, dims=[-1]))
        g_sym = 0.5 * (ga + gb)
        g_asym = (ga - gb).abs()
        c_sym = 0.5 * (ca + cb)
        return self.classifier(torch.cat([g_sym, g_asym, c_sym], dim=1))


# =============================================================================
# Specialist 7C: Forensic Candidate-Difference Template Reranker
# =============================================================================

@torch.inference_mode()
def build_forensic_templates(train_imgs_u8: torch.Tensor, train_labels_0b: np.ndarray,
                             unl_imgs_u8: torch.Tensor, unl_cand_1b: np.ndarray,
                             unl_post: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct clean canonical 32x32 templates (mean & inv_std) for all 1,025
    classes using median-denoised training images + high-confidence (>=0.96) unl."""
    device = train_imgs_u8.device
    # Normalize each image to zero-mean unit-std after 3x3 median denoising so lighting cancels!
    def norm_imgs(im_u8: torch.Tensor) -> torch.Tensor:
        u = (im_u8.float().unsqueeze(1) if im_u8.ndim == 3 else im_u8.float().mean(dim=1, keepdim=True)) / 255.0
        p = F.pad(u, (1, 1, 1, 1), mode="reflect")
        m3 = p.unfold(2, 3, 1).unfold(3, 3, 1).contiguous().view(-1, 32, 32, 9).median(dim=-1).values
        mu = m3.mean(dim=(1, 2), keepdim=True)
        sd = m3.std(dim=(1, 2), keepdim=True).clamp_min(0.04)
        return (m3 - mu) / sd

    N_tr = len(train_imgs_u8)
    sum_c = torch.zeros((NUM_CLASSES, 32, 32), dtype=torch.float32, device=device)
    sq_c = torch.zeros((NUM_CLASSES, 32, 32), dtype=torch.float32, device=device)
    cnt_c = torch.zeros(NUM_CLASSES, dtype=torch.float32, device=device)

    lab_t = torch.as_tensor(train_labels_0b, dtype=torch.long, device=device)
    for off in range(0, N_tr, 4096):
        ni = norm_imgs(train_imgs_u8[off:off + 4096])
        ni_sym = 0.5 * (ni + torch.flip(ni, dims=[-1]))  # canonical bilateral template
        lb = lab_t[off:off + 4096]
        sum_c.index_add_(0, lb, ni_sym)
        sq_c.index_add_(0, lb, ni_sym * ni_sym)
        cnt_c.index_add_(0, lb, torch.ones_like(lb, dtype=torch.float32))

    # Add high-confidence unlabeled sprites
    u_conf = unl_post.max(axis=1)
    hi = np.flatnonzero(u_conf >= 0.96)
    if len(hi) > 0:
        u_cls = torch.as_tensor(unl_cand_1b[hi, unl_post[hi].argmax(axis=1)] - 1, dtype=torch.long, device=device)
        u_w = torch.as_tensor(u_conf[hi], dtype=torch.float32, device=device)
        for off in range(0, len(hi), 4096):
            idx_h = torch.as_tensor(hi[off:off + 4096], device=device)
            ni = norm_imgs(unl_imgs_u8[idx_h])
            ni_sym = 0.5 * (ni + torch.flip(ni, dims=[-1]))
            lb = u_cls[off:off + 4096]
            w = u_w[off:off + 4096]
            sum_c.index_add_(0, lb, ni_sym * w.view(-1, 1, 1))
            sq_c.index_add_(0, lb, (ni_sym * ni_sym) * w.view(-1, 1, 1))
            cnt_c.index_add_(0, lb, w)

    denom = cnt_c.clamp_min(1.0).view(-1, 1, 1)
    mu_c = sum_c / denom
    var_c = (sq_c / denom - mu_c * mu_c).clamp_min(0.08)
    return mu_c, var_c.rsqrt()


@torch.inference_mode()
def score_forensic_diff_top20(imgs_u8: torch.Tensor, cand_1b: np.ndarray,
                              mu_c: torch.Tensor, inv_std_c: torch.Tensor,
                              batch_size: int = 1024) -> np.ndarray:
    """For each image and its Top-20 candidates, computes multi-angle (-7, 0, +7 deg)
    bilateral template log-likelihood RESTRICTED to the forensic discriminative pixels
    where the Top-5 candidates actually differ (`Var_{c in Top5}[mu_c(y, x)]`)!"""
    device = imgs_u8.device
    N, K = cand_1b.shape
    out_logits = np.zeros((N, K), dtype=np.float32)

    # Prebuild 3 rotation matrices (-7, 0, +7 deg)
    angles = [-7.0 * math.pi / 180.0, 0.0, 7.0 * math.pi / 180.0]
    thetas = []
    for ang in angles:
        ca, sa = math.cos(ang), math.sin(ang)
        thetas.append(torch.tensor([[ca, -sa, 0.0], [sa, ca, 0.0]], dtype=torch.float32, device=device))

    for off in range(0, N, batch_size):
        end = min(N, off + batch_size)
        b = end - off
        sl = imgs_u8[off:end]
        u = (sl.float().unsqueeze(1) if sl.ndim == 3 else sl.float().mean(dim=1, keepdim=True)) / 255.0
        p = F.pad(u, (1, 1, 1, 1), mode="reflect")
        m3 = p.unfold(2, 3, 1).unfold(3, 3, 1).contiguous().view(b, 1, 32, 32, 9).median(dim=-1).values
        mu = m3.mean(dim=(2, 3), keepdim=True)
        sd = m3.std(dim=(2, 3), keepdim=True).clamp_min(0.04)
        ni = (m3 - mu) / sd  # (B, 1, 32, 32)

        # Build bilateral views across (-7, 0, +7 deg)
        views = []
        for th in thetas:
            grid = F.affine_grid(th.unsqueeze(0).expand(b, -1, -1), (b, 1, 32, 32), align_corners=False)
            rot = F.grid_sample(ni, grid, mode="bilinear", padding_mode="border", align_corners=False).squeeze(1)
            views.append(0.5 * (rot + torch.flip(rot, dims=[-1])))  # (B, 32, 32)

        c_idx = torch.as_tensor(cand_1b[off:end] - 1, dtype=torch.long, device=device)  # (B, K)
        tpl_mu = mu_c[c_idx]           # (B, K, 32, 32)
        tpl_isd = inv_std_c[c_idx]     # (B, K, 32, 32)

        # Forensic pixel attention: variance across the top-5 candidates at each (y, x)!
        diff_mask = tpl_mu[:, :5].var(dim=1, unbiased=False) + 0.05  # (B, 32, 32)
        diff_mask = diff_mask / diff_mask.mean(dim=(1, 2), keepdim=True)

        best_ll = None
        for vw in views:
            # Weighted negative squared Mahalanobis residual on discriminative pixels
            res2 = ((vw.unsqueeze(1) - tpl_mu) * tpl_isd).pow(2) * diff_mask.unsqueeze(1)
            ll = -0.5 * res2.mean(dim=(2, 3))  # (B, K)
            best_ll = ll if best_ll is None else torch.maximum(best_ll, ll)

        best_ll = best_ll - best_ll.max(dim=1, keepdim=True).values
        out_logits[off:end] = best_ll.cpu().numpy()

    return out_logits


# =============================================================================
# Physical Image Regime Descriptors (FFT High-Freq Share & Border Crop Energy)
# =============================================================================

def compute_physical_regime_zscores(imgs_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute standardized z-scores for:
      1. `z_hf`: 2D Fourier high-frequency energy fraction (r >= 8) -> 16.8x error axis!
      2. `z_border`: perimeter border contrast std -> 11.4x crop/zoom error axis!
    """
    N = len(imgs_np)
    hf_arr = np.empty(N, dtype=np.float32)
    bd_arr = np.empty(N, dtype=np.float32)
    yy, xx = np.mgrid[-16:16, -16:16]
    hi_mask = (np.sqrt(yy * yy + xx * xx) >= 8.0)

    for off in range(0, N, 8192):
        chunk = np.asarray(imgs_np[off:off + 8192], dtype=np.float32)
        if chunk.ndim == 4:
            chunk = chunk.mean(axis=1)
        lum = chunk / 255.0
        bd = np.concatenate([lum[:, 0, :], lum[:, -1, :], lum[:, :, 0], lum[:, :, -1]], axis=1).std(axis=1)
        bd_arr[off:off + len(chunk)] = bd

        centered = lum - lum.mean(axis=(1, 2), keepdims=True)
        p2 = np.abs(np.fft.fftshift(np.fft.fft2(centered), axes=(-2, -1))) ** 2
        hf_arr[off:off + len(chunk)] = p2[:, hi_mask].sum(axis=1) / (p2.sum(axis=(1, 2)) + 1e-8)

    z_hf = np.clip((hf_arr - 0.18) / 0.10, -2.0, 3.0).astype(np.float32)
    z_bd = np.clip((bd_arr - 0.14) / 0.08, -2.0, 3.0).astype(np.float32)
    return z_hf, z_bd


def fuse_council_moe(base_post: np.ndarray,
                     lp_post: np.ndarray | None,
                     spec_post: np.ndarray | None,
                     forensic_ll: np.ndarray | None,
                     z_hf: np.ndarray,
                     z_bd: np.ndarray,
                     w_lp: float, w_lp_bd: float,
                     w_sp: float, w_sp_hf: float,
                     w_for: float, temp: float = 0.85) -> np.ndarray:
    """Zero-Perturbation Confidence x Physical-Regime-Gated MoE Fusion:
      - STRICTLY ZERO (`0.000`) specialist weight when `base_post[:, 0] >= 0.95`
        (~90.5% of images, protecting the `--train-all` 80.92531+ anchor from Brier dilution!).
      - On uncertain images (`base_post[:, 0] < 0.95`):
        - LogPolar specialist fires on border/zoom/rotated sprites (`z_bd > 0.0`)
        - Spectral specialist fires on high-frequency grain sprites (`z_hf > 0.0`)
        - Forensic template tie-breaker adds pixel-difference log-likelihood
    """
    log_p = np.log(np.clip(base_post, 1e-8, 1.0)).astype(np.float32)
    unc = np.clip((0.95 - base_post[:, :1]) / 0.45, 0.0, 1.0).astype(np.float32)
    gate_hf = np.clip((z_hf[:, None] - 0.05) / 1.45, 0.0, 1.0).astype(np.float32)
    gate_bd = np.clip((z_bd[:, None] - 0.05) / 1.45, 0.0, 1.0).astype(np.float32)

    total_spec_w = np.zeros((len(base_post), 1), dtype=np.float32)
    add_term = np.zeros_like(log_p)

    if lp_post is not None and (w_lp > 0.0 or w_lp_bd > 0.0):
        wi = unc * (w_lp + w_lp_bd * gate_bd)
        add_term += wi * (np.log(np.clip(lp_post, 1e-8, 1.0)) / temp)
        total_spec_w += wi

    if spec_post is not None and (w_sp > 0.0 or w_sp_hf > 0.0):
        ws = unc * (w_sp + w_sp_hf * gate_hf)
        add_term += ws * (np.log(np.clip(spec_post, 1e-8, 1.0)) / temp)
        total_spec_w += ws

    total_spec_w = np.clip(total_spec_w, 0.0, 0.30)
    comb = (1.0 - total_spec_w) * log_p + add_term
    if forensic_ll is not None and w_for > 0.0:
        comb = comb + (unc * w_for) * forensic_ll

    comb -= comb.max(axis=1, keepdims=True)
    out = np.exp(comb)
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-8)
    return out.astype(np.float32)


# =============================================================================
# Generic Specialist 2-Stage Trainer (Stage 1 Twin-Weighted + Stage 2 Gen-2 Distill)
# =============================================================================

@torch.inference_mode()
def eval_single(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
                valid_idx: np.ndarray, batch_size: int = 256) -> float:
    model.eval()
    v_idx = torch.as_tensor(valid_idx, device=images.device)
    correct = 0
    for off in range(0, len(v_idx), batch_size):
        b = v_idx[off:off + batch_size]
        x = preprocess(images[b])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).float()
        correct += int((logits.argmax(1) == labels[b]).sum().item())
    return correct / len(valid_idx)


def train_specialist_fold(arch_tag: str, build_fn, fold: int,
                          images: torch.Tensor, labels_np: np.ndarray,
                          rot_deg: float, scale_lo: float, scale_hi: float,
                          s1_epochs: int = 24, s2_epochs: int = 10,
                          seed_base: int = 717171) -> Path:
    best_dist_pt = L / f"{arch_tag}_gen2_f{fold}_best.pt"
    if best_dist_pt.exists():
        print(f"skip existing {best_dist_pt.name}", flush=True)
        return best_dist_pt

    folds = stratified_folds(labels_np, 3, SEED)
    valid_idx = np.flatnonzero(folds == fold)
    train_idx = np.flatnonzero(folds != fold)
    labels = torch.as_tensor(labels_np, device="cuda", dtype=torch.long)
    class_weights, _ = compute_train_twin_weights(images, labels_np, train_idx)

    s1_pt = L / f"{arch_tag}_s1_f{fold}_best.pt"
    if not s1_pt.exists():
        with np.load(L / f"transductive_bank_f{fold}.npz") as bk:
            ext_imgs = clean_impulses_gpu(torch.as_tensor(bk["images"], device="cuda"))
            ext_labs = torch.as_tensor(bk["labels"].astype(np.int64), device="cuda")

        torch.manual_seed(seed_base + fold)
        rng = np.random.default_rng(seed_base + fold)
        model = build_fn().cuda()
        batch_size = 128
        peak_lr = 9.0e-4
        opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.025, fused=True)
        steps_ep = math.ceil(len(train_idx) / batch_size)
        total_steps = steps_ep * s1_epochs
        warmup_steps = steps_ep * 2
        best_top1 = -1.0
        t0 = time.monotonic()

        for epoch in range(1, s1_epochs + 1):
            model.train()
            order = rng.permutation(train_idx)
            for b_id, off in enumerate(range(0, len(order), batch_size)):
                idx = torch.as_tensor(order[off:off + batch_size], device="cuda")
                raw = images[idx]
                tgt = labels[idx]
                n_ext = round(len(raw) * 0.42)
                if n_ext > 0:
                    e_pick = torch.as_tensor(rng.integers(0, len(ext_imgs), size=n_ext), device="cuda")
                    raw = torch.cat((raw[:-n_ext], ext_imgs[e_pick]))
                    tgt = torch.cat((tgt[:-n_ext], ext_labs[e_pick]))
                x = augment(preprocess(raw), 1.2, rot_deg, scale_lo, scale_hi)
                step = (epoch - 1) * steps_ep + b_id
                lr = lr_at_step(step, total_steps, warmup_steps, peak_lr, 1e-6)
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x).float()
                per_sample = F.cross_entropy(logits, tgt, reduction="none", label_smoothing=0.02)
                sw = class_weights[tgt]
                loss = (per_sample * sw).sum() / sw.sum()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            top1 = eval_single(model, images, labels, valid_idx)
            print(f"[{arch_tag}_s1 f{fold}] ep={epoch:02d} top1={top1:.4f} ({time.monotonic()-t0:.0f}s)", flush=True)
            if top1 > best_top1:
                best_top1 = top1
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, s1_pt)
        del model, opt, ext_imgs, ext_labs
        torch.cuda.empty_cache()

    # Stage 2: Battle-BP + Gen-2 soft distillation!
    with np.load(L / "gen2_distill_labeled_oof.npz") as dz:
        t_cand = torch.as_tensor(dz["candidates"].astype(np.int64), device="cuda")
        t_probs = torch.as_tensor(dz["posteriors"].astype(np.float32), device="cuda")

    ext_pack = L / "bp_v5_unl_f0only_pack.npz"
    use_ext = ext_pack.exists() and (fold == 0)
    if use_ext:
        ext_u_imgs = clean_impulses_gpu(torch.as_tensor(np.asarray(np.load("unl/test_32x32.npy")), device="cuda"))
        with np.load(ext_pack) as ez:
            e_cand = torch.as_tensor(ez["candidates"].astype(np.int64), device="cuda")
            e_probs = torch.as_tensor(ez["posteriors"].astype(np.float32), device="cuda")

    torch.manual_seed(seed_base + 100 + fold)
    rng = np.random.default_rng(seed_base + 100 + fold)
    model = build_fn().cuda()
    model.load_state_dict(torch.load(s1_pt, map_location="cpu", weights_only=True))
    batch_size = 128
    peak_lr = 1.6e-4
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.02, fused=True)
    steps_ep = math.ceil(len(train_idx) / batch_size)
    total_steps = steps_ep * s2_epochs
    warmup_steps = steps_ep * 1
    best_top1 = -1.0
    t0 = time.monotonic()

    for epoch in range(1, s2_epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        for b_id, off in enumerate(range(0, len(order), batch_size)):
            idx = torch.as_tensor(order[off:off + batch_size], device="cuda")
            raw = images[idx]
            tgt = labels[idx]
            x = augment(preprocess(raw), 1.0, min(rot_deg, 12.0), 0.94, 1.06)
            step = (epoch - 1) * steps_ep + b_id
            lr = lr_at_step(step, total_steps, warmup_steps, peak_lr, 1e-6)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x).float()
            hard_ps = F.cross_entropy(logits, tgt, reduction="none", label_smoothing=0.01)
            sw = class_weights[tgt]
            hard_l = (hard_ps * sw).sum() / sw.sum()
            soft_l = soft_cross_entropy(logits, t_cand[idx], t_probs[idx])
            loss = 0.45 * hard_l + 0.55 * soft_l
            if use_ext:
                e_pick = torch.as_tensor(rng.integers(0, len(ext_u_imgs), size=len(raw) // 2), device="cuda")
                xe = augment(preprocess(ext_u_imgs[e_pick]), 1.0, 10.0, 0.95, 1.05)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    le = model(xe).float()
                loss = 0.72 * loss + 0.28 * soft_cross_entropy(le, e_cand[e_pick], e_probs[e_pick])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        top1 = eval_single(model, images, labels, valid_idx)
        print(f"[{arch_tag}_gen2 f{fold}] ep={epoch:02d} top1={top1:.4f} ({time.monotonic()-t0:.0f}s)", flush=True)
        if top1 > best_top1:
            best_top1 = top1
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, best_dist_pt)

    del model, opt, t_cand, t_probs
    if use_ext:
        del ext_u_imgs, e_cand, e_probs
    torch.cuda.empty_cache()
    return best_dist_pt


def main() -> None:
    device = "cuda"
    raw_labeled_np = np.load("labeled_32x32.npy", mmap_mode="r")
    raw_unl_np = np.load("unl/test_32x32.npy", mmap_mode="r")
    raw_test_np = np.load("test_32x32.npy", mmap_mode="r")

    images = clean_impulses_gpu(torch.as_tensor(np.asarray(raw_labeled_np).copy(), device=device))
    unl_imgs = clean_impulses_gpu(torch.as_tensor(np.asarray(raw_unl_np).copy(), device=device))
    labels_np = np.load("labeled_ids.npy").astype(np.int64)
    folds_lbl = stratified_folds(labels_np, 3, SEED)
    f0_valid = np.flatnonzero(folds_lbl == 0)
    f0_train = np.flatnonzero(folds_lbl != 0)

    # 1. Load active table & calibration + highest Fold-0 anchor (6C or 6B!)
    m2 = json.loads((L / "milestone2_pair_table_v4.json").read_text())
    nof0_tab = np.load(Path(m2["active_nof0_table"]))
    full_tab = np.load(Path(m2["active_full_table"]))
    m3 = json.loads((L / "milestone3_calibration.json").read_text())["active"]
    s0, s1, debias, beta_lap = float(m3["s0"]), float(m3["s1"]), float(m3["debias"]), float(m3["beta_lap"])

    rows = pd.read_csv("data/train.csv.gz")
    identity = {}
    for side in (1, 2):
        for p, v in zip(rows[f"noisy_image{side}"], rows[f"pokemon_id{side}"]):
            if pd.notna(v):
                identity[p] = int(v)
    paths = sorted(identity)
    true = np.array([identity[p] for p in paths], dtype=np.int64)
    folds = stratified_folds(true - 1, 3, SEED)
    index = pd.Index(paths)
    f1, f2 = index.get_indexer(rows.noisy_image1), index.get_indexer(rows.noisy_image2)
    known = (f1 >= 0) & (f2 >= 0)
    f1, f2 = f1[known], f2[known]
    a, b = true[f1], true[f2]
    y = rows.result.to_numpy(np.float32)[known]
    pair_f0 = (folds[f1] == 0) & (folds[f2] == 0)
    rng = np.random.default_rng(3780)
    pair_rows = np.flatnonzero(pair_f0)
    rng.shuffle(pair_rows)
    split = len(pair_rows) // 2

    known_allowed = np.triu(np.ones((1026, 1026), dtype=np.float32), k=1)
    known_allowed[0, :] = 0.0
    known_allowed[:, 0] = 0.0
    known_allowed[a[~pair_f0], b[~pair_f0]] = 0.0

    unk = rows.pokemon_id1.isna() & rows.pokemon_id2.isna()
    u_idx = pd.Index(Path("data/unlabeled_paths.txt").read_text().split())
    eu = u_idx.get_indexer(rows.noisy_image1[unk])
    ev = u_idx.get_indexer(rows.noisy_image2[unk])
    uy = rows.result[unk].to_numpy(np.float32)

    norm_rows = lambda m: m / np.maximum(m.sum(1, keepdims=True), 1e-8)

    with np.load(L / "rx_v1_fold0_scores.npz") as fz:
        v_idx = fz["valid_indices"].astype(np.int64)
        cand = fz["candidates"].astype(np.int64)
        q_v1 = fz["posteriors"].astype(np.float32)
        oop = fz["oop_mass"].astype(np.float32)
    with np.load(L / "rx_v1_unlabeled_f0only.npz") as uz:
        u_cand = uz["candidates"].astype(np.int64)
        u_v1 = uz["posteriors"].astype(np.float32)

    # Check if Milestone 6C passed; otherwise start directly from verified 80.92531 LB anchor (`rx_v1_*`)!
    m6c_path = L / "milestone6c_grand_slam.json"
    use_6c = m6c_path.exists() and json.loads(m6c_path.read_text()).get("passed", False)
    if use_6c and (L / "rx_v6_grand_slam_f0_post.npz").exists():
        with np.load(L / "rx_v6_grand_slam_f0_post.npz") as gz:
            q_base = gz["q_post"].astype(np.float32)
            u_base = gz["u_post"].astype(np.float32)
        full_u_pack = L / "rx_v6_grand_slam_unlabeled.npz"
        full_t_pack = L / "rx_v6_grand_slam_test.npz"
        print("Milestone 7 starting on top of Milestone 6C Grand-Slam posteriors!", flush=True)
    else:
        q_base = q_v1
        u_base = u_v1
        full_u_pack = L / "rx_v1_unlabeled.npz"
        full_t_pack = L / "rx_v1_test.npz"
        print("Milestone 7 starting on top of verified 80.92531 LB anchor (rx_v1_*)!", flush=True)

    # Physical regime descriptors on Fold-0 valid & Unlabeled
    z_hf_f0, z_bd_f0 = compute_physical_regime_zscores(raw_labeled_np[v_idx])
    z_hf_u, z_bd_u = compute_physical_regime_zscores(raw_unl_np)

    # 2. Specialist 7C: Forensic Candidate-Difference Template Reranker (Leak-Free Fold-0!)
    t0 = time.monotonic()
    mu_f0, isd_f0 = build_forensic_templates(images[f0_train], labels_np[f0_train], unl_imgs, u_cand, u_base)
    for_ll_f0 = score_forensic_diff_top20(images[v_idx], cand, mu_f0, isd_f0)
    for_ll_u0 = score_forensic_diff_top20(unl_imgs, u_cand, mu_f0, isd_f0)
    del mu_f0, isd_f0
    print(f"[7C ForensicDiff] Fold-0 + Unl scored in {time.monotonic()-t0:.1f}s", flush=True)

    # 3. Specialist 7A: CentroidLogPolarNet Fold-0
    lp_f0_pt = train_specialist_fold(
        "logpolar_spec", CentroidLogPolarNet, fold=0,
        images=images, labels_np=labels_np,
        rot_deg=20.0, scale_lo=0.88, scale_hi=1.14,
        s1_epochs=24, s2_epochs=10, seed_base=710000,
    )
    lp_f0_path = L / "logpolar_spec_f0_rot5.npy"
    lp_u0_path = L / "unl_f0_logpolar_spec.npy"
    if not lp_f0_path.exists() or not lp_u0_path.exists():
        m_lp = CentroidLogPolarNet().cuda().eval()
        m_lp.load_state_dict(torch.load(lp_f0_pt, map_location="cpu", weights_only=True))
        if not lp_f0_path.exists():
            zf = np.zeros(len(images), dtype=np.uint8)
            p_lp = score_model(m_lp, "logpolar", images, f0_valid, batch_size=256, facing=zf)
            np.save(lp_f0_path, p_lp.astype(np.float16))
            print(f"[7A LogPolar] Fold-0 6-view TTA top1 = {float((p_lp.argmax(1)==labels_np[f0_valid]).mean()):.5f}", flush=True)
        if not lp_u0_path.exists():
            uzf = np.zeros(len(unl_imgs), dtype=np.uint8)
            pu_lp = score_model(m_lp, "logpolar", unl_imgs, np.arange(len(unl_imgs)), batch_size=256, facing=uzf)
            np.save(lp_u0_path, pu_lp.astype(np.float16))
        del m_lp
        torch.cuda.empty_cache()

    # 4. Specialist 7B: SpectralMultiBandSilhouetteNet Fold-0
    sp_f0_pt = train_specialist_fold(
        "spectral_spec", SpectralMultiBandSilhouetteNet, fold=0,
        images=images, labels_np=labels_np,
        rot_deg=10.0, scale_lo=0.93, scale_hi=1.07,
        s1_epochs=24, s2_epochs=10, seed_base=720000,
    )
    sp_f0_path = L / "spectral_spec_f0_rot5.npy"
    sp_u0_path = L / "unl_f0_spectral_spec.npy"
    if not sp_f0_path.exists() or not sp_u0_path.exists():
        m_sp = SpectralMultiBandSilhouetteNet().cuda().eval()
        m_sp.load_state_dict(torch.load(sp_f0_pt, map_location="cpu", weights_only=True))
        if not sp_f0_path.exists():
            zf = np.zeros(len(images), dtype=np.uint8)
            p_sp = score_model(m_sp, "spectral", images, f0_valid, batch_size=256, facing=zf)
            np.save(sp_f0_path, p_sp.astype(np.float16))
            print(f"[7B Spectral] Fold-0 6-view TTA top1 = {float((p_sp.argmax(1)==labels_np[f0_valid]).mean()):.5f}", flush=True)
        if not sp_u0_path.exists():
            uzf = np.zeros(len(unl_imgs), dtype=np.uint8)
            pu_sp = score_model(m_sp, "spectral", unl_imgs, np.arange(len(unl_imgs)), batch_size=256, facing=uzf)
            np.save(sp_u0_path, pu_sp.astype(np.float16))
        del m_sp
        torch.cuda.empty_cache()

    q_lp = norm_rows(gather_cand(np.load(lp_f0_path, mmap_mode="r"), cand))
    u_lp = norm_rows(gather_cand(np.load(lp_u0_path, mmap_mode="r"), u_cand))
    q_spec = norm_rows(gather_cand(np.load(sp_f0_path, mmap_mode="r"), cand))
    u_spec = norm_rows(gather_cand(np.load(sp_u0_path, mmap_mode="r"), u_cand))

    z_lap_f0 = z_lap_from_gallery(L / "gallery_rx_top40_f0.npz", v_idx)
    pos = np.full(len(paths), -1, dtype=np.int32)
    pos[v_idx] = np.arange(len(v_idx))
    one, two = pos[f1[pair_rows]], pos[f2[pair_rows]]
    actual = y[pair_rows]
    ta, tb = a[pair_rows], b[pair_rows]

    # Candidate MoE recipes: (w_lp, w_lp_bd, w_sp, w_sp_hf, w_for, temp)
    # Note: ALL weights are multiplied by `unc = clip((0.95 - p_top1)/0.45, 0, 1)`, so 90.5% clean/confident images get 0.000 perturbation!
    in_c_f0 = (cand == true[v_idx, None]).astype(np.float32)
    grid = [
        (0.00, 0.00, 0.00, 0.00, 0.00, 0.85),  # Exact current best anchor!
        (0.00, 0.00, 0.00, 0.00, 0.06, 0.85),  # 7C Forensic only (unc-gated)
        (0.02, 0.07, 0.00, 0.00, 0.00, 0.85),  # 7A LogPolar border-gated
        (0.05, 0.05, 0.00, 0.00, 0.00, 0.85),  # 7A LogPolar unc + border
        (0.00, 0.00, 0.02, 0.08, 0.00, 0.85),  # 7B Spectral FFT-noise-gated
        (0.00, 0.00, 0.05, 0.06, 0.00, 0.85),  # 7B Spectral unc + FFT noise
        (0.02, 0.06, 0.02, 0.07, 0.00, 0.85),  # 7A + 7B Physical Regime MoE
        (0.04, 0.05, 0.04, 0.06, 0.00, 0.85),  # 7A + 7B Balanced Regime MoE
        (0.02, 0.06, 0.02, 0.07, 0.05, 0.85),  # Full 7A + 7B + 7C Specialist Council!
        (0.04, 0.04, 0.04, 0.05, 0.04, 0.85),  # Full Council Moderate
        (0.02, 0.07, 0.02, 0.08, 0.00, 0.80),  # Sharp Regime MoE Council
    ]

    trials = []
    for w_lp, w_lp_bd, w_sp, w_sp_hf, w_for, temp in grid:
        qm = fuse_council_moe(q_base, q_lp, q_spec, for_ll_f0, z_hf_f0, z_bd_f0,
                              w_lp, w_lp_bd, w_sp, w_sp_hf, w_for, temp)
        um = fuse_council_moe(u_base, u_lp, u_spec, for_ll_u0, z_hf_u, z_bd_u,
                              w_lp, w_lp_bd, w_sp, w_sp_hf, w_for, temp)

        _, _, ul0 = run_unlabeled_bp_hetero(u_cand, um, eu, ev, uy, nof0_tab, known_allowed, s0=s0, s1=s1, device=device)
        _, _, tl0, _, _ = target_cavity_predict(qm, oop, cand, one, two, known_allowed, ul0, nof0_tab, debias=debias, device=device)
        _, _, ul1 = run_unlabeled_bp_hetero(u_cand, um, eu, ev, uy, nof0_tab, known_allowed, s0=s0, s1=s1, external_log_avail=tl0, device=device)

        best_s = None
        for sc in (0.90, 0.95, 1.00):
            qs = adaptive_power_scale(qm, np.clip(sc - beta_lap * z_lap_f0, 0.55, 1.45))
            pred, top_p, _, _, _ = target_cavity_predict(
                qs, oop, cand, one, two, known_allowed, ul1, nof0_tab,
                debias=debias, cavity_iters=12, use_escape=False, device=device)
            tr_r = rmse(pred[:split], actual[:split])
            te_r = rmse(pred[split:], actual[split:])
            both_ok = float(((top_p[:, 0] == ta) & (top_p[:, 1] == tb))[split:].mean())
            it = {
                "scale_base": sc, "tune_rmse": tr_r, "test_rmse": te_r,
                "tune_score": score_fn(tr_r), "test_score": score_fn(te_r),
                "test_both_correct": both_ok,
            }
            if best_s is None or it["tune_rmse"] < best_s["tune_rmse"]:
                best_s = it

        rec = {
            "weights": [w_lp, w_lp_bd, w_sp, w_sp_hf, w_for, temp],
            "img_top1": float((cand[np.arange(len(v_idx)), qm.argmax(1)] == true[v_idx]).mean()),
            "img_brier": float(((qm - in_c_f0) ** 2).sum(1).mean()),
            "selected": best_s,
        }
        trials.append(rec)
        print(f"[council_f0] w={rec['weights']} top1={rec['img_top1']:.5f} brier={rec['img_brier']:.5f} "
              f"tune={best_s['tune_score']:.4f} test={best_s['test_score']:.4f} both={best_s['test_both_correct']:.5f}", flush=True)

    anchor = trials[0]
    brier_safe = [r for r in trials if r["img_brier"] <= anchor["img_brier"] + 1e-5 and r["img_top1"] >= anchor["img_top1"] - 1e-5]
    pool = brier_safe if brier_safe else [anchor]
    chosen = min(pool, key=lambda r: r["selected"]["tune_rmse"])
    passed = (chosen is not anchor
              and chosen["selected"]["test_score"] > anchor["selected"]["test_score"] + 0.005
              and chosen["img_brier"] <= anchor["img_brier"] + 1e-5)
    print(f"Milestone 7 Council gate: passed={passed} "
          f"(test {anchor['selected']['test_score']:.4f} -> {chosen['selected']['test_score']:.4f}, "
          f"brier {anchor['img_brier']:.5f} -> {chosen['img_brier']:.5f})", flush=True)

    sub_csv = None
    if passed:
        w_lp, w_lp_bd, w_sp, w_sp_hf, w_for, temp = chosen["weights"]
        need_lp = (w_lp > 0.0 or w_lp_bd > 0.0)
        need_sp = (w_sp > 0.0 or w_sp_hf > 0.0)
        need_for = (w_for > 0.0)

        with np.load(full_u_pack) as uz:
            tu_cand = uz["candidates"].astype(np.int64)
            tu_base = uz["posteriors"].astype(np.float32)
            tu_oop = uz["oop_mass"].astype(np.float32)
        with np.load(full_t_pack) as tz:
            t_cand = tz["candidates"].astype(np.int64)
            t_base = tz["posteriors"].astype(np.float32)
            t_oop = tz["oop_mass"].astype(np.float32)

        test_imgs = clean_impulses_gpu(torch.as_tensor(np.asarray(raw_test_np).copy(), device=device))
        z_hf_t, z_bd_t = compute_physical_regime_zscores(raw_test_np)

        tu_lp = t_lp = None
        if need_lp:
            for fold in (1, 2):
                train_specialist_fold("logpolar_spec", CentroidLogPolarNet, fold=fold,
                                      images=images, labels_np=labels_np,
                                      rot_deg=20.0, scale_lo=0.88, scale_hi=1.14,
                                      s1_epochs=24, s2_epochs=10, seed_base=710000)
            u_lp_full = L / "unl_logpolar_spec.npy"
            t_lp_full = L / "test_logpolar_spec.npy"
            if not u_lp_full.exists() or not t_lp_full.exists():
                ms = []
                for f in (0, 1, 2):
                    m = CentroidLogPolarNet().cuda().eval()
                    m.load_state_dict(torch.load(L / f"logpolar_spec_gen2_f{f}_best.pt", map_location="cpu", weights_only=True))
                    ms.append(m)
                if not u_lp_full.exists():
                    acc = sum(score_model(m, "lp", unl_imgs, np.arange(len(unl_imgs)), batch_size=256, facing=np.zeros(len(unl_imgs), dtype=np.uint8)) for m in ms) / 3.0
                    np.save(u_lp_full, acc.astype(np.float16))
                if not t_lp_full.exists():
                    acc = sum(score_model(m, "lp", test_imgs, np.arange(len(test_imgs)), batch_size=256, facing=np.zeros(len(test_imgs), dtype=np.uint8)) for m in ms) / 3.0
                    np.save(t_lp_full, acc.astype(np.float16))
                del ms
                torch.cuda.empty_cache()
            tu_lp = norm_rows(gather_cand(np.load(u_lp_full, mmap_mode="r"), tu_cand))
            t_lp = norm_rows(gather_cand(np.load(t_lp_full, mmap_mode="r"), t_cand))

        tu_spec = t_spec = None
        if need_sp:
            for fold in (1, 2):
                train_specialist_fold("spectral_spec", SpectralMultiBandSilhouetteNet, fold=fold,
                                      images=images, labels_np=labels_np,
                                      rot_deg=10.0, scale_lo=0.93, scale_hi=1.07,
                                      s1_epochs=24, s2_epochs=10, seed_base=720000)
            u_sp_full = L / "unl_spectral_spec.npy"
            t_sp_full = L / "test_spectral_spec.npy"
            if not u_sp_full.exists() or not t_sp_full.exists():
                ms = []
                for f in (0, 1, 2):
                    m = SpectralMultiBandSilhouetteNet().cuda().eval()
                    m.load_state_dict(torch.load(L / f"spectral_spec_gen2_f{f}_best.pt", map_location="cpu", weights_only=True))
                    ms.append(m)
                if not u_sp_full.exists():
                    acc = sum(score_model(m, "sp", unl_imgs, np.arange(len(unl_imgs)), batch_size=256, facing=np.zeros(len(unl_imgs), dtype=np.uint8)) for m in ms) / 3.0
                    np.save(u_sp_full, acc.astype(np.float16))
                if not t_sp_full.exists():
                    acc = sum(score_model(m, "sp", test_imgs, np.arange(len(test_imgs)), batch_size=256, facing=np.zeros(len(test_imgs), dtype=np.uint8)) for m in ms) / 3.0
                    np.save(t_sp_full, acc.astype(np.float16))
                del ms
                torch.cuda.empty_cache()
            tu_spec = norm_rows(gather_cand(np.load(u_sp_full, mmap_mode="r"), tu_cand))
            t_spec = norm_rows(gather_cand(np.load(t_sp_full, mmap_mode="r"), t_cand))

        tu_for = t_for = None
        if need_for:
            mu_all, isd_all = build_forensic_templates(images, labels_np, unl_imgs, tu_cand, tu_base)
            tu_for = score_forensic_diff_top20(unl_imgs, tu_cand, mu_all, isd_all)
            t_for = score_forensic_diff_top20(test_imgs, t_cand, mu_all, isd_all)
            del mu_all, isd_all

        tu_pri = fuse_council_moe(tu_base, tu_lp, tu_spec, tu_for, z_hf_u, z_bd_u,
                                  w_lp, w_lp_bd, w_sp, w_sp_hf, w_for, temp)
        t_post_raw = fuse_council_moe(t_base, t_lp, t_spec, t_for, z_hf_t, z_bd_t,
                                      w_lp, w_lp_bd, w_sp, w_sp_hf, w_for, temp)

        np.savez_compressed(L / "rx_v7_council_unlabeled.npz",
                            candidates=tu_cand.astype(np.uint16),
                            posteriors=tu_pri.astype(np.float32),
                            oop_mass=tu_oop)
        np.savez_compressed(L / "rx_v7_council_test.npz",
                            candidates=t_cand.astype(np.uint16),
                            posteriors=t_post_raw.astype(np.float32),
                            oop_mass=t_oop)

        sc_win = chosen["selected"]["scale_base"]
        z_lap_test = z_lap_from_gallery(L / "gallery_rx_top40_test.npz", None)
        t_post = adaptive_power_scale(t_post_raw, np.clip(sc_win - beta_lap * z_lap_test, 0.55, 1.45))

        with np.load("data/known_battles.npz") as kz:
            ka_all, kb_all = kz["a"].astype(np.int64), kz["b"].astype(np.int64)
        known_full = np.triu(np.ones((1026, 1026), dtype=np.float32), k=1)
        known_full[0, :] = 0.0
        known_full[:, 0] = 0.0
        known_full[ka_all, kb_all] = 0.0

        test_paths = Path("test_paths.txt").read_text().splitlines()
        p_idx = pd.Index(test_paths)
        test_df = pd.read_csv("data/test.csv.gz")
        idx1 = p_idx.get_indexer(test_df.noisy_image1)
        idx2 = p_idx.get_indexer(test_df.noisy_image2)

        _, _, u_lnot = run_unlabeled_bp_hetero(
            tu_cand, tu_pri, eu, ev, uy, full_tab, known_full, s0=s0, s1=s1, device=device)
        for outer in range(2):
            pred, _, test_lnot, _, _ = target_cavity_predict(
                t_post, t_oop, t_cand, idx1, idx2, known_full, u_lnot, full_tab,
                debias=debias, cavity_iters=12, use_escape=False, device=device)
            if outer == 0:
                _, _, u_lnot = run_unlabeled_bp_hetero(
                    tu_cand, tu_pri, eu, ev, uy, full_tab, known_full,
                    s0=s0, s1=s1, external_log_avail=test_lnot, device=device)

        pred = np.clip(pred, 0.0, 10.0)
        sub_csv = L / "submission_rx_v7_specialist_council.csv"
        with sub_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_id", "result"])
            for row_idx, val in enumerate(pred):
                writer.writerow([f"test-{row_idx:06d}", f"{float(val):.6f}"])
        (L / "submission_latest_best.csv").write_bytes(sub_csv.read_bytes())
        print(f"Milestone 7 Specialist Council complete -> {sub_csv}", flush=True)

    report = {
        "anchor": anchor,
        "chosen": chosen,
        "passed": passed,
        "submission_csv": str(sub_csv) if sub_csv else None,
        "trials": trials,
    }
    (L / "milestone7_specialist_council.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
