"""Milestone 9B (v9 Grand Specialist Council):
Adds TWO genuinely non-local / large-receptive-field orthogonal specialists on top of v8:
  1. `CentroidRelPosViT`:
     Soft-centroid-stabilized Overlapping-Stem Vision Transformer (8x8 = 64 spatial tokens,
     dim=320, depth=8, 8 heads, learnable 2D Relative Position Bias table) + Siamese Exact
     D2 Bilateral Flip-Invariant No-GAP Spatial Token Head.
  2. `CentroidConvNeXtV2`:
     Soft-centroid-stabilized 7x7 Large-Kernel ConvNeXt-V2 with Global Response Normalization
     (GRN) across 32x32 -> 16x16 -> 8x8 stages + Siamese Exact D2 Flip-Invariant Spatial Head.

Upgrades over Milestone 7's training & fusion pipeline:
  - Stage 2 soft distillation uses the 97.14%-accurate 6-view `v7` Council OOF posteriors +
    v8 Battle-BP Unlabeled node posteriors across ALL 3 FOLDS (fixing M7's `fold == 0` bug
    while keeping Fold-0 100.000% leak-free via `f0only` packs!).
  - Strict Zero-Perturbation Confidence Gate (`p_max >= 0.95` -> 0.000 specialist weight) +
    Milestone 9A Exact-Quota Equilibrium Debiaser (`kappa=12, alpha=0.25, debias=0.90`).
  - Must beat LOCKED v8 (`img_brier=0.04008`, `HG_test=80.2054`, `M7_test=80.2273`) on Fold-0
    before training Folds 1 & 2 and exporting `submission_rx_v9_grand_council.csv`.
"""
from __future__ import annotations

import argparse
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
from milestone7_specialist_council import (
    CentroidLogPolarNet,
    compute_physical_regime_zscores,
    eval_single,
    fuse_council_moe,
)
from milestone9a_exact_quota_debias import (
    apply_zero_perturbation_bias,
    build_all_folds_v7_oof,
    ensure_logpolar_all_folds_rot5,
    fit_smoothed_exact_quota_bias,
)
from train_large_image import augment, preprocess

L = Path("outputs/large")


def soft_ce_per_sample(logits: torch.Tensor, cand: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Per-image KL of a 40-way teacher against the student, up to teacher entropy."""
    log_p = F.log_softmax(logits, dim=1)
    gathered = log_p.gather(1, (cand - 1).long())
    return -(q * gathered).sum(1)


def score_fn(val: float) -> float:
    return 100.0 * (1.0 - val / DENOMINATOR)


# =============================================================================
# Shared Differentiable Centroid Stabilization Stem Helper
# =============================================================================

class SoftCentroidAligner(nn.Module):
    """Shifts each 32x32 sprite so its soft foreground centroid sits at (0, 0),
    returning `(B, 3, 32, 32)` = `[centered_lum, radial_r, coord_y]`.
    Note: `coord_y` and `radial_r` are BOTH invariant to horizontal flip (`x -> -x`),
    guaranteeing exact commutation with Siamese horizontal reflection!"""
    def __init__(self):
        super().__init__()
        ys = torch.linspace(-1.0, 1.0, 32)
        xs = torch.linspace(-1.0, 1.0, 32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        wy = torch.hann_window(32, periodic=False).view(1, 1, 32, 1).sqrt()
        wx = torch.hann_window(32, periodic=False).view(1, 1, 1, 32).sqrt()
        rr = torch.sqrt(gx * gx + gy * gy).view(1, 1, 32, 32)
        self.register_buffer("grid_y", gy.view(1, 1, 32, 32), persistent=False)
        self.register_buffer("grid_x", gx.view(1, 1, 32, 32), persistent=False)
        self.register_buffer("base_yx", torch.stack([gx, gy], dim=-1).unsqueeze(0), persistent=False)  # (1,32,32,2)
        self.register_buffer("hann2d", wy * wx, persistent=False)
        self.register_buffer("r_map", (rr - rr.mean()) / rr.std(), persistent=False)
        self.register_buffer("y_map", gy.view(1, 1, 32, 32), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        lum = x.mean(dim=1, keepdim=True)  # (B, 1, 32, 32)
        bdr = torch.cat([lum[:, :, 0, :], lum[:, :, -1, :], lum[:, :, :, 0], lum[:, :, :, -1]], dim=-1)
        bg = bdr.median(dim=-1).values.view(b, 1, 1, 1)
        sal = ((lum - bg).abs() * self.hann2d).pow(1.5) + 1e-3
        sal = sal / sal.sum(dim=(2, 3), keepdim=True)
        cx = (sal * self.grid_x).sum(dim=(2, 3)).clamp(-0.24, 0.24)  # (B, 1)
        cy = (sal * self.grid_y).sum(dim=(2, 3)).clamp(-0.24, 0.24)  # (B, 1)

        shift = torch.stack([cx, cy], dim=-1).view(b, 1, 1, 2)
        grid = self.base_yx + shift
        centered = F.grid_sample(lum, grid, mode="bilinear", padding_mode="border", align_corners=True)
        dtype = centered.dtype
        return torch.cat([
            centered,
            self.r_map.expand(b, -1, -1, -1).to(dtype),
            self.y_map.expand(b, -1, -1, -1).to(dtype),
        ], dim=1)


# =============================================================================
# Specialist 9B-1: Centroid-Aligned 2D-RelPos Vision Transformer (`CentroidRelPosViT`)
# =============================================================================

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


class RelPos2DAttention(nn.Module):
    """Multi-head self-attention on an HxW (8x8 = 64) spatial token grid with
    learnable 2D relative position bias `B[dy + H - 1, dx + W - 1, head]`."""
    def __init__(self, dim: int = 320, num_heads: int = 8, grid_hw: int = 8, attn_drop: float = 0.05):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = attn_drop

        # Build (64, 64) relative coordinate index into ((2H-1)*(2W-1), num_heads)
        coords = torch.stack(torch.meshgrid(torch.arange(grid_hw), torch.arange(grid_hw), indexing="ij"))  # (2, H, W)
        flat = coords.reshape(2, -1)                                     # (2, N)
        rel = flat[:, :, None] - flat[:, None, :]                        # (2, N, N)
        rel[0] += grid_hw - 1
        rel[1] += grid_hw - 1
        rel_idx = rel[0] * (2 * grid_hw - 1) + rel[1]                    # (N, N)
        self.register_buffer("rel_idx", rel_idx.reshape(-1), persistent=False)
        self.rel_bias = nn.Parameter(torch.zeros((2 * grid_hw - 1) * (2 * grid_hw - 1), num_heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, H, N, Dh)
        bias = self.rel_bias[self.rel_idx].view(N, N, self.num_heads).permute(2, 0, 1).unsqueeze(0)  # (1, H, N, N)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias.to(q.dtype),
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class ViTBlock(nn.Module):
    def __init__(self, dim: int = 320, num_heads: int = 8, mlp_ratio: float = 3.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RelPos2DAttention(dim, num_heads=num_heads, grid_hw=8)
        self.ls1 = nn.Parameter(0.1 * torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hid),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hid, dim),
        )
        self.ls2 = nn.Parameter(0.1 * torch.ones(dim))
        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp(self.attn(self.norm1(x)) * self.ls1)
        x = x + self.dp(self.mlp(self.norm2(x)) * self.ls2)
        return x


class CentroidRelPosViT(nn.Module):
    """Overlapping Conv Stem (32 -> 16 -> 8) + 8-layer 2D-RelPos Transformer +
    Siamese Exact D2 Flip-Invariant Spatial Token Head."""
    def __init__(self, dim: int = 320, depth: int = 8, num_heads: int = 8):
        super().__init__()
        self.align = SoftCentroidAligner()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 192, 3, stride=2, padding=1, bias=False),   # 16x16
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),
            nn.Conv2d(192, dim, 3, stride=2, padding=1, bias=False),  # 8x8 = 64 tokens!
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )
        dpr = torch.linspace(0.0, 0.12, depth).tolist()
        self.blocks = nn.ModuleList([
            ViTBlock(dim=dim, num_heads=num_heads, mlp_ratio=3.0, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.spatial_proj = nn.Sequential(
            nn.Linear(dim, 24, bias=False),
            nn.LayerNorm(24),
            nn.SiLU(inplace=True),
        )
        # Global mean+std (640) + 64*24 symmetric (1536) + 64*24 |asymmetric| (1536) = 3712
        feat_dim = dim * 2 + 2 * (64 * 24)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.25),
            nn.Linear(feat_dim, 1536, bias=False),
            nn.BatchNorm1d(1536),
            nn.SiLU(inplace=True),
            nn.Dropout(0.18),
            nn.Linear(1536, NUM_CLASSES),
        )

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.align(x)
        h = self.stem(c)                                  # (B, D, 8, 8)
        tok = h.flatten(2).transpose(1, 2)                # (B, 64, D)
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)                              # (B, 64, D)
        g_mean = tok.mean(dim=1)
        g_std = (tok.var(dim=1, unbiased=False) + 1e-5).sqrt()
        glob = torch.cat([g_mean, g_std], dim=1)          # (B, 2D)
        sp = self.spatial_proj(tok).flatten(1)            # (B, 64*24)
        return glob, sp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ra, ga = self._encode(x)
        rb, gb = self._encode(torch.flip(x, dims=[-1]))
        return self.classifier(torch.cat([0.5 * (ra + rb), 0.5 * (ga + gb), (ga - gb).abs()], dim=1))


# =============================================================================
# Specialist 9B-2: Centroid-Aligned 7x7 Large-Kernel ConvNeXt-V2 with GRN
# =============================================================================

class GRN2d(nn.Module):
    """Global Response Normalization (ConvNeXt-V2) over spatial (H, W) for `(B, C, H, W)`."""
    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x.float(), p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return x + (self.gamma * (x.float() * nx) + self.beta).to(x.dtype)


class ConvNeXtV2Block(nn.Module):
    """7x7 depthwise large-kernel block + GroupNorm/ChannelLN + 3x expansion + GRN + DropPath."""
    def __init__(self, dim: int, kernel_size: int = 7, mlp_ratio: int = 3, drop_path: float = 0.0):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=True)
        self.norm = nn.GroupNorm(1, dim)  # Exact Channel-wise LayerNorm in NCHW!
        hid = dim * mlp_ratio
        self.pw1 = nn.Conv2d(dim, hid, 1)
        self.grn = GRN2d(hid)
        self.pw2 = nn.Conv2d(hid, dim, 1)
        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x)
        h = self.norm(h)
        h = F.gelu(self.pw1(h))
        h = self.grn(h)
        h = self.pw2(h)
        return x + self.dp(h)


class CentroidConvNeXtV2(nn.Module):
    """Centroid-stabilized 7x7 Large-Kernel ConvNeXt-V2 (widths 112 -> 224 -> 384)
    + Exact D2 Flip-Invariant Spatial Grid Head."""
    def __init__(self, widths=(112, 224, 384), depths=(3, 4, 5)):
        super().__init__()
        self.align = SoftCentroidAligner()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
            nn.GroupNorm(1, widths[0]),
            nn.GELU(),
        )
        dpr = torch.linspace(0.0, 0.12, sum(depths)).tolist()
        cur = 0
        self.stage1 = nn.Sequential(*[ConvNeXtV2Block(widths[0], 7, 3, dpr[cur + i]) for i in range(depths[0])])
        cur += depths[0]
        self.down2 = nn.Sequential(nn.GroupNorm(1, widths[0]), nn.Conv2d(widths[0], widths[1], 2, stride=2))
        self.stage2 = nn.Sequential(*[ConvNeXtV2Block(widths[1], 7, 3, dpr[cur + i]) for i in range(depths[1])])
        cur += depths[1]
        self.down3 = nn.Sequential(nn.GroupNorm(1, widths[1]), nn.Conv2d(widths[1], widths[2], 2, stride=2))
        self.stage3 = nn.Sequential(*[ConvNeXtV2Block(widths[2], 7, 3, dpr[cur + i]) for i in range(depths[2])])

        self.norm = nn.GroupNorm(1, widths[2])
        self.sp_proj = nn.Sequential(
            nn.Conv2d(widths[2], 24, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
        )
        feat_dim = widths[2] * 2 + 2 * (24 * 8 * 8)  # 768 + 3072 = 3840
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.25),
            nn.Linear(feat_dim, 1536, bias=False),
            nn.BatchNorm1d(1536),
            nn.SiLU(inplace=True),
            nn.Dropout(0.18),
            nn.Linear(1536, NUM_CLASSES),
        )

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(self.align(x))       # 32x32
        h = self.stage1(h)
        h = self.stage2(self.down2(h))     # 16x16
        h = self.norm(self.stage3(self.down3(h)))  # 8x8
        g_mean = h.mean(dim=(2, 3))
        g_std = (h.var(dim=(2, 3), unbiased=False) + 1e-5).sqrt()
        glob = torch.cat([g_mean, g_std], dim=1)
        sp = self.sp_proj(h).flatten(1)
        return glob, sp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ra, ga = self._encode(x)
        rb, gb = self._encode(torch.flip(x, dims=[-1]))
        return self.classifier(torch.cat([0.5 * (ra + rb), 0.5 * (ga + gb), (ga - gb).abs()], dim=1))


# =============================================================================
# Upgraded All-Fold v8-Teacher Specialist Trainer
# =============================================================================

@torch.no_grad()
def ensure_unl_bp_packs(
    u_cand_f0: np.ndarray, um_v7_f0: np.ndarray,
    eu: np.ndarray, ev: np.ndarray, uy: np.ndarray,
    nof0_tab: np.ndarray, known_allowed_f0: np.ndarray,
    full_tab: np.ndarray, s0: float, s1: float, device: str = "cuda",
) -> tuple[Path, Path]:
    """Build 100%-leak-free Fold-0 Unlabeled Battle-BP pack AND Full Unlabeled Battle-BP pack
    from v8 Council posteriors so ALL 3 folds get high-purity unlabeled soft distillation!"""
    p_f0 = L / "bp_v8_unl_f0only_pack.npz"
    if not p_f0.exists():
        u_node, _, _ = run_unlabeled_bp_hetero(
            u_cand_f0, um_v7_f0, eu, ev, uy, nof0_tab, known_allowed_f0, s0=s0, s1=s1, device=device
        )
        np.savez_compressed(p_f0, candidates=u_cand_f0.astype(np.uint16), posteriors=u_node.astype(np.float32))
        print(f"[9B] Built {p_f0.name} (mean max conf={float(u_node.max(1).mean()):.4f})", flush=True)

    p_full = L / "bp_v8_unl_full_pack.npz"
    if not p_full.exists():
        with np.load(L / "rx_v8_debias_unlabeled.npz") as uz:
            tu_cand = uz["candidates"].astype(np.int64)
            tu_pri = uz["posteriors"].astype(np.float32)
        with np.load("data/known_battles.npz") as kz:
            ka_all, kb_all = kz["a"].astype(np.int64), kz["b"].astype(np.int64)
        known_full = np.triu(np.ones((1026, 1026), dtype=np.float32), k=1)
        known_full[0, :] = 0.0
        known_full[:, 0] = 0.0
        known_full[ka_all, kb_all] = 0.0
        u_node_full, _, _ = run_unlabeled_bp_hetero(
            tu_cand, tu_pri, eu, ev, uy, full_tab, known_full, s0=s0, s1=s1, device=device
        )
        np.savez_compressed(p_full, candidates=tu_cand.astype(np.uint16), posteriors=u_node_full.astype(np.float32))
        print(f"[9B] Built {p_full.name} (mean max conf={float(u_node_full.max(1).mean()):.4f})", flush=True)

    return p_f0, p_full


def train_specialist_v9_fold(
    arch_tag: str,
    build_fn,
    fold: int,
    images: torch.Tensor,
    labels_np: np.ndarray,
    teacher_cand_oof: np.ndarray,
    teacher_post_oof: np.ndarray,
    unl_bp_pack: Path,
    rot_deg: float = 16.0,
    scale_lo: float = 0.90,
    scale_hi: float = 1.11,
    s1_epochs: int = 24,
    s2_epochs: int = 12,
    peak_lr: float = 8.5e-4,
    seed_base: int = 910000,
) -> Path:
    best_dist_pt = L / f"{arch_tag}_v9_f{fold}_best.pt"
    done_flag = L / f"{arch_tag}_v9_f{fold}.done"
    if done_flag.exists() and best_dist_pt.exists():
        print(f"[9B] Skip existing {best_dist_pt.name}", flush=True)
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
        opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.03, fused=True)
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
            if epoch % 4 == 0 or epoch == s1_epochs or top1 > best_top1:
                print(f"[{arch_tag}_s1 f{fold}] ep={epoch:02d}/{s1_epochs} top1={top1:.4f} ({time.monotonic()-t0:.0f}s)", flush=True)
            if top1 > best_top1:
                best_top1 = top1
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, s1_pt)
        del model, opt, ext_imgs, ext_labs
        torch.cuda.empty_cache()

    # Stage 2: v8 Council OOF + All-Fold Battle-BP Unlabeled Soft Distillation!
    t_cand = torch.as_tensor(teacher_cand_oof.astype(np.int64), device="cuda")
    t_probs = torch.as_tensor(teacher_post_oof.astype(np.float32), device="cuda")

    ext_u_imgs = clean_impulses_gpu(torch.as_tensor(np.asarray(np.load("unl/test_32x32.npy")), device="cuda"))
    with np.load(unl_bp_pack) as ez:
        e_cand = torch.as_tensor(ez["candidates"].astype(np.int64), device="cuda")
        e_probs = torch.as_tensor(ez["posteriors"].astype(np.float32), device="cuda")

    torch.manual_seed(seed_base + 100 + fold)
    rng = np.random.default_rng(seed_base + 100 + fold)
    model = build_fn().cuda()
    model.load_state_dict(torch.load(s1_pt, map_location="cpu", weights_only=True))
    batch_size = 128
    dist_lr = 2.6e-4
    opt = torch.optim.AdamW(model.parameters(), lr=dist_lr, weight_decay=0.015, fused=True)
    steps_ep = math.ceil(len(train_idx) / batch_size)
    total_steps = steps_ep * s2_epochs
    warmup_steps = steps_ep
    best_top1 = eval_single(model, images, labels, valid_idx)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, best_dist_pt)
    t0 = time.monotonic()

    for epoch in range(1, s2_epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        for b_id, off in enumerate(range(0, len(order), batch_size)):
            idx = torch.as_tensor(order[off:off + batch_size], device="cuda")
            n_ext = round(len(idx) * 0.35)
            idx_l = idx[:-n_ext] if n_ext > 0 else idx
            x_l = augment(preprocess(images[idx_l]), 1.2, rot_deg, scale_lo, scale_hi)
            tgt_l = labels[idx_l]
            c_l, p_l = t_cand[idx_l], t_probs[idx_l]

            if n_ext > 0:
                u_pick = torch.as_tensor(rng.integers(0, len(ext_u_imgs), size=n_ext), device="cuda")
                x_u = augment(preprocess(ext_u_imgs[u_pick]), 1.2, rot_deg, scale_lo, scale_hi)
                x_all = torch.cat([x_l, x_u], dim=0)
            else:
                x_all = x_l

            step = (epoch - 1) * steps_ep + b_id
            lr = lr_at_step(step, total_steps, warmup_steps, dist_lr, 1e-6)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits_all = model(x_all).float()

            logits_l = logits_all[:len(idx_l)]
            hard = F.cross_entropy(logits_l, tgt_l, reduction="none", label_smoothing=0.01)
            soft_l = soft_ce_per_sample(logits_l, c_l, p_l)
            sw_l = class_weights[tgt_l]
            loss_l = ((0.42 * hard + 0.58 * soft_l) * sw_l).sum() / sw_l.sum()

            if n_ext > 0:
                logits_u = logits_all[len(idx_l):]
                loss_u = soft_ce_per_sample(logits_u, e_cand[u_pick], e_probs[u_pick]).mean()
                loss = 0.68 * loss_l + 0.32 * loss_u
            else:
                loss = loss_l

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        top1 = eval_single(model, images, labels, valid_idx)
        print(f"[{arch_tag}_v9 f{fold}] ep={epoch:02d}/{s2_epochs} top1={top1:.4f} ({time.monotonic()-t0:.0f}s)", flush=True)
        if top1 > best_top1:
            best_top1 = top1
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, best_dist_pt)

    done_flag.write_text(f"best_top1={best_top1:.6f}\n")
    del model, opt, ext_u_imgs, t_cand, t_probs, e_cand, e_probs
    torch.cuda.empty_cache()
    return best_dist_pt


# =============================================================================
# Zero-Perturbation Multi-Specialist Council Fusion (v9)
# =============================================================================

def fuse_grand_council_v9(
    base_post: np.ndarray,
    lp_post: np.ndarray,
    vit_post: np.ndarray | None,
    cnx_post: np.ndarray | None,
    z_hf: np.ndarray,
    z_bd: np.ndarray,
    w_lp: float, w_lp_bd: float,
    w_vit: float, w_vit_bd: float,
    w_cnx: float, w_cnx_hf: float,
    temp: float = 0.85,
) -> np.ndarray:
    """Strictly zero-perturbation for `p_max >= 0.95` (`unc == 0.000`)."""
    log_p = np.log(np.clip(base_post, 1e-8, 1.0)).astype(np.float32)
    unc = np.clip((0.95 - base_post[:, :1]) / 0.45, 0.0, 1.0).astype(np.float32)
    gate_hf = np.clip((z_hf[:, None] - 0.05) / 1.45, 0.0, 1.0).astype(np.float32)
    gate_bd = np.clip((z_bd[:, None] - 0.05) / 1.45, 0.0, 1.0).astype(np.float32)

    total_w = np.zeros((len(base_post), 1), dtype=np.float32)
    add_term = np.zeros_like(log_p)

    if w_lp > 0.0 or w_lp_bd > 0.0:
        wi = unc * (w_lp + w_lp_bd * gate_bd)
        add_term += wi * (np.log(np.clip(lp_post, 1e-8, 1.0)) / temp)
        total_w += wi
    if vit_post is not None and (w_vit > 0.0 or w_vit_bd > 0.0):
        wi = unc * (w_vit + w_vit_bd * gate_bd)
        add_term += wi * (np.log(np.clip(vit_post, 1e-8, 1.0)) / temp)
        total_w += wi
    if cnx_post is not None and (w_cnx > 0.0 or w_cnx_hf > 0.0):
        wi = unc * (w_cnx + w_cnx_hf * gate_hf)
        add_term += wi * (np.log(np.clip(cnx_post, 1e-8, 1.0)) / temp)
        total_w += wi

    # Cap total specialist blend weight at 0.36 inside uncertain tail
    cap = 0.36
    over = total_w > cap
    if np.any(over):
        scale = np.where(over, cap / np.maximum(total_w, 1e-8), 1.0)
        add_term *= scale
        total_w = np.minimum(total_w, cap)

    comb = (1.0 - total_w) * log_p + add_term
    comb -= comb.max(axis=1, keepdims=True)
    out = np.exp(comb)
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-8)
    return out.astype(np.float32)


def smoke_test() -> None:
    device = "cuda"
    x = torch.randn(8, 1, 32, 32, device=device)
    for name, fn in (("CentroidRelPosViT", CentroidRelPosViT), ("CentroidConvNeXtV2", CentroidConvNeXtV2)):
        m = fn().to(device).eval()
        n_params = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            y1 = m(x)
            y2 = m(torch.flip(x, dims=[-1]))
        diff = float((y1 - y2).abs().max().item())
        print(f"[Smoke {name}] params={n_params/1e6:.2f}M out={tuple(y1.shape)} exact_hflip_diff={diff:.2e}", flush=True)
        assert diff < 1e-5, f"{name} broke exact D2 symmetry: {diff}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.smoke_only:
        smoke_test()
        return

    t_start = time.monotonic()
    device = "cuda"
    smoke_test()

    raw_labeled_np = np.load("labeled_32x32.npy", mmap_mode="r")
    raw_unl_np = np.load("unl/test_32x32.npy", mmap_mode="r")
    raw_test_np = np.load("test_32x32.npy", mmap_mode="r")

    images = clean_impulses_gpu(torch.as_tensor(np.asarray(raw_labeled_np).copy(), device=device))
    unl_imgs = clean_impulses_gpu(torch.as_tensor(np.asarray(raw_unl_np).copy(), device=device))
    labels_np = np.load("labeled_ids.npy").astype(np.int64)
    folds_lbl = stratified_folds(labels_np, 3, SEED)
    f0_valid = np.flatnonzero(folds_lbl == 0)

    lp_paths = ensure_logpolar_all_folds_rot5(images, labels_np)
    cand_oof_all, q_oof_all, _ = build_all_folds_v7_oof(raw_labeled_np, labels_np, lp_paths)

    m2 = json.loads((L / "milestone2_pair_table_v4.json").read_text())
    nof0_tab = np.load(Path(m2["active_nof0_table"]))
    full_tab = np.load(Path(m2["active_full_table"]))
    m3 = json.loads((L / "milestone3_calibration.json").read_text())["active"]
    s0, s1, beta_lap = float(m3["s0"]), float(m3["s1"]), float(m3["beta_lap"])

    # Locked v8 Exact-Quota Debiaser config from Milestone 9A
    m9a = json.loads((L / "milestone9a_exact_quota_debias.json").read_text())["chosen"]
    kappa_9a = float(m9a["kappa"])
    alpha_t_9a = float(m9a["alpha_t"])
    debias_9a = float(m9a["debias_ep"])

    rows = pd.read_csv("data/train.csv.gz")
    identity = {p: int(v) for side in (1, 2) for p, v in zip(rows[f"noisy_image{side}"], rows[f"pokemon_id{side}"]) if pd.notna(v)}
    paths = sorted(identity)
    true = np.array([identity[p] for p in paths], dtype=np.int64)
    folds = stratified_folds(true - 1, 3, SEED)
    index = pd.Index(paths)
    f1, f2 = index.get_indexer(rows.noisy_image1), index.get_indexer(rows.noisy_image2)
    known = (f1 >= 0) & (f2 >= 0)
    f1, f2 = f1[known], f2[known]
    a, b, y = true[f1], true[f2], rows.result.to_numpy(np.float32)[known]
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
        q_base = fz["posteriors"].astype(np.float32)
        oop = fz["oop_mass"].astype(np.float32)
    with np.load(L / "rx_v1_unlabeled_f0only.npz") as uz:
        u_cand = uz["candidates"].astype(np.int64)
        u_base = uz["posteriors"].astype(np.float32)

    z_hf_f0, z_bd_f0 = compute_physical_regime_zscores(raw_labeled_np[v_idx])
    z_hf_u, z_bd_u = compute_physical_regime_zscores(raw_unl_np)
    q_lp = norm_rows(gather_cand(np.load(lp_paths[0], mmap_mode="r"), cand))
    u_lp = norm_rows(gather_cand(np.load(L / "unl_f0_logpolar_spec.npy", mmap_mode="r"), u_cand))

    um_v7_f0 = fuse_council_moe(u_base, u_lp, None, None, z_hf_u, z_bd_u, 0.05, 0.05, 0.0, 0.0, 0.0, 0.85)
    bp_f0_pack, bp_full_pack = ensure_unl_bp_packs(
        u_cand, um_v7_f0, eu, ev, uy, nof0_tab, known_allowed, full_tab, s0, s1, device=device
    )

    # 1. Train & score Fold-0 of Specialist 9B-1 (`CentroidRelPosViT`)
    vit_f0_pt = train_specialist_v9_fold(
        "centroid_vit", CentroidRelPosViT, fold=0,
        images=images, labels_np=labels_np,
        teacher_cand_oof=cand_oof_all, teacher_post_oof=q_oof_all,
        unl_bp_pack=bp_f0_pack,
        rot_deg=16.0, scale_lo=0.90, scale_hi=1.11,
        s1_epochs=24, s2_epochs=12, peak_lr=8.0e-4, seed_base=910000,
    )
    vit_f0_path = L / "centroid_vit_v9_f0_rot5.npy"
    vit_u0_path = L / "unl_f0_centroid_vit_v9.npy"
    if not vit_f0_path.exists() or not vit_u0_path.exists():
        m_vit = CentroidRelPosViT().cuda().eval()
        m_vit.load_state_dict(torch.load(vit_f0_pt, map_location="cpu", weights_only=True))
        zf_l = np.zeros(len(images), dtype=np.uint8)
        zf_u = np.zeros(len(unl_imgs), dtype=np.uint8)
        if not vit_f0_path.exists():
            p_vit = score_model(m_vit, "vit", images, f0_valid, batch_size=256, facing=zf_l)
            np.save(vit_f0_path, p_vit.astype(np.float16))
            print(f"[9B-1 CentroidViT] Fold-0 6-view TTA top1 = {float((p_vit.argmax(1)==labels_np[f0_valid]).mean()):.5f}", flush=True)
        if not vit_u0_path.exists():
            pu_vit = score_model(m_vit, "vit", unl_imgs, np.arange(len(unl_imgs)), batch_size=256, facing=zf_u)
            np.save(vit_u0_path, pu_vit.astype(np.float16))
        del m_vit
        torch.cuda.empty_cache()

    # 2. Train & score Fold-0 of Specialist 9B-2 (`CentroidConvNeXtV2`)
    cnx_f0_pt = train_specialist_v9_fold(
        "centroid_convnextv2", CentroidConvNeXtV2, fold=0,
        images=images, labels_np=labels_np,
        teacher_cand_oof=cand_oof_all, teacher_post_oof=q_oof_all,
        unl_bp_pack=bp_f0_pack,
        rot_deg=16.0, scale_lo=0.90, scale_hi=1.11,
        s1_epochs=24, s2_epochs=12, peak_lr=8.5e-4, seed_base=920000,
    )
    cnx_f0_path = L / "centroid_convnextv2_v9_f0_rot5.npy"
    cnx_u0_path = L / "unl_f0_centroid_convnextv2_v9.npy"
    if not cnx_f0_path.exists() or not cnx_u0_path.exists():
        m_cnx = CentroidConvNeXtV2().cuda().eval()
        m_cnx.load_state_dict(torch.load(cnx_f0_pt, map_location="cpu", weights_only=True))
        zf_l = np.zeros(len(images), dtype=np.uint8)
        zf_u = np.zeros(len(unl_imgs), dtype=np.uint8)
        if not cnx_f0_path.exists():
            p_cnx = score_model(m_cnx, "cnx", images, f0_valid, batch_size=256, facing=zf_l)
            np.save(cnx_f0_path, p_cnx.astype(np.float16))
            print(f"[9B-2 CentroidConvNeXtV2] Fold-0 6-view TTA top1 = {float((p_cnx.argmax(1)==labels_np[f0_valid]).mean()):.5f}", flush=True)
        if not cnx_u0_path.exists():
            pu_cnx = score_model(m_cnx, "cnx", unl_imgs, np.arange(len(unl_imgs)), batch_size=256, facing=zf_u)
            np.save(cnx_u0_path, pu_cnx.astype(np.float16))
        del m_cnx
        torch.cuda.empty_cache()

    q_vit = norm_rows(gather_cand(np.load(vit_f0_path, mmap_mode="r"), cand))
    u_vit = norm_rows(gather_cand(np.load(vit_u0_path, mmap_mode="r"), u_cand))
    q_cnx = norm_rows(gather_cand(np.load(cnx_f0_path, mmap_mode="r"), cand))
    u_cnx = norm_rows(gather_cand(np.load(cnx_u0_path, mmap_mode="r"), u_cand))

    z_lap_f0 = z_lap_from_gallery(L / "gallery_rx_top40_f0.npz", v_idx)
    pos = np.full(len(paths), -1, dtype=np.int32)
    pos[v_idx] = np.arange(len(v_idx))
    one, two = pos[f1[pair_rows]], pos[f2[pair_rows]]
    actual = y[pair_rows]
    in_c_f0 = (cand == true[v_idx, None]).astype(np.float32)

    E = len(pair_rows)
    ep_cand = np.concatenate([cand[one], cand[two]], axis=0)
    ep_oop = np.concatenate([oop[one], oop[two]], axis=0)
    ep_one = np.arange(E, dtype=np.int32)
    ep_two = np.arange(E, 2 * E, dtype=np.int32)
    ep_zl = np.concatenate([z_lap_f0[one], z_lap_f0[two]], axis=0)

    # Candidate Grand Council recipes: (w_lp, w_lp_bd, w_vit, w_vit_bd, w_cnx, w_cnx_hf, temp)
    # Index 0 is EXACT LOCKED v8!
    grid = [
        (0.05, 0.05, 0.00, 0.00, 0.00, 0.00, 0.85),  # Exact locked v8 anchor!
        (0.04, 0.04, 0.04, 0.03, 0.00, 0.00, 0.85),  # + ViT light
        (0.04, 0.04, 0.06, 0.04, 0.00, 0.00, 0.85),  # + ViT medium
        (0.04, 0.04, 0.00, 0.00, 0.04, 0.03, 0.85),  # + ConvNeXtV2 light
        (0.04, 0.04, 0.00, 0.00, 0.06, 0.04, 0.85),  # + ConvNeXtV2 medium
        (0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.85),  # Full 3-Specialist Council balanced
        (0.04, 0.04, 0.05, 0.03, 0.05, 0.03, 0.85),  # Full 3-Specialist Council strong
        (0.03, 0.04, 0.06, 0.04, 0.06, 0.04, 0.85),  # Full 3-Specialist Council assertive
        (0.04, 0.04, 0.05, 0.03, 0.05, 0.03, 0.80),  # Full 3-Specialist Council sharp
    ]

    trials = []
    for cfg_w in grid:
        w_lp, w_lp_bd, w_vit, w_vit_bd, w_cnx, w_cnx_hf, temp = cfg_w
        qm_raw = fuse_grand_council_v9(
            q_base, q_lp, q_vit, q_cnx, z_hf_f0, z_bd_f0,
            w_lp, w_lp_bd, w_vit, w_vit_bd, w_cnx, w_cnx_hf, temp,
        )
        um_raw = fuse_grand_council_v9(
            u_base, u_lp, u_vit, u_cnx, z_hf_u, z_bd_u,
            w_lp, w_lp_bd, w_vit, w_vit_bd, w_cnx, w_cnx_hf, temp,
        )
        mu_f0 = fit_smoothed_exact_quota_bias([(um_raw, u_cand, 1.0)], kappa=kappa_9a, iters=28, lock_thr=0.95)
        qm = apply_zero_perturbation_bias(qm_raw, cand, mu_f0, alpha=alpha_t_9a)

        _, _, ul0 = run_unlabeled_bp_hetero(u_cand, um_raw, eu, ev, uy, nof0_tab, known_allowed, s0=s0, s1=s1, device=device)
        _, _, tl0, _, _ = target_cavity_predict(qm, oop, cand, one, two, known_allowed, ul0, nof0_tab, debias=debias_9a, device=device)
        _, _, ul1 = run_unlabeled_bp_hetero(u_cand, um_raw, eu, ev, uy, nof0_tab, known_allowed, s0=s0, s1=s1, external_log_avail=tl0, device=device)

        qs = adaptive_power_scale(qm, np.clip(1.0 - beta_lap * z_lap_f0, 0.55, 1.45))
        ep_qs = adaptive_power_scale(np.concatenate([qm[one], qm[two]], 0), np.clip(1.0 - beta_lap * ep_zl, 0.55, 1.45))

        pr_hg, _, _, _, _ = target_cavity_predict(
            ep_qs, ep_oop, ep_cand, ep_one, ep_two, known_allowed, ul1, nof0_tab,
            debias=debias_9a, cavity_iters=12, device=device,
        )
        pr_m7, _, _, _, _ = target_cavity_predict(
            qs, oop, cand, one, two, known_allowed, ul1, nof0_tab,
            debias=debias_9a, cavity_iters=12, device=device,
        )
        rec = {
            "weights": list(cfg_w),
            "img_top1": float((cand[np.arange(len(v_idx)), qm.argmax(1)] == true[v_idx]).mean()),
            "img_brier": float(((qm - in_c_f0) ** 2).sum(1).mean()),
            "hg_tune": score_fn(rmse(pr_hg[:split], actual[:split])),
            "hg_test": score_fn(rmse(pr_hg[split:], actual[split:])),
            "m7_tune": score_fn(rmse(pr_m7[:split], actual[:split])),
            "m7_test": score_fn(rmse(pr_m7[split:], actual[split:])),
        }
        trials.append(rec)
        print(f"[v9_f0] w={rec['weights']} top1={rec['img_top1']:.5f} brier={rec['img_brier']:.5f} | "
              f"HG_tune={rec['hg_tune']:.4f} HG_test={rec['hg_test']:.4f} M7_test={rec['m7_test']:.4f}", flush=True)

    anchor = trials[0]
    brier_safe = [
        r for r in trials[1:]
        if r["img_brier"] < anchor["img_brier"] - 2e-5 and r["img_top1"] >= anchor["img_top1"] - 1e-5
    ]
    chosen = max(brier_safe, key=lambda r: r["hg_tune"]) if brier_safe else anchor
    passed = (
        chosen is not anchor
        and chosen["hg_test"] > anchor["hg_test"] + 0.005
        and chosen["m7_test"] > anchor["m7_test"] + 0.005
    )
    print(f"\n=== [Milestone 9B Gate vs v8] passed={passed} ===\n"
          f"  top1   : {anchor['img_top1']:.5f} -> {chosen['img_top1']:.5f}\n"
          f"  brier  : {anchor['img_brier']:.5f} -> {chosen['img_brier']:.5f}\n"
          f"  HG_test: {anchor['hg_test']:.4f} -> {chosen['hg_test']:.4f}\n"
          f"  M7_test: {anchor['m7_test']:.4f} -> {chosen['m7_test']:.4f}", flush=True)

    sub_csv = None
    if passed:
        w_lp, w_lp_bd, w_vit, w_vit_bd, w_cnx, w_cnx_hf, temp = chosen["weights"]
        need_vit = (w_vit > 0.0 or w_vit_bd > 0.0)
        need_cnx = (w_cnx > 0.0 or w_cnx_hf > 0.0)

        with np.load(L / "rx_v1_unlabeled.npz") as uz:
            tu_cand = uz["candidates"].astype(np.int64)
            tu_base = uz["posteriors"].astype(np.float32)
            tu_oop = uz["oop_mass"].astype(np.float32)
        with np.load(L / "rx_v1_test.npz") as tz:
            t_cand = tz["candidates"].astype(np.int64)
            t_base = tz["posteriors"].astype(np.float32)
            t_oop = tz["oop_mass"].astype(np.float32)

        test_imgs = clean_impulses_gpu(torch.as_tensor(np.asarray(raw_test_np).copy(), device=device))
        z_hf_t, z_bd_t = compute_physical_regime_zscores(raw_test_np)

        tu_lp = norm_rows(gather_cand(np.load(L / "unl_logpolar_spec.npy", mmap_mode="r"), tu_cand))
        t_lp = norm_rows(gather_cand(np.load(L / "test_logpolar_spec.npy", mmap_mode="r"), t_cand))

        tu_vit = t_vit = None
        if need_vit:
            for fold in (1, 2):
                train_specialist_v9_fold(
                    "centroid_vit", CentroidRelPosViT, fold=fold,
                    images=images, labels_np=labels_np,
                    teacher_cand_oof=cand_oof_all, teacher_post_oof=q_oof_all,
                    unl_bp_pack=bp_full_pack,
                    rot_deg=16.0, scale_lo=0.90, scale_hi=1.11,
                    s1_epochs=24, s2_epochs=12, peak_lr=8.0e-4, seed_base=910000,
                )
            u_vit_full = L / "unl_centroid_vit_v9.npy"
            t_vit_full = L / "test_centroid_vit_v9.npy"
            if not u_vit_full.exists() or not t_vit_full.exists():
                ms = []
                for f in (0, 1, 2):
                    m = CentroidRelPosViT().cuda().eval()
                    m.load_state_dict(torch.load(L / f"centroid_vit_v9_f{f}_best.pt", map_location="cpu", weights_only=True))
                    ms.append(m)
                if not u_vit_full.exists():
                    acc = sum(score_model(m, "vit", unl_imgs, np.arange(len(unl_imgs)), batch_size=256, facing=np.zeros(len(unl_imgs), dtype=np.uint8)) for m in ms) / 3.0
                    np.save(u_vit_full, acc.astype(np.float16))
                if not t_vit_full.exists():
                    acc = sum(score_model(m, "vit", test_imgs, np.arange(len(test_imgs)), batch_size=256, facing=np.zeros(len(test_imgs), dtype=np.uint8)) for m in ms) / 3.0
                    np.save(t_vit_full, acc.astype(np.float16))
                del ms
                torch.cuda.empty_cache()
            tu_vit = norm_rows(gather_cand(np.load(u_vit_full, mmap_mode="r"), tu_cand))
            t_vit = norm_rows(gather_cand(np.load(t_vit_full, mmap_mode="r"), t_cand))

        tu_cnx = t_cnx = None
        if need_cnx:
            for fold in (1, 2):
                train_specialist_v9_fold(
                    "centroid_convnextv2", CentroidConvNeXtV2, fold=fold,
                    images=images, labels_np=labels_np,
                    teacher_cand_oof=cand_oof_all, teacher_post_oof=q_oof_all,
                    unl_bp_pack=bp_full_pack,
                    rot_deg=16.0, scale_lo=0.90, scale_hi=1.11,
                    s1_epochs=24, s2_epochs=12, peak_lr=8.5e-4, seed_base=920000,
                )
            u_cnx_full = L / "unl_centroid_convnextv2_v9.npy"
            t_cnx_full = L / "test_centroid_convnextv2_v9.npy"
            if not u_cnx_full.exists() or not t_cnx_full.exists():
                ms = []
                for f in (0, 1, 2):
                    m = CentroidConvNeXtV2().cuda().eval()
                    m.load_state_dict(torch.load(L / f"centroid_convnextv2_v9_f{f}_best.pt", map_location="cpu", weights_only=True))
                    ms.append(m)
                if not u_cnx_full.exists():
                    acc = sum(score_model(m, "cnx", unl_imgs, np.arange(len(unl_imgs)), batch_size=256, facing=np.zeros(len(unl_imgs), dtype=np.uint8)) for m in ms) / 3.0
                    np.save(u_cnx_full, acc.astype(np.float16))
                if not t_cnx_full.exists():
                    acc = sum(score_model(m, "cnx", test_imgs, np.arange(len(test_imgs)), batch_size=256, facing=np.zeros(len(test_imgs), dtype=np.uint8)) for m in ms) / 3.0
                    np.save(t_cnx_full, acc.astype(np.float16))
                del ms
                torch.cuda.empty_cache()
            tu_cnx = norm_rows(gather_cand(np.load(u_cnx_full, mmap_mode="r"), tu_cand))
            t_cnx = norm_rows(gather_cand(np.load(t_cnx_full, mmap_mode="r"), t_cand))

        tu_pri_raw = fuse_grand_council_v9(
            tu_base, tu_lp, tu_vit, tu_cnx, z_hf_u, z_bd_u,
            w_lp, w_lp_bd, w_vit, w_vit_bd, w_cnx, w_cnx_hf, temp,
        )
        t_post_raw = fuse_grand_council_v9(
            t_base, t_lp, t_vit, t_cnx, z_hf_t, z_bd_t,
            w_lp, w_lp_bd, w_vit, w_vit_bd, w_cnx, w_cnx_hf, temp,
        )

        mu_v9 = fit_smoothed_exact_quota_bias([(tu_pri_raw, tu_cand, 1.0)], kappa=kappa_9a, iters=28, lock_thr=0.95)
        t_post_deb = apply_zero_perturbation_bias(t_post_raw, t_cand, mu_v9, alpha=alpha_t_9a)

        np.savez_compressed(
            L / "rx_v9_grand_council_test.npz",
            candidates=t_cand.astype(np.uint16),
            posteriors=t_post_deb.astype(np.float32),
            oop_mass=t_oop.astype(np.float32),
        )

        z_lap_test = z_lap_from_gallery(L / "gallery_rx_top40_test.npz", None)
        t_post = adaptive_power_scale(t_post_deb, np.clip(1.0 - beta_lap * z_lap_test, 0.55, 1.45))

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
            tu_cand, tu_pri_raw, eu, ev, uy, full_tab, known_full, s0=s0, s1=s1, device=device
        )
        for outer in range(2):
            pred, _, test_lnot, _, _ = target_cavity_predict(
                t_post, t_oop, t_cand, idx1, idx2, known_full, u_lnot, full_tab,
                debias=debias_9a, cavity_iters=12, use_escape=False, device=device,
            )
            if outer == 0:
                _, _, u_lnot = run_unlabeled_bp_hetero(
                    tu_cand, tu_pri_raw, eu, ev, uy, full_tab, known_full,
                    s0=s0, s1=s1, external_log_avail=test_lnot, device=device,
                )

        pred = np.clip(pred, 0.0, 10.0)
        sub_csv = L / "submission_rx_v9_grand_council.csv"
        with sub_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_id", "result"])
            for row_idx, val in enumerate(pred):
                writer.writerow([f"test-{row_idx:06d}", f"{float(val):.6f}"])

        (L / "submission_latest_best.csv").write_bytes(sub_csv.read_bytes())
        Path("submission_latest_best.csv").write_bytes(sub_csv.read_bytes())
        print(f"[9B] Milestone 9B Grand Council COMPLETE -> {sub_csv} ({time.monotonic()-t_start:.1f}s)", flush=True)

    report = {
        "anchor_v8": anchor,
        "chosen": chosen,
        "passed": passed,
        "submission_csv": str(sub_csv) if sub_csv else None,
        "trials": trials,
    }
    (L / "milestone9b_orthogonal_council_v9.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
