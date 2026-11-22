"""Milestone 6B (Discovery #3): Bilateral No-GAP Spatial Appendage Specialist
(`BilateralSpatialAppendageNet`) targeting the Twin-30 Confusion Black Hole
(where 2.9% of species cause 24.2% of all battle MSE).

Why all 17 prior models failed on Twin-30 (`#770 <-> #993`, `#524 <-> #237`, etc.):
- Global Average Pooling (`mean((-2, -1))`) collapses `8x8 -> 1x1`, diluting a
  localized 3-pixel horn/tail appendage by 1/64.
- Naive flattening breaks 50/50 horizontal-flip invariance.

Our solution (`BilateralSpatialAppendageNet`):
1. CoordConv radial/vertical channels `(y, |x|)` injected at `32x32` so conv
   filters know exact vertical height (crown/horn vs torso vs feet) and radial
   eccentricity while staying hflip-invariant.
2. Multi-scale `16x16` fine-appendage tap + `8x8` semantic map ->
   `BilateralSymmetricSpatialHead` computing BOTH `(G + flip(G))/2` AND
   `|G - flip(G)|` across the `8x4` half-grid + 8-part spatial attention
   (ZERO spatial averaging loss, EXACT mathematical hflip invariance!).
3. Training-fold-only silhouette twin upweighting (`2.2x` weight on species
   with nearest clean-template correlation `r > 0.88`) + Stage-2 Gen-2 soft
   distillation.
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
from torch.nn import functional as F

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
from mix_distill_posteriors import fuse_posteriors
from train_large_image_external import augment, lr_at_step, preprocess, soft_cross_entropy

L = Path("outputs/large")


def score_fn(val: float) -> float:
    return 100.0 * (1.0 - val / DENOMINATOR)


class SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 12):
        super().__init__()
        mid = max(ch // reduction, 16)
        self.fc1 = nn.Linear(ch, mid)
        self.fc2 = nn.Linear(mid, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.sigmoid(self.fc2(F.silu(self.fc1(x.mean(dim=(-2, -1))))))
        return x * w[:, :, None, None]


class PreActSpatialBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, drop_path: float = 0.0):
        super().__init__()
        self.drop_path = drop_path
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.se = SEBlock(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
            )
        else:
            self.short = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.bn1(x)))
        h = self.se(self.conv2(F.silu(self.bn2(h))))
        if self.training and self.drop_path > 0.0:
            keep = 1.0 - self.drop_path
            mask = torch.empty((x.size(0), 1, 1, 1), dtype=h.dtype, device=h.device).bernoulli_(keep)
            h = h * (mask / keep)
        return self.short(x) + h


class BilateralSpatialAppendageNet(nn.Module):
    """Hflip-invariant No-GAP multi-scale spatial appendage classifier."""

    def __init__(self, widths: tuple[int, int, int] = (96, 192, 320),
                 blocks_per_stage: tuple[int, int, int] = (4, 5, 5),
                 num_parts: int = 8, drop_path: float = 0.12, dropout: float = 0.20):
        super().__init__()
        ys = torch.linspace(-1.0, 1.0, 32).view(1, 1, 32, 1).expand(1, 1, 32, 32)
        xs = torch.linspace(-1.0, 1.0, 32).abs().view(1, 1, 1, 32).expand(1, 1, 32, 32)
        rr = torch.sqrt(ys * ys + xs * xs)
        self.register_buffer("coord_grid", torch.cat([ys, xs, rr], dim=1), persistent=False)

        self.stem = nn.Sequential(
            nn.Conv2d(4, widths[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.SiLU(),
        )
        total_b = sum(blocks_per_stage)
        b_idx = 0

        def make_stage(in_c: int, out_c: int, n_b: int, stride: int) -> nn.Sequential:
            nonlocal b_idx
            layers = []
            for i in range(n_b):
                dp = drop_path * b_idx / max(total_b - 1, 1)
                layers.append(PreActSpatialBlock(
                    in_c if i == 0 else out_c, out_c,
                    stride=stride if i == 0 else 1, drop_path=dp))
                b_idx += 1
            return nn.Sequential(*layers)

        self.stage1 = make_stage(widths[0], widths[0], blocks_per_stage[0], stride=1)  # 32x32
        self.stage2 = make_stage(widths[0], widths[1], blocks_per_stage[1], stride=2)  # 16x16
        self.stage3 = make_stage(widths[1], widths[2], blocks_per_stage[2], stride=2)  # 8x8

        # Preserve 16x16 local appendage detail via max+avg dual tap into 8x8
        self.hi_tap = nn.Sequential(
            nn.BatchNorm2d(widths[1]),
            nn.SiLU(),
            nn.Conv2d(widths[1], 48, 1, bias=False),
        )
        self.lo_norm = nn.Sequential(
            nn.BatchNorm2d(widths[2]),
            nn.SiLU(),
        )
        comb_c = widths[2] + 96
        self.grid_proj = nn.Sequential(
            nn.Conv2d(comb_c, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        # 8 learned spatial part detectors (horns, ears, tail, limbs...)
        self.part_queries = nn.Conv2d(comb_c, num_parts, 1, bias=True)
        self.part_val = nn.Conv2d(comb_c, 96, 1, bias=False)

        # Descriptor across ALL 8x8 = 64 cells (exact D2 symmetry ga <-> gb!):
        # - Sym bilateral grid (ga + gb): 32 * 8 * 8 = 2048
        # - Asym appendage grid |ga - gb|: 32 * 8 * 8 = 2048
        # - Part-pooled descriptors: 8 * 96 = 768
        # - Global max + avg context: 2 * widths[2] = 640
        feat_dim = 2048 + 2048 + (num_parts * 96) + (2 * widths[2])
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1536),
            nn.GELU(),
            nn.LayerNorm(1536),
            nn.Dropout(dropout * 0.5),
            nn.Linear(1536, NUM_CLASSES),
        )

    def _features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coords = self.coord_grid.expand(x.size(0), -1, -1, -1).to(dtype=x.dtype)
        h1 = self.stage1(self.stem(torch.cat([x, coords], dim=1)))
        h2 = self.stage2(h1)  # (B, C2, 16, 16)
        h3 = self.lo_norm(self.stage3(h2))  # (B, C3, 8, 8)
        hi = self.hi_tap(h2)  # (B, 48, 16, 16)
        comb = torch.cat([h3, F.max_pool2d(hi, 2, 2), F.avg_pool2d(hi, 2, 2)], dim=1)
        return comb, h3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        comb_a, h3_a = self._features(x)
        comb_b, h3_b = self._features(torch.flip(x, dims=[-1]))

        ga = self.grid_proj(comb_a)
        gb = self.grid_proj(comb_b)
        # Under x -> flip(x), comb_a <-> comb_b and ga <-> gb, so both
        # 0.5*(ga + gb) and |ga - gb| across ALL 8x8 cells are EXACTLY invariant!
        g_sym = (0.5 * (ga + gb)).flatten(1)
        g_asym = (ga - gb).abs().flatten(1)

        attn_a = torch.softmax(self.part_queries(comb_a).flatten(2), dim=-1)
        vals_a = self.part_val(comb_a).flatten(2)
        attn_b = torch.softmax(self.part_queries(comb_b).flatten(2), dim=-1)
        vals_b = self.part_val(comb_b).flatten(2)
        parts = 0.5 * (torch.bmm(attn_a, vals_a.transpose(1, 2))
                       + torch.bmm(attn_b, vals_b.transpose(1, 2))).flatten(1)

        ctx = 0.5 * (
            torch.cat([h3_a.mean(dim=(-2, -1)), h3_a.amax(dim=(-2, -1))], dim=1)
            + torch.cat([h3_b.mean(dim=(-2, -1)), h3_b.amax(dim=(-2, -1))], dim=1)
        )
        feat = torch.cat([g_sym, g_asym, parts, ctx], dim=1)
        return self.classifier(feat)


@torch.no_grad()
def compute_train_twin_weights(images: torch.Tensor, labels_np: np.ndarray,
                               train_idx: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """Compute hflip-canonicalized class templates strictly on `train_idx` and
    upweight classes that have a near-identical visual twin (r > 0.88).
    """
    templates = torch.zeros((NUM_CLASSES, 1024), device="cuda", dtype=torch.float32)
    counts = torch.zeros(NUM_CLASSES, device="cuda", dtype=torch.float32)
    for start in range(0, len(train_idx), 2048):
        idx = torch.as_tensor(train_idx[start:start + 2048], device="cuda")
        x = preprocess(images[idx]).float()  # (B, 1, 32, 32)
        # Canonicalize left-vs-right center of mass so templates align!
        left_mass = x[:, 0, :, :16].abs().sum(dim=(-2, -1))
        right_mass = x[:, 0, :, 16:].abs().sum(dim=(-2, -1))
        xf = torch.where((right_mass > left_mass)[:, None, None, None], torch.flip(x, dims=[-1]), x)
        flat = xf.reshape(len(idx), 1024)
        tgt = torch.as_tensor(labels_np[train_idx[start:start + 2048]], device="cuda", dtype=torch.long)
        templates.index_add_(0, tgt, flat)
        counts.index_add_(0, tgt, torch.ones_like(tgt, dtype=torch.float32))

    templates = templates / counts.clamp_min(1.0)[:, None]
    templates = templates - templates.mean(dim=1, keepdim=True)
    templates = F.normalize(templates, dim=1)
    sim = templates @ templates.T
    sim.fill_diagonal_(-1.0)
    max_sim, nearest_twin = sim.max(dim=1)
    # Continuous boost from 1.0 up to 2.4x for r >= 0.86 twin species
    boost = torch.clamp((max_sim - 0.85) / 0.12, 0.0, 1.0)
    class_weights = 1.0 + 1.4 * boost
    n_twins = int((max_sim > 0.88).sum().item())
    print(f"[twin_mining] classes with twin r>0.88: {n_twins}, max_r={float(max_sim.max()):.4f}",
          flush=True)
    return class_weights, max_sim.cpu().numpy()


@torch.inference_mode()
def evaluate_single(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
                    valid_idx: np.ndarray, batch_size: int = 256) -> dict:
    model.eval()
    c1 = c5 = 0
    loss_sum = 0.0
    for start in range(0, len(valid_idx), batch_size):
        idx = torch.as_tensor(valid_idx[start:start + batch_size], device=images.device)
        x = preprocess(images[idx])
        target = labels[idx]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).float()
        c1 += int((logits.argmax(1) == target).sum().item())
        c5 += int((logits.topk(5, dim=1).indices == target[:, None]).any(1).sum().item())
        loss_sum += float(F.cross_entropy(logits, target, reduction="sum").item())
    n = len(valid_idx)
    return {"top1": c1 / n, "top5": c5 / n, "ce": loss_sum / n}


def train_spatial_fold(fold: int, images: torch.Tensor, labels_np: np.ndarray,
                       seed_base: int = 616161) -> Path:
    best_dist_pt = L / f"spatial_twin_gen2_f{fold}_best.pt"
    if best_dist_pt.exists():
        print(f"skip existing {best_dist_pt.name}", flush=True)
        return best_dist_pt

    folds = stratified_folds(labels_np, 3, SEED)
    valid_idx = np.flatnonzero(folds == fold)
    train_idx = np.flatnonzero(folds != fold)
    labels = torch.as_tensor(labels_np, device="cuda", dtype=torch.long)
    class_weights, _ = compute_train_twin_weights(images, labels_np, train_idx)

    s1_pt = L / f"spatial_twin_s1_f{fold}_best.pt"
    if not s1_pt.exists():
        with np.load(L / f"transductive_bank_f{fold}.npz") as bk:
            ext_imgs = clean_impulses_gpu(torch.as_tensor(bk["images"], device="cuda"))
            ext_labs = torch.as_tensor(bk["labels"].astype(np.int64), device="cuda")

        torch.manual_seed(seed_base + fold)
        rng = np.random.default_rng(seed_base + fold)
        model = BilateralSpatialAppendageNet().cuda()
        batch_size = 128
        epochs = 26
        peak_lr = 9.0e-4
        opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.025, fused=True)
        steps_ep = math.ceil(len(train_idx) / batch_size)
        total_steps = steps_ep * epochs
        warmup_steps = steps_ep * 2
        best_top1 = -1.0
        best_ep = 0
        t0 = time.monotonic()

        for epoch in range(1, epochs + 1):
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
                # Milder affine jitter so 2-3px horns are not interpolated out of existence!
                x = augment(preprocess(raw), 1.0, 8.0, 0.94, 1.06)
                step = (epoch - 1) * steps_ep + b_id
                lr = lr_at_step(step, total_steps, warmup_steps, peak_lr, 1e-6)
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x).float()
                per_sample = F.cross_entropy(logits, tgt, reduction="none", label_smoothing=0.02)
                sample_w = class_weights[tgt]
                loss = (per_sample * sample_w).sum() / sample_w.sum()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            m = evaluate_single(model, images, labels, valid_idx)
            print(f"[spatial_s1 f{fold}] ep={epoch:02d} top1={m['top1']:.4f} "
                  f"top5={m['top5']:.4f} ({time.monotonic()-t0:.0f}s)", flush=True)
            if m["top1"] > best_top1 + 0.0002:
                best_top1 = m["top1"]
                best_ep = epoch
                torch.save(model.state_dict(), s1_pt)
            if epoch >= 16 and epoch - best_ep >= 7:
                break
        del model, ext_imgs, ext_labs
        torch.cuda.empty_cache()

    # Stage 2: 12-epoch Gen-2 soft distillation with twin emphasis
    with np.load(L / "gen2_distill_labeled_oof.npz") as tz:
        lab_cand = torch.as_tensor(tz["candidates"].astype(np.int64), device="cuda")
        lab_q = torch.as_tensor(tz["posteriors"].astype(np.float32), device="cuda")

    torch.manual_seed(seed_base + 100 + fold)
    rng = np.random.default_rng(seed_base + 100 + fold)
    model = BilateralSpatialAppendageNet().cuda()
    model.load_state_dict(torch.load(s1_pt, map_location="cpu", weights_only=True))
    batch_size = 128
    epochs = 12
    peak_lr = 1.2e-4
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.02, fused=True)
    steps_ep = math.ceil(len(train_idx) / batch_size)
    total_steps = steps_ep * epochs
    warmup_steps = steps_ep
    best_top1 = -1.0
    t0 = time.monotonic()

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        for b_id, off in enumerate(range(0, len(order), batch_size)):
            idx = torch.as_tensor(order[off:off + batch_size], device="cuda")
            x = augment(preprocess(images[idx]), 0.8, 6.0, 0.96, 1.04)
            tgt = labels[idx]
            step = (epoch - 1) * steps_ep + b_id
            lr = lr_at_step(step, total_steps, warmup_steps, peak_lr, 1e-6)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x).float()
            hard_vec = F.cross_entropy(logits, tgt, reduction="none", label_smoothing=0.01)
            sw = class_weights[tgt]
            hard_loss = (hard_vec * sw).sum() / sw.sum()
            soft_loss = soft_cross_entropy(logits, lab_cand[idx], lab_q[idx])
            loss = 0.45 * hard_loss + 0.55 * soft_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        m = evaluate_single(model, images, labels, valid_idx)
        print(f"[spatial_gen2 f{fold}] ep={epoch:02d} top1={m['top1']:.4f} "
              f"top5={m['top5']:.4f} ({time.monotonic()-t0:.0f}s)", flush=True)
        if m["top1"] > best_top1 + 0.0001:
            best_top1 = m["top1"]
            torch.save(model.state_dict(), best_dist_pt)
            (L / f"spatial_twin_gen2_f{fold}_best.json").write_text(
                json.dumps({"epoch": epoch, **m}, indent=2) + "\n")

    del model, lab_cand, lab_q
    torch.cuda.empty_cache()
    return best_dist_pt


def adaptive_specialist_fuse(base_post: np.ndarray, spec_post: np.ndarray,
                             w_global: float, w_ambig: float,
                             temp: float = 0.85) -> np.ndarray:
    """Fuse specialist stronger on ambiguous / twin-contested samples where
    `base_post[:, 0] < 0.97`!
    """
    base_f = np.clip(base_post.astype(np.float64), 1e-12, 1.0)
    spec_f = np.clip(spec_post.astype(np.float64), 1e-12, 1.0)
    # Uncertainty factor in [0, 1]: 0 when top1=1.0, rises smoothly when top1 < 0.97
    unc = np.clip((0.97 - base_f[:, :1]) / 0.47, 0.0, 1.0)
    alpha = np.clip(w_global + w_ambig * unc, 0.0, 0.45)
    log_p = (1.0 - alpha) * np.log(base_f) + alpha * (np.log(spec_f) / max(temp, 1e-3))
    log_p -= log_p.max(axis=1, keepdims=True)
    p = np.exp(log_p)
    return (p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def pick_best_current_tables_and_packs() -> tuple[np.ndarray, np.ndarray, Path, Path, Path, Path]:
    # Prefer PairTable v5 if Milestone 6A passed, else PairTable v4
    m6a = L / "milestone6a_table_v5.json"
    if m6a.exists() and json.loads(m6a.read_text()).get("passed"):
        nof0_tab = np.load(L / "pair_v5_nofold0_blend_table.npy")
        full_tab = np.load(L / "pair_v5_full_blend_table.npy")
    else:
        m2 = json.loads((L / "milestone2_pair_table_v4.json").read_text())
        nof0_tab = np.load(Path(m2["active_nof0_table"]))
        full_tab = np.load(Path(m2["active_full_table"]))

    # Prefer Milestone 4 rx_v4 packs if Milestone 4 passed, else rx_v1
    m4 = L / "milestone4_gen2_distill.json"
    if m4.exists() and json.loads(m4.read_text()).get("passed") and (L / "rx_v4_test.npz").exists():
        t_pack = L / "rx_v4_test.npz"
        u_pack = L / "rx_v4_unlabeled.npz"
    else:
        t_pack = L / "rx_v1_test.npz"
        u_pack = L / "rx_v1_unlabeled.npz"

    return nof0_tab, full_tab, L / "rx_v1_fold0_scores.npz", L / "rx_v1_unlabeled_f0only.npz", u_pack, t_pack


def main() -> None:
    device = "cuda"
    images = clean_impulses_gpu(torch.as_tensor(np.asarray(np.load("labeled_32x32.npy")), device=device))
    labels_np = np.load("labeled_ids.npy").astype(np.int64)
    folds = stratified_folds(labels_np, 3, SEED)
    f0_valid = np.flatnonzero(folds == 0)

    # 1. Train Fold 0 & export 6-view TTA
    f0_pt = train_spatial_fold(0, images, labels_np)
    f0_probs_path = L / "spatial_twin_f0_rot5.npy"
    if not f0_probs_path.exists():
        m0 = BilateralSpatialAppendageNet().cuda()
        m0.load_state_dict(torch.load(f0_pt, map_location="cpu", weights_only=True))
        m0.eval()
        p0 = score_model(m0, "spatial_twin", images, f0_valid, batch_size=256,
                         facing=np.zeros(len(f0_valid), dtype=np.uint8))
        np.save(f0_probs_path, p0.astype(np.float16))
        del m0
        torch.cuda.empty_cache()

    p0_arr = np.load(f0_probs_path, mmap_mode="r")
    f0_top1 = float((p0_arr.argmax(1) == labels_np[f0_valid]).mean())
    print(f"[spatial_twin] Fold-0 6-view TTA top1 = {f0_top1:.5f}", flush=True)

    unl_imgs = clean_impulses_gpu(torch.as_tensor(np.asarray(np.load("unl/test_32x32.npy")), device=device))
    u0_probs_path = L / "unl_f0_spatial_twin.npy"
    if not u0_probs_path.exists():
        m0 = BilateralSpatialAppendageNet().cuda()
        m0.load_state_dict(torch.load(f0_pt, map_location="cpu", weights_only=True))
        m0.eval()
        pu0 = score_model(m0, "spatial_twin", unl_imgs, np.arange(len(unl_imgs)), batch_size=256,
                          facing=np.zeros(len(unl_imgs), dtype=np.uint8))
        np.save(u0_probs_path, pu0.astype(np.float16))
        del m0
        torch.cuda.empty_cache()

    # 2. Leak-free Fold-0 gate (testing BOTH uniform fusion and uncertainty-gated twin fusion!)
    nof0_tab, full_tab, f0_pack, uf0_pack, full_u_pack, full_t_pack = pick_best_current_tables_and_packs()
    m3 = json.loads((L / "milestone3_calibration.json").read_text())["active"]
    s0, s1 = float(m3["s0"]), float(m3["s1"])
    debias = float(m3["debias"])
    beta_lap = float(m3["beta_lap"])

    rows = pd.read_csv("data/train.csv.gz")
    identity = {}
    for side in (1, 2):
        for p, v in zip(rows[f"noisy_image{side}"], rows[f"pokemon_id{side}"]):
            if pd.notna(v):
                identity[p] = int(v)
    paths = sorted(identity)
    true = np.array([identity[p] for p in paths], dtype=np.int64)
    folds_p = stratified_folds(true - 1, 3, SEED)
    index = pd.Index(paths)
    f1, f2 = index.get_indexer(rows.noisy_image1), index.get_indexer(rows.noisy_image2)
    known = (f1 >= 0) & (f2 >= 0)
    f1, f2 = f1[known], f2[known]
    a, b = true[f1], true[f2]
    y = rows.result.to_numpy(np.float32)[known]
    pair_f0 = (folds_p[f1] == 0) & (folds_p[f2] == 0)
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

    with np.load(uf0_pack) as uz:
        u_cand = uz["candidates"].astype(np.int64)
        u_base = uz["posteriors"].astype(np.float32)
    with np.load(f0_pack) as fz:
        v_idx = fz["valid_indices"].astype(np.int64)
        cand = fz["candidates"].astype(np.int64)
        q_base = fz["posteriors"].astype(np.float32)
        oop = fz["oop_mass"].astype(np.float32)

    z_lap_f0 = z_lap_from_gallery(L / "gallery_rx_top40_f0.npz", v_idx)
    pos = np.full(len(paths), -1, dtype=np.int32)
    pos[v_idx] = np.arange(len(v_idx))
    one, two = pos[f1[pair_rows]], pos[f2[pair_rows]]
    actual = y[pair_rows]
    ta, tb = a[pair_rows], b[pair_rows]

    q_sp = gather_cand(np.load(f0_probs_path, mmap_mode="r"), cand)
    q_sp /= np.maximum(q_sp.sum(1, keepdims=True), 1e-8)
    u_sp = gather_cand(np.load(u0_probs_path, mmap_mode="r"), u_cand)
    u_sp /= np.maximum(u_sp.sum(1, keepdims=True), 1e-8)

    configs = [
        (0.00, 0.00, 0.85),
        (0.05, 0.00, 0.85),
        (0.08, 0.00, 0.85),
        (0.12, 0.00, 0.85),
        (0.04, 0.10, 0.85),
        (0.06, 0.14, 0.85),
        (0.08, 0.16, 0.80),
    ]
    trials = []
    for wg, wa, tmp in configs:
        q_mix = q_base if (wg == 0.0 and wa == 0.0) else adaptive_specialist_fuse(q_base, q_sp, wg, wa, tmp)
        u_mix = u_base if (wg == 0.0 and wa == 0.0) else adaptive_specialist_fuse(u_base, u_sp, wg, wa, tmp)
        _, _, unl_lnot0 = run_unlabeled_bp_hetero(
            u_cand, u_mix, eu, ev, uy, nof0_tab, known_allowed, s0=s0, s1=s1, device=device)
        _, _, test_lnot, _, _ = target_cavity_predict(
            q_mix, oop, cand, one, two, known_allowed, unl_lnot0, nof0_tab,
            debias=debias, device=device)
        _, _, unl_lnot = run_unlabeled_bp_hetero(
            u_cand, u_mix, eu, ev, uy, nof0_tab, known_allowed,
            s0=s0, s1=s1, external_log_avail=test_lnot, device=device)

        best_s = None
        for sc in (0.90, 0.95, 1.00):
            s_vec = np.clip(sc - beta_lap * z_lap_f0, 0.55, 1.45)
            qs = adaptive_power_scale(q_mix, s_vec)
            pred, top_p, _, _, _ = target_cavity_predict(
                qs, oop, cand, one, two, known_allowed, unl_lnot, nof0_tab,
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
        rec = {"w_global": wg, "w_ambig": wa, "temp": tmp, "selected": best_s}
        trials.append(rec)
        print(f"[spatial_f0] wg={wg:.2f} wa={wa:.2f} T={tmp:.2f} "
              f"tune={best_s['tune_score']:.4f} test={best_s['test_score']:.4f} "
              f"both={best_s['test_both_correct']:.5f}", flush=True)

    anchor = next(r for r in trials if r["w_global"] == 0.0 and r["w_ambig"] == 0.0)
    chosen = min(trials, key=lambda r: (r["selected"]["tune_rmse"], r["w_global"] + r["w_ambig"]))
    passed = ((chosen["w_global"] > 0.0 or chosen["w_ambig"] > 0.0)
              and chosen["selected"]["test_score"] > anchor["selected"]["test_score"] + 0.005)
    print(f"Milestone 6B gate: passed={passed} "
          f"(test {anchor['selected']['test_score']:.4f} -> {chosen['selected']['test_score']:.4f})",
          flush=True)

    sub_csv = None
    if passed:
        pts = [f0_pt]
        for fold in (1, 2):
            pts.append(train_spatial_fold(fold, images, labels_np))

        oof_sp_path = L / "oof_spatial_twin.npy"
        if not oof_sp_path.exists():
            oof_arr = np.empty((len(labels_np), NUM_CLASSES), dtype=np.float16)
            oof_arr[f0_valid] = np.load(f0_probs_path)
            for fold in (1, 2):
                fv = np.flatnonzero(folds == fold)
                mf = BilateralSpatialAppendageNet().cuda()
                mf.load_state_dict(torch.load(pts[fold], map_location="cpu", weights_only=True))
                mf.eval()
                oof_arr[fv] = score_model(mf, "spatial_twin", images, fv, batch_size=256,
                                          facing=np.zeros(len(fv), dtype=np.uint8)).astype(np.float16)
                del mf
                torch.cuda.empty_cache()
            np.save(oof_sp_path, oof_arr)

        models = []
        for pt in pts:
            m = BilateralSpatialAppendageNet().cuda()
            m.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
            m.eval()
            models.append(m)

        unl_sp_path = L / "unl_spatial_twin.npy"
        if not unl_sp_path.exists():
            acc = np.zeros((len(unl_imgs), NUM_CLASSES), dtype=np.float32)
            uz_f = np.zeros(len(unl_imgs), dtype=np.uint8)
            for m in models:
                acc += score_model(m, "spatial_twin", unl_imgs, np.arange(len(unl_imgs)), batch_size=256, facing=uz_f)
            acc /= len(models)
            np.save(unl_sp_path, acc.astype(np.float16))

        test_imgs = clean_impulses_gpu(torch.as_tensor(np.asarray(np.load("test_32x32.npy")), device=device))
        test_sp_path = L / "test_spatial_twin.npy"
        if not test_sp_path.exists():
            acc = np.zeros((len(test_imgs), NUM_CLASSES), dtype=np.float32)
            tz_f = np.zeros(len(test_imgs), dtype=np.uint8)
            for m in models:
                acc += score_model(m, "spatial_twin", test_imgs, np.arange(len(test_imgs)), batch_size=256, facing=tz_f)
            acc /= len(models)
            np.save(test_sp_path, acc.astype(np.float16))
        del models, test_imgs
        torch.cuda.empty_cache()

        wg, wa, tmp = chosen["w_global"], chosen["w_ambig"], chosen["temp"]
        sc_win = chosen["selected"]["scale_base"]

        with np.load(full_u_pack) as uz:
            tu_cand = uz["candidates"].astype(np.int64)
            tu_base = uz["posteriors"].astype(np.float32)
            tu_oop = uz["oop_mass"].astype(np.float32)
        with np.load(full_t_pack) as tz:
            t_cand = tz["candidates"].astype(np.int64)
            t_base = tz["posteriors"].astype(np.float32)
            t_oop = tz["oop_mass"].astype(np.float32)

        tu_sp = gather_cand(np.load(unl_sp_path, mmap_mode="r"), tu_cand)
        tu_sp /= np.maximum(tu_sp.sum(1, keepdims=True), 1e-8)
        t_sp = gather_cand(np.load(test_sp_path, mmap_mode="r"), t_cand)
        t_sp /= np.maximum(t_sp.sum(1, keepdims=True), 1e-8)

        tu_pri = adaptive_specialist_fuse(tu_base, tu_sp, wg, wa, tmp)
        t_post_raw = adaptive_specialist_fuse(t_base, t_sp, wg, wa, tmp)

        np.savez_compressed(L / "rx_v6b_unlabeled.npz",
                            candidates=tu_cand.astype(np.uint16),
                            posteriors=tu_pri.astype(np.float32),
                            oop_mass=tu_oop)
        np.savez_compressed(L / "rx_v6b_test.npz",
                            candidates=t_cand.astype(np.uint16),
                            posteriors=t_post_raw.astype(np.float32),
                            oop_mass=t_oop)

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
        sub_csv = L / "submission_rx_v6b_twin_spatial.csv"
        with sub_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_id", "result"])
            for row_idx, val in enumerate(pred):
                writer.writerow([f"test-{row_idx:06d}", f"{float(val):.6f}"])
        (L / "submission_latest_best.csv").write_bytes(sub_csv.read_bytes())
        print(f"Milestone 6B complete -> {sub_csv}", flush=True)

    report = {
        "fold0_rot5_top1": f0_top1,
        "anchor": anchor,
        "chosen": chosen,
        "passed": passed,
        "submission_csv": str(sub_csv) if sub_csv else None,
        "trials": trials,
    }
    (L / "milestone6b_spatial_twin.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
