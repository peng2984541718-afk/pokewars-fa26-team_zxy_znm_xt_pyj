"""Within-species correlation after a small rigid alignment.

Tests the claim that each species is one mother sprite. Twenty random species,
the ten cleanest images of each, Pearson correlation before and after
(hflip, dx/dy in [-2, 2], angle in [-8, 8] at 1 degree). A same-image
2px/5-degree warp is recovered with the same search so the ceiling is known.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT = Path("outputs/large/sprite_alignment_probe.json")


def impulse_rate(images: np.ndarray) -> np.ndarray:
    rates = np.empty(len(images), dtype=np.float32)
    for start in range(0, len(images), 8192):
        chunk = np.asarray(images[start:start + 8192])
        rates[start:start + len(chunk)] = np.mean(
            (chunk == 25) | (chunk == 245), axis=(1, 2))
    return rates


def transform_grid() -> tuple[np.ndarray, int]:
    flips, angles, dxs, dys = [], [], [], []
    for flip in (0, 1):
        for angle in range(-8, 9):
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    flips.append(flip)
                    angles.append(angle)
                    dxs.append(dx)
                    dys.append(dy)
    grid = np.stack([flips, angles, dxs, dys], axis=1).astype(np.float32)
    identity = int(np.where((grid == (0, 0, 0, 0)).all(1))[0][0])
    return grid, identity


def warp(images: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """images (N, 1, 32, 32), grid rows (flip, angle_deg, dx, dy) -> (N, T, 32, 32)."""
    n = images.shape[0]
    count = grid.shape[0]
    flip, angle, dx, dy = grid.T
    angle = torch.deg2rad(angle)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    sign = torch.where(flip > 0.5, -torch.ones_like(cosine), torch.ones_like(cosine))
    theta = torch.zeros(count, 2, 3, device=images.device)
    theta[:, 0, 0] = cosine * sign
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine * sign
    theta[:, 1, 1] = cosine
    theta[:, 0, 2] = dx * (2 / 32)
    theta[:, 1, 2] = dy * (2 / 32)
    theta = theta.repeat(n, 1, 1)
    source = images.repeat_interleave(count, dim=0)
    sampled = F.grid_sample(
        source, F.affine_grid(theta, source.shape, align_corners=False),
        align_corners=False, padding_mode="border")
    return sampled[:, 0].reshape(n, count, -1)


def pearson_against(reference: torch.Tensor, candidates: torch.Tensor,
                    reference_mask: torch.Tensor | None = None,
                    candidate_mask: torch.Tensor | None = None) -> torch.Tensor:
    """reference (N, P), candidates (N, T, P) -> (N, T)."""
    if reference_mask is None:
        reference = reference - reference.mean(-1, keepdim=True)
        candidates = candidates - candidates.mean(-1, keepdim=True)
        numer = (candidates * reference[:, None, :]).sum(-1)
        denom = (candidates.square().sum(-1).sqrt()
                 * reference.square().sum(-1).sqrt()[:, None]).clamp_min(1e-8)
        return numer / denom
    mask = candidate_mask * reference_mask[:, None, :]
    weight = mask.sum(-1).clamp_min(1)
    ref_mean = (reference[:, None, :] * mask).sum(-1) / weight
    cand_mean = (candidates * mask).sum(-1) / weight
    ref_c = (reference[:, None, :] - ref_mean[..., None]) * mask
    cand_c = (candidates - cand_mean[..., None]) * mask
    numer = (ref_c * cand_c).sum(-1)
    denom = (ref_c.square().sum(-1).sqrt() * cand_c.square().sum(-1).sqrt()).clamp_min(1e-8)
    return numer / denom


def summarize(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "fraction_at_least_0.85": float(np.mean(values >= 0.85)),
        "fraction_at_least_0.70": float(np.mean(values >= 0.70)),
    }


def main() -> None:
    torch.set_float32_matmul_precision("high")
    images = np.load("labeled_32x32.npy", mmap_mode="r")
    labels = np.load("labeled_ids.npy").astype(np.int64)
    rates = impulse_rate(images)
    rng = np.random.default_rng(20261119)
    eligible = [c for c in range(1025) if np.count_nonzero(labels == c) >= 30]
    species = np.sort(rng.choice(eligible, size=20, replace=False))
    grid_np, identity = transform_grid()
    grid = torch.as_tensor(grid_np, device="cuda")

    raw_best, raw_plain, masked_best = [], [], []
    per_species = []
    for species_id in species:
        pool = np.flatnonzero(labels == species_id)
        chosen = pool[np.argsort(rates[pool])[:10]]
        raw = torch.as_tensor(np.asarray(images[chosen]), device="cuda", dtype=torch.float32)
        valid = ((raw != 25) & (raw != 245)).float()
        flat = raw.reshape(10, -1)
        mask = valid.reshape(10, -1)
        warped = warp(raw.unsqueeze(1), grid)
        warped_mask = warp(valid.unsqueeze(1), grid) > 0.999
        plain, best, best_masked = [], [], []
        for left in range(10):
            right = [j for j in range(10) if j != left]
            reference = flat[left]
            candidates = warped[right]
            score = pearson_against(reference.expand(len(right), -1), candidates)
            masked = pearson_against(
                reference.expand(len(right), -1), candidates,
                mask[left].expand(len(right), -1), warped_mask[right])
            plain.append(score[:, identity])
            best.append(score.max(1).values)
            best_masked.append(masked.max(1).values)
        plain = torch.cat(plain).cpu().numpy()
        best = torch.cat(best).cpu().numpy()
        best_masked = torch.cat(best_masked).cpu().numpy()
        # Each unordered pair is scored twice (each image as the reference).
        raw_plain.append(plain)
        raw_best.append(best)
        masked_best.append(best_masked)
        per_species.append({
            "species": int(species_id + 1),
            "clean_impulse_rate_max": float(rates[chosen].max()),
            "unaligned_median": float(np.median(plain)),
            "aligned_median": float(np.median(best)),
            "aligned_masked_median": float(np.median(best_masked)),
        })
        print(f"species {species_id + 1:4d} unaligned {np.median(plain):.3f} "
              f"aligned {np.median(best):.3f} masked {np.median(best_masked):.3f}",
              flush=True)

    cleanest = []
    for species_id in species:
        pool = np.flatnonzero(labels == species_id)
        cleanest.append(pool[np.argmin(rates[pool])])
    base = torch.as_tensor(np.asarray(images[cleanest]), device="cuda", dtype=torch.float32).unsqueeze(1)
    known = torch.tensor([[0, 5, 2, -1]], device="cuda", dtype=torch.float32)
    shifted = warp(base, known).reshape(len(species), 1, 32, 32)
    original = base[:, 0].reshape(len(species), -1)
    recovered = warp(shifted, grid)
    shifted = shifted[:, 0].reshape(len(species), -1)
    shifted_corr = pearson_against(original, shifted[:, None, :]).squeeze(1)
    recovered_corr = pearson_against(original, recovered).max(1).values

    cross_raw = base
    cross_warped = warp(cross_raw, grid)
    cross_flat = cross_raw[:, 0].reshape(len(species), -1)
    cross_plain, cross_best = [], []
    for left in range(len(species)):
        right = [j for j in range(len(species)) if j != left]
        score = pearson_against(cross_flat[left].expand(len(right), -1), cross_warped[right])
        cross_plain.append(score[:, identity])
        cross_best.append(score.max(1).values)
    cross_plain = torch.cat(cross_plain).cpu().numpy()
    cross_best = torch.cat(cross_best).cpu().numpy()

    report = {
        "protocol": (
            "20 species with at least 30 labeled images, seed 20261119. "
            "Ten lowest-impulse images per species. Pearson on all 32x32 pixels, "
            "and a second score that ignores pixels that are exactly 25 or 245 "
            "in both images. Search is hflip x angle[-8,8] step 1 x shift[-2,2]."
        ),
        "within_species_unaligned": summarize(np.concatenate(raw_plain)),
        "within_species_aligned": summarize(np.concatenate(raw_best)),
        "within_species_aligned_impulse_masked": summarize(np.concatenate(masked_best)),
        "between_species_unaligned": summarize(cross_plain),
        "between_species_aligned": summarize(cross_best),
        "same_image_shift2_rot5": {
            "median_before_search": float(shifted_corr.median()),
            "median_after_search": float(recovered_corr.median()),
            "min_after_search": float(recovered_corr.min()),
        },
        "per_species": per_species,
        "decision": None,
    }
    aligned = report["within_species_aligned"]["p50"]
    if aligned >= 0.85:
        report["decision"] = "mother sprite plausible: aligned within-species median is at least 0.85"
    elif aligned >= 0.70:
        report["decision"] = "borderline: aligned within-species median is between 0.70 and 0.85"
    else:
        report["decision"] = "not a single mother sprite: aligned within-species median stays below 0.70"
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "per_species"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
