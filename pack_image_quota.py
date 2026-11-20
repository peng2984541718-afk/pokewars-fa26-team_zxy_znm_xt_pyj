"""Sinkhorn image-class quotas on ranker posteriors.

Fold-0 labeled images are exactly 53 per species (1025 * 53 = 54325).
Test images average 160000 / 1025 = 156.09756 per species.
Unlabeled priors are left alone; unlabeled BP already targets 67.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

L = Path("outputs/large")


def sinkhorn_quota(cand: np.ndarray, post: np.ndarray, target: np.ndarray,
                   iters: int = 25) -> tuple[np.ndarray, np.ndarray]:
    q = np.clip(post.astype(np.float64), 1e-12, 1.0)
    q /= q.sum(1, keepdims=True)
    tgt = np.zeros(1026, dtype=np.float64)
    tgt[1:] = np.asarray(target, dtype=np.float64)
    mass = np.zeros(1026, dtype=np.float64)
    for _ in range(iters):
        mass = np.bincount(cand.ravel(), weights=q.ravel(), minlength=1026).astype(np.float64)
        mass[0] = 1.0
        scale = tgt / np.maximum(mass, 1e-8)
        scale[0] = 1.0
        scale = np.clip(scale, 0.05, 20.0)
        q *= scale[cand]
        q /= np.maximum(q.sum(1, keepdims=True), 1e-12)
    mass = np.bincount(cand.ravel(), weights=q.ravel(), minlength=1026)
    return q.astype(np.float32), mass[1:]


def report(name: str, cand: np.ndarray, post: np.ndarray, mass: np.ndarray, target) -> None:
    hard = np.bincount(cand[np.arange(len(cand)), post.argmax(1)], minlength=1026)[1:]
    tgt = np.asarray(target, dtype=np.float64)
    print(
        f"{name}: soft mean/std/min/max={mass.mean():.2f}/{mass.std():.2f}/{mass.min():.1f}/{mass.max():.1f} "
        f"max|soft-tgt|={np.max(np.abs(mass - tgt)):.3f} "
        f"hard mean/std/min/max={hard.mean():.1f}/{hard.std():.1f}/{int(hard.min())}/{int(hard.max())}",
        flush=True,
    )


def pack(src: Path, dst: Path, target) -> None:
    with np.load(src) as z:
        data = {k: z[k] for k in z.files}
    cand = data["candidates"].astype(np.int64)
    post = data["posteriors"].astype(np.float32)
    tgt = np.full(1025, float(target), dtype=np.float64) if np.ndim(target) == 0 else np.asarray(target)
    before = np.bincount(cand[np.arange(len(cand)), post.argmax(1)], minlength=1026)[1:]
    print(
        f"{src.name} before hard mean/std/min/max="
        f"{before.mean():.1f}/{before.std():.1f}/{int(before.min())}/{int(before.max())}",
        flush=True,
    )
    new_post, mass = sinkhorn_quota(cand, post, tgt)
    data["posteriors"] = new_post
    data["quota_target"] = tgt.astype(np.float32)
    data["quota_soft_mass"] = mass.astype(np.float32)
    np.savez_compressed(dst, **data)
    report(dst.name, cand, new_post, mass, tgt)


def main() -> None:
    pack(L / "phase2_imp_ranker_fold0_scores.npz",
         L / "quota_uniform_fold0_scores.npz", 53.0)
    pack(L / "phase2_imp_ranker_test.npz",
         L / "quota_uniform_test.npz", 160000.0 / 1025.0)


if __name__ == "__main__":
    main()
