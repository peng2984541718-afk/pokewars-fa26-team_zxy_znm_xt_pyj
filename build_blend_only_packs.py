"""CPU-only: turn Phase-2 6-view probability matrices into ranker-shaped packs.
Uses the same 0.35/0.22/0.33/0.10 blend as extract_gallery_features_v2.
Does not touch the GPU (the gallery job keeps running).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from image_experiment import SEED, stratified_folds

L = Path("outputs/large")
W = {"wrn28": 0.35, "wrn40": 0.22, "dense": 0.33, "conv": 0.10}
K = 40
CHUNK = 8192


def blend_topk(paths: dict[str, Path], n_expected: int | None = None):
    mats = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    n, c = mats["wrn28"].shape
    if n_expected is not None and n != n_expected:
        raise SystemExit(f"{paths['wrn28']} has {n} rows, expected {n_expected}")
    cand = np.empty((n, K), dtype=np.uint16)
    post = np.empty((n, K), dtype=np.float32)
    oop = np.empty(n, dtype=np.float32)
    for start in range(0, n, CHUNK):
        end = min(n, start + CHUNK)
        blend = None
        for name, weight in W.items():
            sub = np.asarray(mats[name][start:end], dtype=np.float32)
            blend = weight * sub if blend is None else blend + weight * sub
        part = np.argpartition(blend, -K, axis=1)[:, -K:]
        part_vals = np.take_along_axis(blend, part, axis=1)
        order = np.argsort(-part_vals, axis=1)
        cls0 = np.take_along_axis(part, order, axis=1)
        vals = np.take_along_axis(blend, cls0, axis=1)
        cand[start:end] = (cls0 + 1).astype(np.uint16)
        post[start:end] = (vals / np.maximum(vals.sum(1, keepdims=True), 1e-12)).astype(np.float32)
        oop[start:end] = np.clip(1.0 - vals.sum(1), 1e-6, 1.0).astype(np.float32)
        print(f"  {start}:{end}/{n}", flush=True)
    return cand, post, oop


def save_pack(path: Path, cand, post, oop, **extra) -> None:
    np.savez_compressed(path, candidates=cand, posteriors=post, oop_mass=oop, **extra)
    print(f"saved {path} n={len(cand)} top1_mass={post[:, 0].mean():.4f} oop={oop.mean():.4f}", flush=True)


def main() -> None:
    labels = np.load("labeled_ids.npy").astype(np.int64)
    folds = stratified_folds(labels, 3, SEED)
    va0 = np.flatnonzero(folds == 0)

    gal = L / "gallery_p2imp_top40_f0.npz"
    if gal.exists():
        with np.load(gal) as z:
            bp = z["blend_probs"].astype(np.float32)
            post = bp / np.maximum(bp.sum(1, keepdims=True), 1e-12)
            save_pack(
                L / "blend_only_fold0_scores.npz",
                z["candidates"].astype(np.uint16),
                post.astype(np.float32),
                z["oop_mass"].astype(np.float32),
                valid_indices=z["valid_indices"],
                labels=z["labels"],
            )
            print(
                f"fold0 gallery blend top1={(z['candidates'][:, 0] == z['labels']).mean():.5f} "
                f"top40={(z['candidates'] == z['labels'][:, None]).any(1).mean():.5f}",
                flush=True,
            )
    else:
        cand, post, oop = blend_topk({
            "wrn28": L / "oof_wrn28_p2imp.npy",
            "wrn40": L / "oof_wrn40_p2imp.npy",
            "dense": L / "oof_dense_p2imp.npy",
            "conv": L / "oof_conv_p2imp.npy",
        }, n_expected=len(labels))
        save_pack(
            L / "blend_only_fold0_scores.npz",
            cand[va0], post[va0], oop[va0],
            valid_indices=va0,
            labels=(labels[va0] + 1).astype(np.uint16),
        )

    jobs = [
        ("blend_only_unlabeled_f0only.npz", {
            "wrn28": L / "unl_f0_wrn28_p2imp.npy",
            "wrn40": L / "unl_f0_wrn40_p2imp.npy",
            "dense": L / "unl_f0_dense_p2imp.npy",
            "conv": L / "unl_f0_conv_p2imp.npy",
        }),
        ("blend_only_unlabeled.npz", {
            "wrn28": L / "unl_wrn28_p2imp.npy",
            "wrn40": L / "unl_wrn40_p2imp.npy",
            "dense": L / "unl_dense_p2imp.npy",
            "conv": L / "unl_conv_p2imp.npy",
        }),
        ("blend_only_test.npz", {
            "wrn28": L / "test_wrn28_p2imp.npy",
            "wrn40": L / "test_wrn40_p2imp.npy",
            "dense": L / "test_dense_p2imp.npy",
            "conv": L / "test_conv_p2imp.npy",
        }),
    ]
    for name, paths in jobs:
        print("packing", name, flush=True)
        cand, post, oop = blend_topk(paths)
        save_pack(L / name, cand, post, oop)


if __name__ == "__main__":
    main()
