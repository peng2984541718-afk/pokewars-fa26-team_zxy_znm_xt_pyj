"""Replace gallery class probabilities with single-facing probabilities.

Candidate identities and kNN columns stay put, so the saved ranker features
remain aligned. Only the backbone probabilities, blend, entropy, and out-of-pool
mass change.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

L = Path("outputs/large")
W = {"wrn28": 0.35, "wrn40": 0.22, "dense": 0.33, "conv": 0.10}


def load_rows(paths: dict[str, Path], rows: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for name, path in paths.items():
        mat = np.load(path, mmap_mode="r")
        out[name] = np.asarray(mat[rows], dtype=np.float32)
    return out


def splice(src: Path, dst: Path, paths: dict[str, Path], rows: np.ndarray) -> None:
    with np.load(src) as z:
        data = {k: z[k] for k in z.files}
    cand = data["candidates"].astype(np.int64) - 1
    probs = load_rows(paths, rows)
    blend = np.zeros((len(rows), probs["wrn28"].shape[1]), dtype=np.float32)
    for name, sub in probs.items():
        blend += W[name] * sub
        data[f"{name}_probs"] = np.take_along_axis(sub, cand, axis=1).astype(np.float16)
        p = np.clip(sub, 1e-8, 1.0)
        ent = -(p * np.log(p)).sum(1)
        top2 = np.partition(sub, -2, axis=1)[:, -2:]
        top1 = top2.max(1)
        img = np.stack([top1, top1 - top2.min(1), ent], axis=1).astype(np.float16)
        data[f"{name}_img"] = img
    gathered = np.take_along_axis(blend, cand, axis=1)
    data["blend_probs"] = gathered.astype(np.float16)
    data["oop_mass"] = np.clip(1.0 - gathered.sum(1), 1e-6, 1.0).astype(np.float32)
    np.savez_compressed(dst, **data)
    print(f"saved {dst.name} rows={len(rows)} top1_mass={gathered[:, 0].mean():.4f}", flush=True)


def main() -> None:
    oof = {
        "wrn28": L / "oof_wrn28_face.npy",
        "wrn40": L / "oof_wrn40_face.npy",
        "dense": L / "oof_dense_face.npy",
        "conv": L / "oof_conv_face.npy",
    }
    for fold in range(3):
        with np.load(L / f"gallery_p2imp_top40_f{fold}.npz") as z:
            rows = z["valid_indices"].astype(np.int64)
        splice(L / f"gallery_p2imp_top40_f{fold}.npz",
               L / f"gallery_face_top40_f{fold}.npz", oof, rows)
    specs = [
        ("gallery_p2imp_top40_unlabeled.npz", "gallery_face_top40_unlabeled.npz", {
            "wrn28": L / "unl_wrn28_face.npy", "wrn40": L / "unl_wrn40_face.npy",
            "dense": L / "unl_dense_face.npy", "conv": L / "unl_conv_face.npy",
        }),
        ("gallery_p2imp_top40_unl_f0only.npz", "gallery_face_top40_unl_f0only.npz", {
            "wrn28": L / "unl_f0_wrn28_face.npy", "wrn40": L / "unl_f0_wrn40_face.npy",
            "dense": L / "unl_f0_dense_face.npy", "conv": L / "unl_f0_conv_face.npy",
        }),
        ("gallery_p2imp_top40_test.npz", "gallery_face_top40_test.npz", {
            "wrn28": L / "test_wrn28_face.npy", "wrn40": L / "test_wrn40_face.npy",
            "dense": L / "test_dense_face.npy", "conv": L / "test_conv_face.npy",
        }),
    ]
    for src, dst, paths in specs:
        n = np.load(next(iter(paths.values())), mmap_mode="r").shape[0]
        splice(L / src, L / dst, paths, np.arange(n))


if __name__ == "__main__":
    main()
