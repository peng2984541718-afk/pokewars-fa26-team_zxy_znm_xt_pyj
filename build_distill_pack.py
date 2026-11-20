"""Align the 79.9 teacher with image rows for soft distillation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

L = Path("outputs/large")


def main() -> None:
    n_labeled = len(np.load("labeled_ids.npy"))
    with np.load(L / "phase2_imp_ranker_oof.npz") as z:
        idx = z["indices"].astype(np.int64)
        if len(idx) != n_labeled or len(np.unique(idx)) != n_labeled:
            raise SystemExit(f"OOF indices are not a permutation of the labeled images: {len(idx)}")
        cand = np.empty((n_labeled, z["candidates"].shape[1]), dtype=np.uint16)
        post = np.empty((n_labeled, z["posteriors"].shape[1]), dtype=np.float16)
        cand[idx] = z["candidates"]
        post[idx] = z["posteriors"]
    np.savez_compressed(L / "distill_labeled_oof.npz", candidates=cand, posteriors=post)
    print(f"labeled teacher {cand.shape} top1_mass={post.astype(np.float32)[:, 0].mean():.4f}", flush=True)

    unl = np.asarray(np.load("unl/test_32x32.npy"))
    test = np.asarray(np.load("test_32x32.npy"))
    with np.load(L / "phase2_imp_ranker_unlabeled.npz") as u, np.load(L / "phase2_imp_ranker_test.npz") as t:
        if len(unl) != len(u["candidates"]) or len(test) != len(t["candidates"]):
            raise SystemExit("External teacher rows do not match images")
        cand_e = np.concatenate([u["candidates"], t["candidates"]]).astype(np.uint16)
        post_e = np.concatenate([u["posteriors"], t["posteriors"]]).astype(np.float16)
    np.save(L / "distill_external_images.npy", np.concatenate([unl, test], axis=0))
    np.savez_compressed(L / "distill_external_full.npz", candidates=cand_e, posteriors=post_e)
    print(f"external teacher images={len(cand_e)} unl={len(unl)} test={len(test)}", flush=True)


if __name__ == "__main__":
    main()
