"""Reallocate only uncertain test images toward remaining-partner expected counts.

Locked images keep the 79.9 ranker posterior. Uncertain images (high oop or
entropy) are Sinkhorn-matched to leftover slots: 0.307 * remaining partners
minus already-locked counts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

L = Path("outputs/large")


def main() -> None:
    kb = np.load("data/known_battles.npz")
    ub = np.load(L / "unlabeled_constrained_battles_v2.npz")
    deg = np.bincount(np.r_[kb["a"], kb["b"], ub["a"], ub["b"]], minlength=1026)[1:]
    remain = (1024 - deg).astype(np.float64)
    n_rem_pairs = remain.sum() / 2.0
    p_pair = 80000.0 / n_rem_pairs
    target = p_pair * remain

    with np.load(L / "phase2_imp_ranker_test.npz") as z:
        data = {k: z[k] for k in z.files}
    cand = data["candidates"].astype(np.int64)
    post = np.clip(data["posteriors"].astype(np.float64), 1e-12, 1.0)
    post /= post.sum(1, keepdims=True)
    oop = data["oop_mass"].astype(np.float64)
    ent = -(post * np.log(post)).sum(1)

    uncertain = (oop >= 0.12) | (ent >= 0.5)
    if uncertain.mean() < 0.05:
        uncertain = oop >= np.quantile(oop, 0.92)
    locked = ~uncertain
    hard = cand[np.arange(len(cand)), post.argmax(1)]
    lock_cnt = np.bincount(hard[locked], minlength=1026)[1:].astype(np.float64)
    slack = np.clip(target - lock_cnt, 0.0, None)
    n_u = int(uncertain.sum())
    if slack.sum() <= 0:
        slack[:] = n_u / 1025.0
    else:
        slack *= n_u / slack.sum()

    idx = np.flatnonzero(uncertain)
    q = post[idx].copy()
    c = cand[idx]
    tgt = np.zeros(1026, dtype=np.float64)
    tgt[1:] = slack
    for _ in range(25):
        mass = np.bincount(c.ravel(), weights=q.ravel(), minlength=1026).astype(np.float64)
        mass[0] = 1.0
        scale = np.clip(tgt / np.maximum(mass, 1e-8), 0.05, 20.0)
        scale[0] = 1.0
        q *= scale[c]
        q /= np.maximum(q.sum(1, keepdims=True), 1e-12)
    post[idx] = q

    new_hard = cand[np.arange(len(cand)), post.argmax(1)]
    cnt0 = np.bincount(hard, minlength=1026)[1:]
    cnt1 = np.bincount(new_hard, minlength=1026)[1:]
    moved = int((new_hard != hard).sum())
    print(
        f"uncertain={n_u} ({100 * n_u / len(cand):.2f}%) moved={moved} "
        f"hard std {cnt0.std():.2f}->{cnt1.std():.2f} "
        f"min/max {int(cnt0.min())}/{int(cnt0.max())}->{int(cnt1.min())}/{int(cnt1.max())} "
        f"corr_remain {np.corrcoef(cnt1, remain)[0,1]:.3f} "
        f"target mean/std {target.mean():.1f}/{target.std():.1f}",
        flush=True,
    )
    data["posteriors"] = post.astype(np.float32)
    data["uncertain"] = uncertain.astype(np.uint8)
    np.savez_compressed(L / "gated_remain_test.npz", **data)
    print("saved outputs/large/gated_remain_test.npz", flush=True)


if __name__ == "__main__":
    main()
