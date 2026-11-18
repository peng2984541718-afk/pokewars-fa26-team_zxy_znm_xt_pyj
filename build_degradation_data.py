"""Fold-safe clean silhouette sources and a real degradation-parameter pool.

Sources: the least noisy labeled images outside the validation fold, plus the
least noisy images of that fold's pseudo-label bank. Parameter pool: estimated
parameters of noisy labeled images outside the validation fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from degradation import PARAM_NAMES, estimate_params, laplacian_noise, silhouettes
from image_experiment import SEED, stratified_folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--bank", type=Path, default=None,
                        help="Fold-safe pseudo-label bank (default pseudo_bank_f{fold}.npz)")
    parser.add_argument("--clean-quantile", type=float, default=.2)
    parser.add_argument("--hard-quantile", type=float, default=.4)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    bank_path = args.bank or Path(f"outputs/large/pseudo_bank_f{args.fold}.npz")
    output = args.output or Path(f"outputs/large/degradation_f{args.fold}.npz")

    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    images = np.load(args.data_dir / "labeled_32x32.npy")
    folds = stratified_folds(labels, 3, SEED)
    noise = laplacian_noise(images)
    clean_cut, hard_cut = np.quantile(noise, [args.clean_quantile, args.hard_quantile])
    train = folds != args.fold

    clean = np.flatnonzero(train & (noise <= clean_cut))
    with np.load(bank_path) as bank:
        bank_images, bank_labels = bank["images"], bank["labels"].astype(np.int64)
    bank_clean = laplacian_noise(bank_images) <= clean_cut
    source_images = np.concatenate([images[clean], bank_images[bank_clean]])
    source_labels = np.concatenate([labels[clean], bank_labels[bank_clean]])
    source_params = estimate_params(source_images)
    s, residual = silhouettes(source_images, source_params)

    hard = np.flatnonzero(train & (noise >= hard_cut))
    pool = estimate_params(images[hard])

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, silhouettes=s, residual=residual,
                        labels=source_labels.astype(np.int16),
                        from_bank=np.r_[np.zeros(len(clean), bool), np.ones(bank_clean.sum(), bool)],
                        source_params=source_params, params=pool, params_index=hard)
    report = {"fold": args.fold, "clean_noise_cut": float(clean_cut),
              "hard_noise_cut": float(hard_cut), "labeled_sources": int(len(clean)),
              "bank_sources": int(bank_clean.sum()),
              "classes_covered": int(len(np.unique(source_labels))),
              "min_sources_per_class": int(np.bincount(source_labels, minlength=1025).min()),
              "param_pool": int(len(pool)),
              "source_params_mean": dict(zip(PARAM_NAMES, source_params.mean(0).round(4).tolist())),
              "pool_params_mean": dict(zip(PARAM_NAMES, pool.mean(0).round(4).tolist()))}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
