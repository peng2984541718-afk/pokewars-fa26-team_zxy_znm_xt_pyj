"""Fit the published top-20 candidate ranker on folds 1+2, score fold 0."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np

from predict_meta_top20_test import FEATURE_NAMES, features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/large/meta_top20_fold0_scores.npz"))
    parser.add_argument("--run-name", default="feature_rerank_rot5_top20")
    parser.add_argument("--expected-top1", type=float, default=51520/54325,
                        help="Recorded fold-0 CV accuracy to check; negative skips the check")
    args = parser.parse_args()
    root = args.input_dir
    conv = np.load(root / "multibackbone_convnextv2_oof.npy", mmap_mode="r")
    dense = np.load(root / "multibackbone_densenet169_oof.npy", mmap_mode="r")
    train_x, train_y = [], []
    for fold in (1, 2):
        with np.load(root / f"{args.run_name}_f{fold}.npz") as z:
            indices = z["valid_indices"]
            train_x.append(features(z, conv[indices], dense[indices]))
            train_y.append((z["candidates"] == z["labels"][:, None]).reshape(-1).astype(np.uint8))
    x = np.concatenate(train_x)
    y = np.concatenate(train_y)
    del train_x, train_y
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=.05,
                               num_leaves=15, min_child_samples=1000,
                               colsample_bytree=.8, subsample=.8,
                               reg_lambda=2.0, n_jobs=16, verbosity=-1)
    model.fit(x, y, feature_name=FEATURE_NAMES,
              categorical_feature=[len(FEATURE_NAMES)-1])
    del x, y
    print("Trained on image folds 1+2", flush=True)
    with np.load(root / f"{args.run_name}_f0.npz") as z:
        indices = z["valid_indices"]
        candidates = z["candidates"].astype(np.uint16)
        wrn_probs = z["cnn_probs"].astype(np.float16)
        labels = z["labels"].astype(np.uint16)
        x = features(z, conv[indices], dense[indices])
    scores = model.predict_proba(x)[:, 1].reshape(-1, 20).astype(np.float32)
    meta = np.take_along_axis(candidates, scores.argmax(1)[:, None], axis=1)[:, 0]
    top1 = float(np.mean(meta == labels))
    if args.expected_top1 >= 0 and abs(top1 - args.expected_top1) > .0002:
        raise ValueError(f"Fold-0 top1 {top1:.6f} differs from recorded CV")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, valid_indices=indices,
                        candidates=candidates, wrn_probs=wrn_probs,
                        ranker_scores=scores.astype(np.float16),
                        labels=labels, meta_top1=meta)
    print(f"Saved {args.output}: {len(indices)} rows, top1={top1:.6f}", flush=True)


if __name__ == "__main__":
    main()
