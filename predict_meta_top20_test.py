"""Fit the 94.91%-CV top-20 LightGBM ranker and export test image IDs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from image_experiment import NUM_CLASSES


FEATURE_NAMES = [
    "log_wrn_p", "wrn_rank", "log_wrn_p_over_top1", "wrn_top1_margin",
    "wrn_top10_mass", "log_conv_p", "log_conv_p_over_top1",
    "conv_rank_top10", "conv_top1_p", "knn1", "knn5", "prototype",
    "knn1_vs_best", "simple_fused_score", "log_dense_p",
    "log_dense_relative", "dense_top1_p", "dense_rank_top10", "candidate_id",
]


def features(z, conv_full: np.ndarray, dense_full: np.ndarray) -> np.ndarray:
    ids = z["candidates"].astype(np.int64)
    wrn = z["cnn_probs"].astype(np.float32)
    if ids.shape[1] != 20 or np.any(ids < 1) or np.any(ids > NUM_CLASSES):
        raise ValueError("Expected 20 valid 1-based candidate IDs")
    candidate = ids - 1
    knn1 = z["knn1"].astype(np.float32)
    conv = np.take_along_axis(conv_full, candidate, axis=1).astype(np.float32)
    conv_top = np.max(conv_full, axis=1).astype(np.float32)
    conv_top10 = np.argpartition(conv_full, -10, axis=1)[:, -10:]
    conv_rank = np.where((candidate[:, :, None] == conv_top10[:, None, :]).any(2),
                         1, 11).astype(np.float32)
    dense = np.take_along_axis(dense_full, candidate, axis=1).astype(np.float32)
    dense_top = np.max(dense_full, axis=1).astype(np.float32)
    dense_top10 = np.argpartition(dense_full, -10, axis=1)[:, -10:]
    dense_rank = np.where((candidate[:, :, None] == dense_top10[:, None, :]).any(2),
                          1, 11).astype(np.float32)
    log_wrn = np.log(np.maximum(wrn, 1e-12))
    log_conv = np.log(np.maximum(conv, 1e-12))
    log_dense = np.log(np.maximum(dense, 1e-12))
    cols = [
        log_wrn,
        np.broadcast_to(np.arange(1, 21, dtype=np.float32), ids.shape),
        log_wrn-log_wrn[:, :1],
        np.broadcast_to(wrn[:, :1]-wrn[:, 1:2], ids.shape),
        np.broadcast_to(wrn.sum(1, keepdims=True), ids.shape),
        log_conv,
        log_conv-np.log(np.maximum(conv_top[:, None], 1e-12)),
        conv_rank,
        np.broadcast_to(conv_top[:, None], ids.shape),
        knn1,
        z["knn5"].astype(np.float32),
        z["prototype"].astype(np.float32),
        knn1-knn1.max(1, keepdims=True),
        log_wrn+32*knn1,
        log_dense,
        log_dense-np.log(np.maximum(dense_top[:, None], 1e-12)),
        np.broadcast_to(dense_top[:, None], ids.shape),
        dense_rank,
        ids.astype(np.float32),
    ]
    x = np.stack(cols, axis=2).reshape(-1, len(cols))
    if x.shape[1] != len(FEATURE_NAMES) or not np.all(np.isfinite(x)):
        raise ValueError("Invalid ranker feature matrix")
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/large/test_meta_three_arch_top20_160000_top10.csv"))
    parser.add_argument("--run-name", default="feature_rerank_rot5_top20",
                        help="Prefix of the fold-safe OOF feature files")
    parser.add_argument("--test-features", default="test_wrn_rot5_top20_features.npz")
    parser.add_argument("--cv-report", default="meta_rerank_three_arch_top20_cv.json",
                        help="evaluate_meta_rerank.py output for these features")
    args = parser.parse_args()
    folder = args.input_dir
    conv_oof = np.load(folder / "multibackbone_convnextv2_oof.npy", mmap_mode="r")
    dense_oof = np.load(folder / "multibackbone_densenet169_oof.npy", mmap_mode="r")
    train_x, train_y = [], []
    for fold in range(3):
        with np.load(folder / f"{args.run_name}_f{fold}.npz") as z:
            indices = z["valid_indices"]
            x = features(z, conv_oof[indices], dense_oof[indices])
            ids = z["candidates"]
            y = (ids == z["labels"][:, None]).reshape(-1).astype(np.uint8)
        train_x.append(x)
        train_y.append(y)
        print(f"Prepared training fold {fold}", flush=True)
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
    print("Trained final candidate ranker", flush=True)
    paths = (args.data_dir / "test_paths.txt").read_text().splitlines()
    with np.load(folder / args.test_features) as z:
        ids = z["candidates"].astype(np.int64)
        wrn = z["cnn_probs"].astype(np.float32)
        if ids.shape != (len(paths), 20) or len(paths) != 160000:
            raise ValueError("Test feature count mismatch")
        conv = np.load(folder / "test_convnextv2_probs.npy", mmap_mode="r")
        dense = np.load(folder / "test_densenet169_probs.npy", mmap_mode="r")
        scores = model.predict_proba(features(z, conv, dense))[:, 1].reshape(-1, 20)
    order = np.argsort(-scores, axis=1)[:, :10]
    top_ids = np.take_along_axis(ids, order, axis=1)
    top_scores = np.take_along_axis(scores, order, axis=1)
    top_wrn = np.take_along_axis(wrn, order, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "top1_pokemon_id", "top10_pokemon_ids",
                         "top10_ranker_scores", "top10_wrn_probabilities"])
        for path, row_ids, row_scores, row_wrn in zip(paths, top_ids, top_scores, top_wrn):
            writer.writerow([path, int(row_ids[0]), json.dumps(row_ids.tolist()),
                             json.dumps(row_scores.astype(float).tolist()),
                             json.dumps(row_wrn.astype(float).tolist())])
    cv = json.loads((folder / args.cv_report).read_text())["overall"]
    metadata = {
        "images": len(paths), "top_k": 10,
        "cv_source": args.cv_report,
        "cv_top1": cv["meta_top1_accuracy"],
        "cv_ranked_top10": cv["meta_top10_accuracy"],
        "cv_candidate_top20_oracle": cv["top10_accuracy"],
        "feature_run_name": args.run_name,
        "test_features": args.test_features,
        "ranker": "LightGBM trained on all 3 fold-safe OOF feature sets",
        "candidate_pool": "WRN rotation-TTA top20",
        "other_features": "ConvNeXt V2, DenseNet-169, WRN kNN1/kNN5/prototypes",
        "ranker_scores_calibrated_probabilities": False,
        "wrn_probabilities_calibrated": False,
        "kaggle_submission": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"Saved {len(paths)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
