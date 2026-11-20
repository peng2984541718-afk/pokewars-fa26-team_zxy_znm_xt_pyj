"""Cross-fit a small candidate ranker using WRN, ConvNeXt and kNN OOF scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from image_experiment import SEED, stratified_folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/large/meta_rerank_cv.json"))
    parser.add_argument("--class-feature", action="store_true")
    parser.add_argument("--run-name", default="feature_rerank_rot5")
    parser.add_argument("--save-predictions", type=Path, default=None)
    parser.add_argument("--dense-full", type=Path, default=None)
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--estimators", type=int, default=200)
    parser.add_argument("--min-child-samples", type=int, default=1000)
    args = parser.parse_args()
    labels = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64) + 1
    folds = stratified_folds(labels - 1, 3, SEED)
    conv_full = np.load(args.input_dir / "multibackbone_convnextv2_oof.npy",
                        mmap_mode="r")
    dense_full = (np.load(args.dense_full, mmap_mode="r")
                  if args.dense_full is not None else None)
    features = []
    targets = []
    row_labels = []
    candidates = []
    wrn_list = []
    fold_ids = []
    baseline = []
    indices_list = []
    feature_names = [
        "log_wrn_p", "wrn_rank", "log_wrn_p_over_top1", "wrn_top1_margin",
        "wrn_top10_mass", "log_conv_p", "log_conv_p_over_top1",
        "conv_rank_top10", "conv_top1_p", "knn1", "knn5", "prototype",
        "knn1_vs_best", "simple_fused_score",
    ]
    if dense_full is not None:
        feature_names.extend(("log_dense_p", "log_dense_relative",
                              "dense_top1_p", "dense_rank_top10"))
    if args.class_feature:
        feature_names.append("candidate_id")
    for fold in range(3):
        z = np.load(args.input_dir / f"{args.run_name}_f{fold}.npz")
        indices = z["valid_indices"]
        if np.any(folds[indices] != fold) or not np.array_equal(z["labels"], labels[indices]):
            raise ValueError("Feature cache and fold assignment differ")
        ids = z["candidates"].astype(np.int64)
        candidate_count = ids.shape[1]
        wrn = z["cnn_probs"].astype(np.float32)
        conv = np.take_along_axis(np.asarray(conv_full[indices], dtype=np.float32),
                                  ids - 1, axis=1)
        conv_top = np.max(conv_full[indices], axis=1).astype(np.float32)
        conv_top10 = np.argpartition(conv_full[indices], -10, axis=1)[:, -10:]
        conv_rank = np.where((ids[:, :, None] - 1 == conv_top10[:, None, :]).any(2),
                             1, 11).astype(np.float32)
        log_wrn = np.log(np.maximum(wrn, 1e-12))
        log_conv = np.log(np.maximum(conv, 1e-12))
        knn1 = z["knn1"].astype(np.float32)
        columns = [
            log_wrn, np.broadcast_to(np.arange(1, candidate_count+1, dtype=np.float32), ids.shape),
            log_wrn - log_wrn[:, :1],
            np.broadcast_to((wrn[:, :1] - wrn[:, 1:2]), ids.shape),
            np.broadcast_to(wrn.sum(1, keepdims=True), ids.shape),
            log_conv, log_conv - np.log(np.maximum(conv_top[:, None], 1e-12)),
            conv_rank, np.broadcast_to(conv_top[:, None], ids.shape),
            knn1, z["knn5"].astype(np.float32), z["prototype"].astype(np.float32),
            knn1 - knn1.max(1, keepdims=True), log_wrn + 32 * knn1,
        ]
        if dense_full is not None:
            dense = np.take_along_axis(np.asarray(dense_full[indices], dtype=np.float32),
                                       ids - 1, axis=1)
            dense_top = np.max(dense_full[indices], axis=1).astype(np.float32)
            dense_top10 = np.argpartition(dense_full[indices], -10, axis=1)[:, -10:]
            log_dense = np.log(np.maximum(dense, 1e-12))
            dense_rank = np.where((ids[:, :, None] - 1 == dense_top10[:, None, :]).any(2),
                                  1, 11).astype(np.float32)
            columns.extend((log_dense,
                            log_dense - np.log(np.maximum(dense_top[:, None], 1e-12)),
                            np.broadcast_to(dense_top[:, None], ids.shape), dense_rank))
        if args.class_feature:
            columns.append(ids.astype(np.float32))
        features.append(np.stack(columns, axis=2).reshape(-1, len(columns)))
        targets.append((ids == labels[indices, None]).reshape(-1).astype(np.uint8))
        row_labels.append(labels[indices])
        candidates.append(ids)
        wrn_list.append(wrn)
        fold_ids.append(np.full(len(indices), fold, dtype=np.int8))
        baseline.append(ids[np.arange(len(ids)), (log_wrn + 32*knn1).argmax(1)])
        indices_list.append(indices)
    x = np.concatenate(features)
    y = np.concatenate(targets)
    true = np.concatenate(row_labels)
    ids = np.concatenate(candidates)
    all_wrn = np.concatenate(wrn_list)
    fold_ids = np.concatenate(fold_ids)
    base = np.concatenate(baseline)
    all_indices = np.concatenate(indices_list)
    output = {"feature_names": feature_names, "model": "LightGBM binary candidate ranker",
              "folds": []}
    all_pred = np.empty_like(base)
    all_scores = np.empty_like(all_wrn, dtype=np.float32)
    all_top10_hit = np.zeros(len(base), dtype=bool)
    for fold in range(3):
        train_rows = fold_ids != fold
        valid_rows = ~train_rows
        train_candidate = np.repeat(train_rows, candidate_count)
        model = lgb.LGBMClassifier(n_estimators=args.estimators, learning_rate=0.05,
                                   num_leaves=args.num_leaves,
                                   min_child_samples=args.min_child_samples,
                                   colsample_bytree=0.8, subsample=0.8,
                                   reg_lambda=2.0, n_jobs=16, verbosity=-1)
        model.fit(x[train_candidate], y[train_candidate], feature_name=feature_names,
                  categorical_feature=[len(feature_names)-1] if args.class_feature else "auto")
        scores = model.predict_proba(x[np.repeat(valid_rows, candidate_count)])[:, 1].reshape(-1, candidate_count)
        all_scores[valid_rows] = scores
        predicted = ids[valid_rows][np.arange(valid_rows.sum()), scores.argmax(1)]
        all_pred[valid_rows] = predicted
        selected_top10 = np.take_along_axis(ids[valid_rows],
                                             np.argsort(-scores, axis=1)[:, :10], axis=1)
        all_top10_hit[valid_rows] = (selected_top10 == true[valid_rows, None]).any(1)
        outcome = {"fold": fold, "count": int(valid_rows.sum()),
                   "cnn_top1_correct": int((ids[valid_rows, 0] == true[valid_rows]).sum()),
                   "knn1_top1_correct": int((base[valid_rows] == true[valid_rows]).sum()),
                   "meta_top1_correct": int((predicted == true[valid_rows]).sum()),
                   "gained_vs_knn1": int(((base[valid_rows] != true[valid_rows]) &
                                          (predicted == true[valid_rows])).sum()),
                   "lost_vs_knn1": int(((base[valid_rows] == true[valid_rows]) &
                                        (predicted != true[valid_rows])).sum()),
                   "meta_top10_correct": int(all_top10_hit[valid_rows].sum())}
        output["folds"].append(outcome)
        print(outcome, flush=True)
    output["overall"] = {"count": len(true),
                         "cnn_top1_accuracy": float(np.mean(ids[:, 0] == true)),
                         "knn1_top1_accuracy": float(np.mean(base == true)),
                         "meta_top1_accuracy": float(np.mean(all_pred == true)),
                         "meta_top10_accuracy": float(all_top10_hit.mean()),
                         "top10_accuracy": float(np.mean((ids == true[:, None]).any(1)))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    if args.save_predictions is not None:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_predictions, indices=all_indices,
                            labels=true, candidates=ids.astype(np.uint16),
                            wrn_probs=all_wrn.astype(np.float16),
                            ranker_scores=all_scores.astype(np.float16),
                            cnn_top1=ids[:, 0], knn_top1=base, meta_top1=all_pred,
                            meta_top10_hit=all_top10_hit, folds=fold_ids)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
