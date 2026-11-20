"""Cross-fit a top-10 selector over three-backbone candidate unions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from image_experiment import SEED, stratified_folds


def top_ids(p: np.ndarray, k: int) -> np.ndarray:
    ids = np.argpartition(p, -k, axis=1)[:, -k:]
    vals = np.take_along_axis(p, ids, axis=1)
    return np.take_along_axis(ids, np.argsort(-vals, axis=1), axis=1)


def candidate_rank(ids: np.ndarray, top: np.ndarray) -> np.ndarray:
    match = ids[:, :, None] == top[:, None, :]
    return np.where(match.any(2), match.argmax(2)+1, top.shape[1]+1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/large/three_arch_candidate_ranker_cv.json"))
    parser.add_argument("--objective", choices=("binary", "lambdarank"), default="binary")
    parser.add_argument("--knn-run-name", default=None,
                        help="Optional fold-safe feature reranker NPZ prefix")
    parser.add_argument("--save-scores", type=Path, default=None)
    args = parser.parse_args()
    y = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    folds = stratified_folds(y, 3, SEED)
    names = ("wrn_rot5", "convnextv2", "densenet169")
    probs = {name: np.load(args.input_dir / f"multibackbone_{name}_oof.npy",
                           mmap_mode="r") for name in names}
    top = {"wrn_rot5": top_ids(probs["wrn_rot5"], 20),
           "convnextv2": top_ids(probs["convnextv2"], 10),
           "densenet169": top_ids(probs["densenet169"], 10)}
    ids = np.concatenate([top[name] for name in names], axis=1)
    valid = np.ones(ids.shape, dtype=bool)
    valid[:, 20:30] = ~(top["convnextv2"][:, :, None] ==
                        top["wrn_rot5"][:, None, :]).any(2)
    valid[:, 30:] = ~((top["densenet169"][:, :, None] ==
                       top["wrn_rot5"][:, None, :]).any(2) |
                      (top["densenet169"][:, :, None] ==
                       top["convnextv2"][:, None, :]).any(2))
    columns = []
    feature_names = []
    for name in names:
        p = probs[name]
        candidate_p = np.take_along_axis(p, ids, axis=1).astype(np.float32)
        top_p = np.max(p, axis=1).astype(np.float32)
        second_p = np.take_along_axis(p, top[name][:, 1:2], axis=1)[:, 0].astype(np.float32)
        log_p = np.log(np.maximum(candidate_p, 1e-12))
        columns.extend((log_p, candidate_rank(ids, top[name]),
                        log_p-np.log(np.maximum(top_p[:, None], 1e-12)),
                        np.broadcast_to(top_p[:, None], ids.shape),
                        np.broadcast_to((top_p-second_p)[:, None], ids.shape)))
        feature_names.extend((f"{name}_log_p", f"{name}_rank",
                              f"{name}_log_relative", f"{name}_top1_p",
                              f"{name}_margin"))
    combined = (.5*np.take_along_axis(probs["wrn_rot5"], ids, axis=1).astype(np.float32)
                +.2*np.take_along_axis(probs["convnextv2"], ids, axis=1).astype(np.float32)
                +.3*np.take_along_axis(probs["densenet169"], ids, axis=1).astype(np.float32))
    columns.append(combined)
    feature_names.append("fixed_blend_p")
    if args.knn_run_name is not None:
        knn_columns = {name: np.full(ids.shape, -1, dtype=np.float32)
                       for name in ("knn1", "knn5", "prototype")}
        for fold in range(3):
            z = np.load(args.input_dir / f"{args.knn_run_name}_f{fold}.npz")
            indices = z["valid_indices"]
            if np.any(folds[indices] != fold):
                raise ValueError("kNN features and fold assignment differ")
            match = ids[indices, :, None] == (z["candidates"][:, None, :] - 1)
            has = match.any(2)
            positions = match.argmax(2)
            for name in knn_columns:
                selected = np.take_along_axis(z[name], positions, axis=1)
                knn_columns[name][indices] = np.where(has, selected, -1)
        columns.extend(knn_columns[name] for name in knn_columns)
        feature_names.extend(knn_columns)
    columns.append(ids.astype(np.float32))
    feature_names.append("candidate_id")
    x = np.stack(columns, axis=2).reshape(-1, len(feature_names))
    labels = (ids == y[:, None]).reshape(-1).astype(np.uint8)
    keep = valid.reshape(-1)
    result = {"candidate_pool": "WRN top20 + ConvNeXt top10 + DenseNet top10",
              "feature_names": feature_names, "objective": args.objective, "folds": []}
    predictions = np.empty((len(y), 10), dtype=np.int32)
    all_scores = (np.empty(ids.shape, dtype=np.float32)
                  if args.save_scores is not None else None)
    for fold in range(3):
        train_rows = folds != fold
        test_rows = ~train_rows
        train_candidates = np.repeat(train_rows, ids.shape[1]) & keep
        common = dict(n_estimators=200, learning_rate=.05, num_leaves=15,
                      min_child_samples=1000, colsample_bytree=.8,
                      reg_lambda=2, n_jobs=16, verbosity=-1)
        if args.objective == "binary":
            model = lgb.LGBMClassifier(**common)
            model.fit(x[train_candidates], labels[train_candidates],
                      feature_name=feature_names,
                      categorical_feature=[len(feature_names)-1])
            score = model.predict_proba(x[np.repeat(test_rows, ids.shape[1])])[:, 1]
        else:
            model = lgb.LGBMRanker(objective="lambdarank", **common)
            group = valid[train_rows].sum(1).astype(np.int32)
            model.fit(x[train_candidates], labels[train_candidates], group=group,
                      feature_name=feature_names,
                      categorical_feature=[len(feature_names)-1])
            score = model.predict(x[np.repeat(test_rows, ids.shape[1])])
        score = score.reshape(-1, ids.shape[1])
        score[~valid[test_rows]] = -np.inf
        if all_scores is not None:
            all_scores[test_rows] = score
        predictions[test_rows] = np.take_along_axis(ids[test_rows],
                                    np.argsort(-score, axis=1)[:, :10], axis=1)
        true = y[test_rows]
        part = {"fold": fold, "count": len(true),
                "top1_correct": int((predictions[test_rows, 0] == true).sum()),
                "top10_correct": int((predictions[test_rows] == true[:, None]).any(1).sum())}
        result["folds"].append(part)
        print(part, flush=True)
    result["overall"] = {
        "count": len(y), "top1": float(np.mean(predictions[:, 0] == y)),
        "top10": float(np.mean((predictions == y[:, None]).any(1))),
        "candidate_oracle": float(np.mean(((ids == y[:, None]) & valid).any(1)))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if all_scores is not None:
        args.save_scores.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_scores, ids=ids.astype(np.uint16),
                            valid=valid, scores=all_scores,
                            labels=y.astype(np.uint16), folds=folds)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
