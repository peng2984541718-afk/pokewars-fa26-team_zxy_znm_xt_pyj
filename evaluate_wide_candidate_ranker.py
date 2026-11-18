"""Cross-fit a top-10 selector over WRN/ConvNeXt top-20 candidate unions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from image_experiment import SEED, stratified_folds


def ranked_ids(probabilities: np.ndarray, k: int) -> np.ndarray:
    ids = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    probs = np.take_along_axis(probabilities, ids, axis=1)
    return np.take_along_axis(ids, np.argsort(-probs, axis=1), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/large"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/large/wide_candidate_ranker_cv.json"))
    args = parser.parse_args()
    y = np.load(args.data_dir / "labeled_ids.npy").astype(np.int64)
    folds = stratified_folds(y, 3, SEED)
    wrn = np.load(args.input_dir / "multibackbone_wrn_rot5_oof.npy", mmap_mode="r")
    conv = np.load(args.input_dir / "multibackbone_convnextv2_oof.npy", mmap_mode="r")
    wrn_ids = ranked_ids(wrn, 20)
    conv_ids = ranked_ids(conv, 20)
    ids = np.concatenate((wrn_ids, conv_ids), axis=1)
    valid = np.ones(ids.shape, dtype=bool)
    valid[:, 20:] = ~(conv_ids[:, :, None] == wrn_ids[:, None, :]).any(2)
    wp = np.take_along_axis(wrn, ids, axis=1).astype(np.float32)
    cp = np.take_along_axis(conv, ids, axis=1).astype(np.float32)
    wrn_rank = np.concatenate((np.broadcast_to(np.arange(1, 21), wrn_ids.shape),
                               np.full(conv_ids.shape, 21)), axis=1).astype(np.float32)
    conv_rank = np.concatenate((np.full(wrn_ids.shape, 21),
                                np.broadcast_to(np.arange(1, 21), conv_ids.shape)),
                               axis=1).astype(np.float32)
    for j in range(20):
        match = wrn_ids == conv_ids[:, j:j+1]
        conv_rank[:, :20][match] = j + 1
        match = conv_ids == wrn_ids[:, j:j+1]
        wrn_rank[:, 20:][match] = j + 1
    wrn_conf = np.max(wrn, axis=1).astype(np.float32)
    conv_conf = np.max(conv, axis=1).astype(np.float32)
    wrn_second = np.take_along_axis(wrn, wrn_ids[:, 1:2], axis=1)[:, 0].astype(np.float32)
    conv_second = np.take_along_axis(conv, conv_ids[:, 1:2], axis=1)[:, 0].astype(np.float32)
    log_wp = np.log(np.maximum(wp, 1e-12))
    log_cp = np.log(np.maximum(cp, 1e-12))
    names = ("log_wrn_p", "log_conv_p", "wrn_rank", "conv_rank",
             "log_wrn_relative", "log_conv_relative", "wrn_conf", "conv_conf",
             "wrn_top1_margin", "conv_top1_margin", "blended_probability",
             "candidate_id")
    columns = [
        log_wp, log_cp, wrn_rank, conv_rank,
        log_wp - np.log(np.maximum(wrn_conf[:, None], 1e-12)),
        log_cp - np.log(np.maximum(conv_conf[:, None], 1e-12)),
        np.broadcast_to(wrn_conf[:, None], ids.shape),
        np.broadcast_to(conv_conf[:, None], ids.shape),
        np.broadcast_to((wrn_conf-wrn_second)[:, None], ids.shape),
        np.broadcast_to((conv_conf-conv_second)[:, None], ids.shape),
        .8*wp+.2*cp, ids.astype(np.float32),
    ]
    x = np.stack(columns, axis=2).reshape(-1, len(names))
    labels = (ids == y[:, None]).reshape(-1).astype(np.uint8)
    keep = valid.reshape(-1)
    output = {"features": names, "pool": "WRN top20 union ConvNeXt top20",
              "folds": []}
    predicted_ids = np.empty((len(y), 10), dtype=np.int32)
    for fold in range(3):
        train = np.repeat(folds != fold, ids.shape[1]) & keep
        valid_rows = folds == fold
        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                   num_leaves=15, min_child_samples=1000,
                                   colsample_bytree=.8, reg_lambda=2,
                                   n_jobs=16, verbosity=-1)
        model.fit(x[train], labels[train], feature_name=list(names),
                  categorical_feature=[len(names) - 1])
        scores = model.predict_proba(x[np.repeat(valid_rows, ids.shape[1])])[:, 1]
        scores = scores.reshape(-1, ids.shape[1])
        scores[~valid[valid_rows]] = -1
        chosen = np.argsort(-scores, axis=1)[:, :10]
        predicted_ids[valid_rows] = np.take_along_axis(ids[valid_rows], chosen, axis=1)
        true = y[valid_rows]
        part = {"fold": fold, "count": len(true),
                "wrn_top1": float(np.mean(wrn_ids[valid_rows, 0] == true)),
                "ranked_top1": float(np.mean(predicted_ids[valid_rows, 0] == true)),
                "wrn_top10": float(np.mean((wrn_ids[valid_rows, :10] == true[:, None]).any(1))),
                "ranked_top10": float(np.mean((predicted_ids[valid_rows] == true[:, None]).any(1)))}
        output["folds"].append(part)
        print(part, flush=True)
    output["overall"] = {
        "count": len(y), "wrn_top1": float(np.mean(wrn_ids[:, 0] == y)),
        "ranked_top1": float(np.mean(predicted_ids[:, 0] == y)),
        "wrn_top10": float(np.mean((wrn_ids[:, :10] == y[:, None]).any(1))),
        "ranked_top10": float(np.mean((predicted_ids == y[:, None]).any(1))),
        "union_oracle": float(np.mean(((ids == y[:, None]) & valid).any(1)))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
