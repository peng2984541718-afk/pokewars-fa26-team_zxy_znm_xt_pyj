"""Reproducible image-recognition and battle-score baseline for Pokéwars.

The Kaggle zip is read directly; its 389k small PNG files need not be extracted.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsqr


CSV_PREFIX = "data/data/"
# IDs are 1..1025; index zero is intentionally unused.
NUM_CLASSES = 1026
SEED = 3780


def read_tables(archive: ZipFile):
    tables = []
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        with archive.open(CSV_PREFIX + name) as source:
            tables.append(pd.read_csv(source))
    return tables


def image_paths(frame):
    return pd.concat((frame.noisy_image1, frame.noisy_image2), ignore_index=True)


def archive_name(path):
    # CSV: images/train/abc.png; Kaggle zip: data/data/train/abc.png.
    return CSV_PREFIX + path.removeprefix("images/")


def load_features(archive, paths, cache_file):
    if cache_file.exists():
        features = np.load(cache_file, mmap_mode="r")
        if features.shape == (len(paths), 256):
            print(f"Loaded {len(paths):,} cached image features", flush=True)
            return features
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(cache_file, mode="w+", dtype=np.uint8,
                                          shape=(len(paths), 256))
    start = time.monotonic()
    for idx, path in enumerate(paths):
        with archive.open(archive_name(path)) as source:
            with Image.open(io.BytesIO(source.read())) as image:
                pixels = np.asarray(image.convert("L"), dtype=np.uint16)
        if pixels.shape != (32, 32):
            raise ValueError(f"Unexpected shape for {path}: {pixels.shape}")
        features[idx] = pixels.reshape(16, 2, 16, 2).mean(axis=(1, 3)).reshape(-1)
        if (idx + 1) % 50000 == 0:
            print(f"Decoded {idx + 1:,}/{len(paths):,} images in "
                  f"{time.monotonic() - start:.0f}s", flush=True)
    features.flush()
    return features


def collect_labels(train, index):
    labels = np.full(len(index), -1, dtype=np.int16)
    for side in (1, 2):
        known = train[f"pokemon_id{side}"].notna()
        for path, label in zip(train.loc[known, f"noisy_image{side}"],
                               train.loc[known, f"pokemon_id{side}"].astype(int)):
            idx = index[path]
            if labels[idx] >= 0 and labels[idx] != label:
                raise ValueError(f"Conflicting identity labels for {path}")
            labels[idx] = label
    return labels


def centroids(features, labels, training_mask):
    selected = np.flatnonzero(training_mask & (labels >= 0))
    sums = np.zeros((NUM_CLASSES, features.shape[1]), dtype=np.float64)
    counts = np.bincount(labels[selected], minlength=NUM_CLASSES)
    # Batches avoid an extra full-size float image matrix.
    for block in np.array_split(selected, max(1, (len(selected) + 9999) // 10000)):
        np.add.at(sums, labels[block], np.asarray(features[block], dtype=np.float32))
    center = (sums / np.maximum(counts[:, None], 1)).astype(np.float32)
    return center, counts


def classify(features, indices, centers, counts, batch_size=2048):
    result = np.empty(len(indices), dtype=np.int16)
    center_norm = np.sum(centers * centers, axis=1)
    center_norm[counts == 0] = np.inf
    for offset in range(0, len(indices), batch_size):
        block = np.asarray(features[indices[offset:offset + batch_size]], dtype=np.float32)
        # Accelerate occasionally raises spurious floating-point warnings here;
        # the finite check still catches any actual invalid result.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            distances = center_norm[None, :] - 2 * block @ centers.T
        if np.isnan(distances).any():
            raise FloatingPointError("NaN in image-classification distances")
        result[offset:offset + len(block)] = np.argmin(distances, axis=1)
    return result


def fit_strengths(first, second, scores, regularization=10.0):
    """Fit score ~= 5 + strength[first] - strength[second]."""
    n = len(scores)
    row = np.repeat(np.arange(n), 2)
    col = np.column_stack((first, second)).reshape(-1)
    values = np.tile((1.0, -1.0), n)
    design = coo_matrix((values, (row, col)), shape=(n, NUM_CLASSES)).tocsr()
    ridge = coo_matrix((np.full(NUM_CLASSES, np.sqrt(regularization)),
                        (np.arange(NUM_CLASSES), np.arange(NUM_CLASSES))),
                       shape=(NUM_CLASSES, NUM_CLASSES)).tocsr()
    matrix = vstack((design, ridge), format="csr")
    target = np.concatenate((scores - 5.0, np.zeros(NUM_CLASSES)))
    return lsqr(matrix, target, atol=1e-6, btol=1e-6)[0]


def predict_scores(first, second, strengths):
    return np.clip(5.0 + strengths[first] - strengths[second], 0.0, 10.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("data/competition.zip"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--no-submission", action="store_true")
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")

    start = time.monotonic()
    with ZipFile(args.zip) as archive:
        train, test, sample = read_tables(archive)
        train_paths = pd.unique(image_paths(train)).tolist()
        test_paths = pd.unique(image_paths(test)).tolist()
        if set(train_paths) & set(test_paths):
            raise ValueError("Train and test image paths overlap")
        paths = train_paths + test_paths
        index = {path: idx for idx, path in enumerate(paths)}
        train_first = train.noisy_image1.map(index).to_numpy()
        train_second = train.noisy_image2.map(index).to_numpy()
        test_first = test.noisy_image1.map(index).to_numpy()
        test_second = test.noisy_image2.map(index).to_numpy()
        labels = collect_labels(train, index)
        features = load_features(archive, paths, Path("data/features_16x16.npy"))

    print(f"Rows: train={len(train):,}, test={len(test):,}; "
          f"images: train={len(train_paths):,}, test={len(test_paths):,}; "
          f"labeled images={np.sum(labels[:len(train_paths)] >= 0):,}", flush=True)
    if len(sample) != len(test) or list(sample.columns) != ["row_id", "result"]:
        raise ValueError("Unexpected sample submission format")

    rng = np.random.default_rng(SEED)
    # Assign each *image* to one fold. Every validation image is unseen by its
    # classifier, even when it occurs in multiple matchup rows.
    image_fold = rng.integers(args.folds, size=len(train_paths))
    oof_ids = np.empty(len(train_paths), dtype=np.int16)
    fold_metrics = []
    for fold in range(args.folds):
        training_mask = np.zeros(len(paths), dtype=bool)
        training_mask[:len(train_paths)] = image_fold != fold
        centers, counts = centroids(features, labels, training_mask)
        validation_images = np.flatnonzero(image_fold == fold)
        oof_ids[validation_images] = classify(features, validation_images, centers, counts)
        known = labels[validation_images] >= 0
        accuracy = np.mean(oof_ids[validation_images[known]] == labels[validation_images[known]])
        print(f"Image fold {fold + 1}/{args.folds}: identity accuracy={accuracy:.4f} "
              f"on {known.sum():,} labeled images", flush=True)

    Path("outputs").mkdir(exist_ok=True)
    pd.DataFrame({"image_path": train_paths, "predicted_pokemon_id": oof_ids}).to_csv(
        "outputs/train_oof_identity.csv", index=False)

    score_fold = rng.permutation(len(train)) % args.folds
    known_pair = train.pokemon_id1.notna() & train.pokemon_id2.notna()
    known_pair = known_pair.to_numpy()
    true_first = train.pokemon_id1.fillna(-1).to_numpy(dtype=np.int16)
    true_second = train.pokemon_id2.fillna(-1).to_numpy(dtype=np.int16)
    actual = train.result.to_numpy(dtype=np.float64)
    oof_score = np.empty(len(train), dtype=np.float64)
    for fold in range(args.folds):
        fit = (score_fold != fold) & known_pair
        valid = score_fold == fold
        strengths = fit_strengths(true_first[fit], true_second[fit], actual[fit])
        predictions = predict_scores(oof_ids[train_first[valid]],
                                     oof_ids[train_second[valid]], strengths)
        oof_score[valid] = predictions
        rmse = float(np.sqrt(np.mean((predictions - actual[valid]) ** 2)))
        oracle = predict_scores(true_first[valid & known_pair],
                                true_second[valid & known_pair], strengths)
        oracle_rmse = float(np.sqrt(np.mean((oracle - actual[valid & known_pair]) ** 2)))
        fold_metrics.append({"fold": fold + 1, "rows": int(valid.sum()),
                             "rmse": rmse, "rmse_with_true_ids": oracle_rmse})
        print(f"Score fold {fold + 1}/{args.folds}: RMSE={rmse:.4f}; "
              f"true-ID RMSE={oracle_rmse:.4f}", flush=True)

    delta = oof_score - 5.0
    # A single scalar blend learned from out-of-fold predictions guards against
    # weak image recognition making a submission worse than the constant prior.
    blend = float(np.clip(np.dot(actual - 5.0, delta) /
                          max(np.dot(delta, delta), 1e-12), 0.0, 1.0))
    blended_oof = np.clip(5.0 + blend * delta, 0.0, 10.0)
    metrics = {"seed": SEED, "folds": fold_metrics,
               "mean_raw_rmse": float(np.mean([item["rmse"] for item in fold_metrics])),
               "constant_5_rmse": float(np.sqrt(np.mean((actual - 5.0) ** 2))),
               "blend": blend,
               "blended_oof_rmse": float(np.sqrt(np.mean((blended_oof - actual) ** 2))),
               "oof_identity_accuracy": float(np.mean(oof_ids[labels[:len(train_paths)] >= 0]
                                                   == labels[:len(train_paths)][labels[:len(train_paths)] >= 0])),
               "elapsed_seconds": round(time.monotonic() - start, 1)}
    Path("outputs/cv_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Raw CV RMSE={metrics['mean_raw_rmse']:.4f}; "
          f"blended OOF RMSE={metrics['blended_oof_rmse']:.4f}; "
          f"blend={blend:.4f}; identity accuracy={metrics['oof_identity_accuracy']:.4f}",
          flush=True)

    if not args.no_submission:
        training_mask = np.zeros(len(paths), dtype=bool)
        training_mask[:len(train_paths)] = True
        centers, counts = centroids(features, labels, training_mask)
        test_indices = np.arange(len(train_paths), len(paths))
        test_ids = classify(features, test_indices, centers, counts)
        pd.DataFrame({"image_path": test_paths, "predicted_pokemon_id": test_ids}).to_csv(
            "outputs/test_identity.csv", index=False)
        strengths = fit_strengths(true_first[known_pair], true_second[known_pair], actual[known_pair])
        raw_test = predict_scores(test_ids[test_first - len(train_paths)],
                                  test_ids[test_second - len(train_paths)], strengths)
        predicted = np.clip(5.0 + blend * (raw_test - 5.0), 0.0, 10.0)
        submission = sample[["row_id"]].copy()
        submission["result"] = predicted
        submission.to_csv("outputs/submission.csv", index=False)
        print(f"Wrote outputs/submission.csv with {len(submission):,} rows", flush=True)


if __name__ == "__main__":
    main()
