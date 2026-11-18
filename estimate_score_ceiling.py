"""Estimate the score ceiling imposed by OOF image identity mistakes.

For misidentified battles, use an observed score of the predicted ordered
species pair when that pair occurs elsewhere in the labeled training data.
This is a counterfactual lookup estimate, not a measured Kaggle score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


DENOMINATOR = 4.382900214041612


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-zip", type=Path, default=Path("data/competition.zip"))
    parser.add_argument("--image-oof", type=Path, default=Path(
        "outputs/large/meta_rerank_three_arch_top20_oof.npz"))
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/large/score_ceiling_estimate.json"))
    args = parser.parse_args()
    with ZipFile(args.data_zip) as archive:
        rows = pd.read_csv(archive.open("data/data/train.csv"))
    labels = {}
    for side in (1, 2):
        for path, value in zip(rows[f"noisy_image{side}"], rows[f"pokemon_id{side}"]):
            if pd.notna(value):
                labels[path] = int(value)
    paths = sorted(labels)
    true = np.array([labels[path] for path in paths], dtype=np.int16)
    with np.load(args.image_oof) as z:
        pred = np.zeros(len(paths), dtype=np.int16)
        pred[z["indices"]] = z["meta_top1"]
        if not np.array_equal(true[z["indices"]], z["labels"]):
            raise ValueError("Image OOF indices do not match true IDs")
    path_index = pd.Index(paths)
    first = path_index.get_indexer(rows.noisy_image1)
    second = path_index.get_indexer(rows.noisy_image2)
    known = (first >= 0) & (second >= 0)
    first, second = first[known], second[known]
    a, b = true[first], true[second]
    pa, pb = pred[first], pred[second]
    score = rows.result.to_numpy(dtype=np.float32)[known]
    wrong = (a != pa) | (b != pb)
    # Ordered pairs only. No unverified score-reversal assumption is needed.
    observed = np.full((1026, 1026), np.nan, dtype=np.float32)
    observed[a, b] = score
    counterfactual = observed[pa, pb]
    lookup = wrong & np.isfinite(counterfactual)
    squared_error = (counterfactual[lookup]-score[lookup])**2
    wrong_rate = float(wrong.mean())
    rmse = float(np.sqrt(wrong_rate*squared_error.mean()))
    rng = np.random.default_rng(3780)
    bootstrap = np.empty(2000, dtype=np.float64)
    for trial in range(len(bootstrap)):
        sample = squared_error[rng.integers(len(squared_error), size=len(squared_error))]
        bootstrap[trial] = 100*(1-np.sqrt(wrong_rate*sample.mean())/DENOMINATOR)
    report = {
        "known_identity_battles":int(known.sum()),
        "image_oof_top1":float(np.mean(pred == true)),
        "battles_with_at_least_one_wrong_identity":int(wrong.sum()),
        "wrong_battle_fraction":wrong_rate,
        "wrong_battles_with_observed_counterfactual":int(lookup.sum()),
        "counterfactual_coverage_among_wrong":float(lookup.sum()/wrong.sum()),
        "counterfactual_error_rmse":float(np.sqrt(squared_error.mean())),
        "estimated_perfect_score_model_rmse":rmse,
        "estimated_score":100*(1-rmse/DENOMINATOR),
        "bootstrap_95_score":np.quantile(bootstrap, [.025, .975]).tolist(),
        "assumptions":[
            "Battle result is fixed by the two true species IDs",
            "Observed wrong-pair lookups represent unobserved wrong pairs",
            "Validation image error distribution resembles the test set",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
