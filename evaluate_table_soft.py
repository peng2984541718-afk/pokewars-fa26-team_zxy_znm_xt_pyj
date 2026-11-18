"""End-to-end score of pair tables on the train_pair_mlp.py held-out battles.

Image posteriors are the out-of-fold WRN top-20 probabilities raised to a power
gamma, chosen on the selection half; scores are on the test half. Use only
tables that never saw the held-out battles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_soft_score import DENOMINATOR, expected_scores, rmse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tables", type=Path, nargs="+")
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv.gz"))
    parser.add_argument("--priors", type=Path, default=Path("outputs/large/wrn_top20_oof_compact.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/large/table_soft_eval.json"))
    parser.add_argument("--holdout", type=float, default=.2,
                        help="0.2 for train_pair_mlp.py tables, 0.05 for --full tables")
    args = parser.parse_args()
    rows = pd.read_csv(args.train_csv)
    identity = {}
    for side in (1, 2):
        for path, value in zip(rows[f"noisy_image{side}"], rows[f"pokemon_id{side}"]):
            if pd.notna(value):
                identity[path] = int(value)
    index = pd.Index(sorted(identity))
    true = np.array([identity[p] for p in index])
    first, second = index.get_indexer(rows.noisy_image1), index.get_indexer(rows.noisy_image2)
    known = (first >= 0) & (second >= 0)
    first, second = first[known], second[known]
    y = rows.result.to_numpy(np.float32)[known]
    permutation = np.random.default_rng(3780).permutation(len(y))
    valid = permutation[int((1 - args.holdout) * len(y)):]
    parts = {"select": valid[:len(valid) // 2], "test": valid[len(valid) // 2:]}
    with np.load(args.priors) as z:
        candidates = z["candidates"].astype(np.int64)
        wrn = np.maximum(z["probabilities"].astype(np.float32), 1e-12)
    report = {"image_top1": float(np.mean(candidates[:, 0] == true)), "tables": {}}
    for path in args.tables:
        table = np.load(path)
        result = {}
        for gamma in (.5, 1., 1.5, 2.):
            q = wrn ** gamma
            q /= q.sum(1, keepdims=True)
            result[gamma] = {}
            for name, rows_ in parts.items():
                u, v = first[rows_], second[rows_]
                soft = expected_scores(table, candidates[u], candidates[v], q[u], q[v])
                result[gamma][name] = rmse(soft, y[rows_])
        gamma = min(result, key=lambda g: result[g]["select"])
        test = parts["test"]
        u, v = first[test], second[test]
        report["tables"][str(path)] = {
            "gamma": gamma, "soft_test_rmse": result[gamma]["test"],
            "soft_test_score": 100 * (1 - result[gamma]["test"] / DENOMINATOR),
            "hard_test_score": 100 * (1 - rmse(table[candidates[u, 0], candidates[v, 0]], y[test]) / DENOMINATOR),
            "true_id_test_score": 100 * (1 - rmse(table[true[u], true[v]], y[test]) / DENOMINATOR)}
        print(path, json.dumps(report["tables"][str(path)]), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
