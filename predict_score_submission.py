"""Expected battle scores for the test set from image posteriors and the pair model.

The pair model is trained on all battles with two known species IDs, minus a
small holdout used only for early stopping. Each test image's posterior is its
normalized top-10 ranker scores, the choice selected in evaluate_score_fold0_mlp.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_score_fold0_mlp import model_table
from evaluate_soft_score import expected_scores
from train_pair_mlp import PairMLP, rmse, predict, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battles", type=Path, default=Path("data/known_battles.npz"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv.gz"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "outputs/large/test_meta_three_arch_top20_160000_top10.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/large/submission_pair_mlp.csv"))
    parser.add_argument("--dim", type=int, default=24)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=.2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--holdout", type=float, default=.05)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--table", type=Path, default=None,
                        help="Use a saved pair table instead of training one")
    args = parser.parse_args()
    args.loss = "bce"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with np.load(args.battles) as z:
        a, b, y = z["a"].astype(np.int64), z["b"].astype(np.int64), z["y"].astype(np.float32)
    permutation = np.random.default_rng(3780).permutation(len(y))
    cut = int((1 - args.holdout) * len(y))
    tensors = lambda idx: tuple(torch.as_tensor(v[idx], device=device) for v in (a, b, y))
    data = {"train": tensors(permutation[:cut]), "select": tensors(permutation[cut:])}
    tables, fits = [], []
    if args.table is not None:
        tables, args.seeds = [np.load(args.table)], []
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = PairMLP(args.dim, args.hidden, args.layers, args.dropout).to(device)
        best, best_epoch, _ = train(model, data, args, args.lr, args.weight_decay,
                                    args.epochs, 2048)
        fits.append({"seed": seed, "holdout_rmse": best, "best_epoch": best_epoch})
        tables.append(model_table(model, device))
        print(json.dumps(fits[-1]), flush=True)
    table = np.mean(tables, axis=0)
    holdout = permutation[cut:]
    ensemble_holdout = rmse(torch.as_tensor(table[a[holdout], b[holdout]]),
                            torch.as_tensor(y[holdout]))

    candidates = pd.read_csv(args.candidates)
    ids = np.array([json.loads(v) for v in candidates.top10_pokemon_ids], dtype=np.int64)
    scores = np.array([json.loads(v) for v in candidates.top10_ranker_scores], dtype=np.float64)
    posterior = np.maximum(scores, 1e-12) ** args.gamma
    posterior = (posterior / posterior.sum(1, keepdims=True)).astype(np.float32)
    index = pd.Index(candidates.image_path)
    test = pd.read_csv(args.test_csv)
    one, two = index.get_indexer(test.noisy_image1), index.get_indexer(test.noisy_image2)
    if np.any(one < 0) or np.any(two < 0):
        raise ValueError("A test image is missing from the candidate file")
    result = expected_scores(table, ids[one], ids[two], posterior[one], posterior[two])
    submission = pd.DataFrame({"row_id": [f"test-{i:06d}" for i in range(len(test))],
                               "result": np.clip(result, 0, 10)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    if args.table is None:
        np.save(args.output.with_name("pair_mlp_table.npy"), table)
    report = {"fits": fits, "ensemble_holdout_rmse": ensemble_holdout,
              "rows": len(submission), "mean_result": float(submission.result.mean()),
              "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
