"""Add pseudo-labeled battles without species IDs to the pair-model training.

Clean protocol on the train_pair_mlp.py split: a base model trained on the
80% split builds the table for joint inference on the unlabeled battles, so
the held-out battles never influence the pseudo-labels. With --full, the base
model is trained on all labeled battles except a small early-stop holdout,
and the final tables are saved for test prediction.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_score_fold0_mlp import model_table
from image_experiment import SEED, stratified_folds
from joint_identity_inference import belief_propagation
from train_pair_mlp import PairMLP, predict, rmse, train


def fit_tables(data, args, device, seeds):
    tables, fits = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        model = PairMLP(args.dim, args.hidden, args.layers, args.dropout).to(device)
        start = time.monotonic()
        best, best_epoch, _ = train(model, data, args, args.lr, args.weight_decay,
                                    args.epochs, 2048)
        fits.append({"seed": seed, "select_rmse": best, "best_epoch": best_epoch,
                     "seconds": round(time.monotonic() - start, 1)})
        tables.append(model_table(model, device))
        print(json.dumps(fits[-1]), flush=True)
    return np.mean(tables, axis=0), fits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battles", type=Path, default=Path("data/known_battles.npz"))
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv.gz"))
    parser.add_argument("--priors", type=Path, default=Path("outputs/large/unlabeled_wrn_rot5_top20.csv"))
    parser.add_argument("--unlabeled-paths", type=Path, default=Path("data/unlabeled_paths.txt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/large/pair_semi.json"))
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--exclude-fold0-pairs", action="store_true",
                        help="Train without fold-0/fold-0 battles; test RMSE is on them")
    parser.add_argument("--holdout", type=float, default=.05)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--min-weight", type=float, default=.5)
    parser.add_argument("--dim", type=int, default=24)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=.3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--initial-table", type=Path, default=None,
                        help="Use this table as round 0 instead of training one")
    parser.add_argument("--pseudo-battles", type=Path, default=None,
                        help="Precomputed constrained-BP battles .npz (a, b, y, weight)")
    args = parser.parse_args()
    args.loss = "bce"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = lambda v, dtype=None: torch.as_tensor(v, device=device, dtype=dtype)

    with np.load(args.battles) as z:
        a, b, y = z["a"].astype(np.int64), z["b"].astype(np.int64), z["y"].astype(np.float32)
    permutation = np.random.default_rng(3780).permutation(len(y))
    if args.exclude_fold0_pairs:
        # Same rows as evaluate_score_fold0.py: never train on fold-0/fold-0 battles.
        rows = pd.read_csv(args.train_csv)
        identity = {}
        for side in (1, 2):
            for path, value in zip(rows[f"noisy_image{side}"], rows[f"pokemon_id{side}"]):
                if pd.notna(value):
                    identity[path] = int(value)
        paths = sorted(identity)
        folds = stratified_folds(np.array([identity[p] for p in paths]) - 1, 3, SEED)
        index = pd.Index(paths)
        first, second = index.get_indexer(rows.noisy_image1), index.get_indexer(rows.noisy_image2)
        known = (first >= 0) & (second >= 0)
        first, second = first[known], second[known]
        if not np.array_equal(np.array([identity[p] for p in index[first]]), a):
            raise ValueError("known_battles.npz is not in train.csv known-row order")
        pair_fold0 = (folds[first] == 0) & (folds[second] == 0)
        rng = np.random.default_rng(3780)
        outside = np.flatnonzero(~pair_fold0)
        rng.shuffle(outside)
        cut = int(.9 * len(outside))
        train_idx, select_idx = outside[:cut], outside[cut:]
        test_idx = np.flatnonzero(pair_fold0)
    elif args.full:
        cut = int((1 - args.holdout) * len(y))
        train_idx, select_idx, test_idx = permutation[:cut], permutation[cut:], permutation[cut:]
    else:
        cut = int(.8 * len(y))
        valid = permutation[cut:]
        train_idx, select_idx, test_idx = permutation[:cut], valid[:len(valid) // 2], valid[len(valid) // 2:]

    rows = pd.read_csv(args.train_csv)
    unknown = rows.pokemon_id1.isna() & rows.pokemon_id2.isna()
    index = pd.Index(args.unlabeled_paths.read_text().split())
    eu = index.get_indexer(rows.noisy_image1[unknown])
    ev = index.get_indexer(rows.noisy_image2[unknown])
    uy = rows.result[unknown].to_numpy(np.float32)
    if args.pseudo_battles is None:
        frame = pd.read_csv(args.priors)
        if not frame.image_path.equals(pd.Series(index, name="image_path")):
            raise ValueError("Prior file is not in unlabeled path order")
        candidates = np.array([json.loads(v) for v in frame.topk_pokemon_ids], dtype=np.int64)
        prior = np.array([json.loads(v) for v in frame.topk_probabilities], dtype=np.float32)
        prior /= prior.sum(1, keepdims=True)

    base = {"train": (t(a[train_idx]), t(b[train_idx]), t(y[train_idx])),
            "select": (t(a[select_idx]), t(b[select_idx]), t(y[select_idx]))}
    if args.initial_table is not None:
        table, fits = np.load(args.initial_table), [{"initial_table": str(args.initial_table)}]
    else:
        table, fits = fit_tables(base, args, device, args.seeds)
    np.save(args.output.with_name(f"{args.output.stem}_table_round0.npy"), table)
    history = [{"round": 0, "fits": fits,
                "test_rmse": rmse(t(table[a[test_idx], b[test_idx]]), t(y[test_idx]))}]
    print(json.dumps(history[-1]), flush=True)
    for round_id in range(1, args.rounds + 1):
        if args.pseudo_battles is not None:
            with np.load(args.pseudo_battles) as pb:
                pa = pb["a"].astype(np.int64)
                pb_id = pb["b"].astype(np.int64)
                py_val = pb["y"].astype(np.float32)
                weight = pb["weight"].astype(np.float32)
            keep = weight >= args.min_weight
            data = {"train": (t(np.r_[a[train_idx], pa[keep]]),
                              t(np.r_[b[train_idx], pb_id[keep]]),
                              t(np.r_[y[train_idx], py_val[keep]]),
                              t(np.r_[np.ones(len(train_idx), np.float32), weight[keep]])),
                    "select": base["select"]}
        else:
            posterior = belief_propagation(t(candidates), t(prior), t(eu), t(ev), t(uy),
                                           t(table, torch.float32), .5, .05, 1.).cpu().numpy()
            best = posterior.argmax(1)
            pseudo = candidates[np.arange(len(best)), best]
            confidence = posterior[np.arange(len(best)), best]
            weight = confidence[eu] * confidence[ev]
            keep = weight >= args.min_weight
            data = {"train": (t(np.r_[a[train_idx], pseudo[eu][keep]]),
                              t(np.r_[b[train_idx], pseudo[ev][keep]]),
                              t(np.r_[y[train_idx], uy[keep]]),
                              t(np.r_[np.ones(len(train_idx), np.float32), weight[keep]])),
                    "select": base["select"]}
        table, fits = fit_tables(data, args, device, args.seeds)
        np.save(args.output.with_name(f"{args.output.stem}_table_round{round_id}.npy"), table)
        history.append({"round": round_id, "fits": fits,
                        "pseudo_battles": int(keep.sum()),
                        "mean_pseudo_weight": float(weight[keep].mean()),
                        "test_rmse": rmse(t(table[a[test_idx], b[test_idx]]), t(y[test_idx]))})
        print(json.dumps(history[-1]), flush=True)
    if args.full and args.pseudo_battles is None:
        np.savez_compressed(args.output.with_name(f"{args.output.stem}_pseudo_labels.npz"),
                            pseudo_label=pseudo.astype(np.uint16), confidence=confidence,
                            candidates=candidates.astype(np.uint16), posterior=posterior)
    report = {"history": history,
              "protocol": ("no fold-0/fold-0 battles; test = those battles" if args.exclude_fold0_pairs else
                           "full data, 5% early-stop holdout" if args.full else
                           "train_pair_mlp 80/20 split; test = second half of validation"),
              "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
