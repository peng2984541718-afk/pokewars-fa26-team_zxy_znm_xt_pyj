"""Fold-0 score pilot of evaluate_score_fold0.py with the neural pair model.

Same rows, splits, image ranker scores and posterior grid as that script; only
the species-pair score table changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_score_models import PairScore
from evaluate_soft_score import DENOMINATOR, expected_scores, rmse
from image_experiment import SEED, stratified_folds
from train_pair_mlp import PairMLP, train


@torch.no_grad()
def model_table(model, device: str) -> np.ndarray:
    model.eval()
    left = torch.arange(1026, device=device).repeat_interleave(1026)
    right = torch.arange(1026, device=device).repeat(1026)
    values = torch.cat([model(left[i:i + 262144], right[i:i + 262144])
                        for i in range(0, len(left), 262144)])
    return values.reshape(1026, 1026).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv.gz"))
    parser.add_argument("--ranker-scores", type=Path, default=Path(
        "outputs/large/meta_top20_fold0_scores.npz"))
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/large/score_fold0_mlp_soft.json"))
    parser.add_argument("--model", choices=("pairscore", "mlp"), default="mlp")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--loss", choices=("mse", "bce"), default="bce")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--save-table", type=Path, default=None,
                        help="Save the 1026x1026 table; it never saw fold-0/fold-0 battles")
    parser.add_argument("--table", type=Path, default=None,
                        help="Precomputed table that never saw fold-0/fold-0 battles")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = pd.read_csv(args.train_csv)
    identity = {}
    for side in (1, 2):
        for path, value in zip(rows[f"noisy_image{side}"], rows[f"pokemon_id{side}"]):
            if pd.notna(value):
                identity[path] = int(value)
    paths = sorted(identity)
    true = np.array([identity[path] for path in paths], dtype=np.int16)
    folds = stratified_folds(true - 1, 3, SEED)
    path_index = pd.Index(paths)
    first = path_index.get_indexer(rows.noisy_image1)
    second = path_index.get_indexer(rows.noisy_image2)
    known = (first >= 0) & (second >= 0)
    first, second = first[known], second[known]
    a, b = true[first], true[second]
    y = rows.result.to_numpy(dtype=np.float32)[known]
    pair_fold0 = (folds[first] == 0) & (folds[second] == 0)
    rng = np.random.default_rng(3780)
    train_rows = np.flatnonzero(~pair_fold0)
    rng.shuffle(train_rows)
    cut = int(.9 * len(train_rows))
    fit, early = train_rows[:cut], train_rows[cut:]
    pair_rows = np.flatnonzero(pair_fold0)
    rng.shuffle(pair_rows)
    split = len(pair_rows) // 2

    tensors = lambda idx: (torch.as_tensor(a[idx], dtype=torch.long, device=device),
                           torch.as_tensor(b[idx], dtype=torch.long, device=device),
                           torch.as_tensor(y[idx], device=device))
    data = {"train": tensors(fit), "select": tensors(early)}
    tables, fits = [], []
    for seed in ([] if args.table is not None else args.seeds):
        torch.manual_seed(seed)
        if args.model == "pairscore":
            model = PairScore(64).to(device)
            lr, wd, batch = .02, .01, 8192
        else:
            model = PairMLP(args.dim, args.hidden, args.layers, args.dropout).to(device)
            lr, wd, batch = args.lr, args.weight_decay, args.batch_size
        best, best_epoch, epochs_run = train(model, data, args, lr, wd, args.epochs, batch)
        fits.append({"seed": seed, "early_stop_rmse": best, "best_epoch": best_epoch})
        tables.append(model_table(model, device))
        print(json.dumps(fits[-1]), flush=True)
    if args.table is not None:
        table = np.load(args.table)
        fits = [{"table": str(args.table)}]
    else:
        table = np.mean(tables, axis=0)
    if args.save_table is not None and args.table is None:
        np.save(args.save_table, table)

    with np.load(args.ranker_scores) as z:
        valid_idx = z["valid_indices"]
        candidates = z["candidates"].astype(np.int16)
        wrn = z["wrn_probs"].astype(np.float32)
        ranker = z["ranker_scores"].astype(np.float32)
        meta = z["meta_top1"].astype(np.int16)
        if not np.array_equal(true[valid_idx], z["labels"]):
            raise ValueError("Fold-0 ranker scores do not match labeled image indices")
    position = np.full(len(paths), -1, dtype=np.int32)
    position[valid_idx] = np.arange(len(valid_idx))
    one, two = position[first[pair_rows]], position[second[pair_rows]]
    if np.any(one < 0) or np.any(two < 0):
        raise ValueError("A fold-0 pair contains an image outside fold 0")
    ids1, ids2 = candidates[one], candidates[two]
    actual = y[pair_rows]
    hard = table[meta[one], meta[two]]
    oracle = table[a[pair_rows], b[pair_rows]]
    wrn = np.maximum(wrn, 1e-12)
    wrn /= wrn.sum(1, keepdims=True)
    ranker = np.maximum(ranker, 1e-12)
    hard_onehot = (candidates == meta[:, None]).astype(np.float32)

    def posterior(gamma, ranker_fraction, hard_mass):
        r = ranker ** gamma
        r /= r.sum(1, keepdims=True)
        soft = ranker_fraction * r + (1 - ranker_fraction) * wrn
        return hard_mass * hard_onehot + (1 - hard_mass) * soft

    trials = []
    for gamma in (.5, 1., 2., 4., 8.):
        for ranker_fraction in (0., .5, 1.):
            for hard_mass in (0., .25, .5):
                q = posterior(gamma, ranker_fraction, hard_mass)
                prediction = expected_scores(table, ids1, ids2, q[one], q[two])
                trials.append({"gamma": gamma, "ranker_fraction": ranker_fraction,
                               "hard_mass": hard_mass,
                               "tune_rmse": rmse(prediction[:split], actual[:split]),
                               "test_rmse": rmse(prediction[split:], actual[split:])})
    selected = min(trials, key=lambda item: item["tune_rmse"])
    q = posterior(selected["gamma"], selected["ranker_fraction"], selected["hard_mass"])
    soft = expected_scores(table, ids1, ids2, q[one], q[two])
    wrong = (meta[one] != a[pair_rows]) | (meta[two] != b[pair_rows])
    test = slice(split, None)
    groups = {name: {"count": int(mask[test].sum()),
                     "hard_rmse": rmse(hard[test][mask[test]], actual[test][mask[test]]),
                     "soft_rmse": rmse(soft[test][mask[test]], actual[test][mask[test]])}
              for name, mask in (("both_images_correct", ~wrong),
                                 ("at_least_one_image_wrong", wrong))}
    score = lambda value: 100 * (1 - value / DENOMINATOR)
    report = {
        "pair_model": vars(args) | {"train_csv": str(args.train_csv),
                                     "ranker_scores": str(args.ranker_scores),
                                     "output": str(args.output),
                                     "save_table": str(args.save_table),
                                     "table": str(args.table)},
        "fits": fits,
        "true_ids_test_rmse": rmse(oracle[test], actual[test]),
        "hard_meta_test_rmse": rmse(hard[test], actual[test]),
        "selected_soft": selected,
        "true_ids_test_score": score(rmse(oracle[test], actual[test])),
        "hard_test_score": score(rmse(hard[test], actual[test])),
        "soft_test_score": score(selected["test_rmse"]),
        "test_error_groups": groups,
        "method": "Same split as evaluate_score_fold0.py; parameters chosen on the first half of fold-0/fold-0 battles, scores on the second half.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
