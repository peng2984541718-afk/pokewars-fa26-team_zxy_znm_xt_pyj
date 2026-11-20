"""Infer image identities jointly from image priors and battle results.

Each image is a node with a top-K candidate prior. Each battle (u, v, y) is a
factor psi(i, j) = (1 - eps) N(y; T[i, j], sigma^2) + eps / 10 over the two
images' candidates, where T is the species-pair score table. Loopy belief
propagation gives per-image posteriors.

--simulate: fold-0 labeled images with out-of-fold priors, fold-0/fold-0
battles, and a table trained without those battles. Reports accuracy.
--unlabeled: images from battles without species labels; writes pseudo-labels.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from image_experiment import SEED, stratified_folds


def belief_propagation(candidates, prior, eu, ev, y, table, sigma, eps, gamma,
                       iterations=20, damping=.5):
    """Return (N, K) posteriors. candidates hold species IDs indexing table."""
    device = prior.device
    pair = table[candidates[eu][:, :, None], candidates[ev][:, None, :]]
    gauss = torch.exp(-(y[:, None, None] - pair) ** 2 / (2 * sigma ** 2)) / (
        math.sqrt(2 * math.pi) * sigma)
    log_psi = torch.log((1 - eps) * gauss + eps / 10)
    log_prior = gamma * torch.log(prior.clamp_min(1e-12))
    n, k = prior.shape
    to_u = torch.zeros(len(eu), k, device=device)
    to_v = torch.zeros(len(eu), k, device=device)
    for _ in range(iterations):
        belief = log_prior.clone()
        belief.index_add_(0, eu, to_u)
        belief.index_add_(0, ev, to_v)
        cavity_u = torch.log_softmax(belief[eu] - to_u, 1)
        cavity_v = torch.log_softmax(belief[ev] - to_v, 1)
        new_u = torch.logsumexp(log_psi + cavity_v[:, None, :], 2)
        new_v = torch.logsumexp(log_psi + cavity_u[:, :, None], 1)
        new_u = new_u - torch.logsumexp(new_u, 1, keepdim=True)
        new_v = new_v - torch.logsumexp(new_v, 1, keepdim=True)
        to_u = damping * to_u + (1 - damping) * new_u
        to_v = damping * to_v + (1 - damping) * new_v
    belief = log_prior.clone()
    belief.index_add_(0, eu, to_u)
    belief.index_add_(0, ev, to_v)
    return torch.softmax(belief, 1)


def simulate(args, device):
    rows = pd.read_csv(args.train_csv)
    identity = {}
    for side in (1, 2):
        for path, value in zip(rows[f"noisy_image{side}"], rows[f"pokemon_id{side}"]):
            if pd.notna(value):
                identity[path] = int(value)
    paths = sorted(identity)
    true = np.array([identity[p] for p in paths], dtype=np.int64)
    folds = stratified_folds(true - 1, 3, SEED)
    index = pd.Index(paths)
    first, second = index.get_indexer(rows.noisy_image1), index.get_indexer(rows.noisy_image2)
    known = (first >= 0) & (second >= 0)
    first, second = first[known], second[known]
    y = rows.result.to_numpy(np.float32)[known]
    inside = (folds[first] == 0) & (folds[second] == 0)
    nodes = np.flatnonzero(folds == 0)
    local = np.full(len(paths), -1)
    local[nodes] = np.arange(len(nodes))
    with np.load(args.priors) as z:
        candidates = z["candidates"].astype(np.int64)[nodes]
        prior = z["probabilities"].astype(np.float32)[nodes]
    labels = true[nodes]
    if np.mean(candidates[:, 0] == labels) < .5:
        raise ValueError("Prior candidates do not look like 1-based species IDs")
    eu, ev, yy = local[first[inside]], local[second[inside]], y[inside]
    degree = np.bincount(np.r_[eu, ev], minlength=len(nodes))
    rng = np.random.default_rng(3780)
    half = rng.random(len(nodes)) < .5
    t = lambda v, dtype=None: torch.as_tensor(v, device=device, dtype=dtype)
    table = t(np.load(args.table), torch.float32)
    c, p = t(candidates), t(prior / prior.sum(1, keepdims=True))
    e_u, e_v, e_y = t(eu), t(ev), t(yy)
    base = candidates[:, 0] == labels
    report = {"images": len(nodes), "battles": int(inside.sum()),
              "mean_degree": float(degree.mean()),
              "prior_top1": float(base.mean()),
              "prior_top1_by_degree": {}, "grid": []}
    for d in (0, 1, 2, 3):
        mask = degree == d if d < 3 else degree >= 3
        report["prior_top1_by_degree"][f"{d}{'+' if d == 3 else ''}"] = {
            "images": int(mask.sum()), "accuracy": float(base[mask].mean())}
    for sigma, eps, gamma in itertools.product(args.sigmas, args.epsilons, args.gammas):
        posterior = belief_propagation(c, p, e_u, e_v, e_y, table, sigma, eps, gamma)
        top = candidates[np.arange(len(nodes)), posterior.argmax(1).cpu().numpy()]
        correct = top == labels
        record = {"sigma": sigma, "eps": eps, "gamma": gamma,
                  "select_top1": float(correct[half].mean()),
                  "report_top1": float(correct[~half].mean()),
                  "prior_report_top1": float(base[~half].mean()),
                  "by_degree": {f"{d}{'+' if d == 3 else ''}": float(
                      correct[~half & ((degree == d) if d < 3 else (degree >= 3))].mean())
                      for d in (0, 1, 2, 3)}}
        report["grid"].append(record)
        print(json.dumps(record), flush=True)
    report["selected"] = max(report["grid"], key=lambda r: r["select_top1"])
    return report


def unlabeled(args, device):
    rows = pd.read_csv(args.train_csv)
    unknown = rows.pokemon_id1.isna() & rows.pokemon_id2.isna()
    paths = args.unlabeled_paths.read_text().split()
    index = pd.Index(paths)
    eu = index.get_indexer(rows.noisy_image1[unknown])
    ev = index.get_indexer(rows.noisy_image2[unknown])
    if np.any(eu < 0) or np.any(ev < 0):
        raise ValueError("Unlabeled battle image missing from path list")
    y = rows.result[unknown].to_numpy(np.float32)
    candidates_frame = pd.read_csv(args.priors)
    if list(candidates_frame.image_path) != paths:
        raise ValueError("Prior file is not in unlabeled path order")
    candidates = np.array([json.loads(v) for v in candidates_frame.topk_pokemon_ids], dtype=np.int64)
    prior = np.array([json.loads(v) for v in candidates_frame.topk_probabilities], dtype=np.float32)
    t = lambda v, dtype=None: torch.as_tensor(v, device=device, dtype=dtype)
    posterior = belief_propagation(
        t(candidates), t(prior / prior.sum(1, keepdims=True)), t(eu), t(ev), t(y),
        t(np.load(args.table), torch.float32), args.sigmas[0], args.epsilons[0], args.gammas[0]
    ).cpu().numpy()
    best = posterior.argmax(1)
    pseudo = candidates[np.arange(len(paths)), best]
    confidence = posterior[np.arange(len(paths)), best]
    np.savez_compressed(args.output.with_suffix(".npz"), candidates=candidates.astype(np.uint16),
                        posterior=posterior.astype(np.float32), pseudo_label=pseudo.astype(np.uint16),
                        confidence=confidence.astype(np.float32),
                        battle_u=eu, battle_v=ev, battle_y=y)
    changed = pseudo != candidates[:, 0]
    return {"images": len(paths), "battles": int(unknown.sum()),
            "sigma": args.sigmas[0], "eps": args.epsilons[0], "gamma": args.gammas[0],
            "changed_from_image_top1": float(changed.mean()),
            "confidence_quantiles": np.quantile(confidence, [.05, .25, .5]).tolist(),
            "fraction_confidence_ge_0.9": float(np.mean(confidence >= .9))}


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--simulate", action="store_true")
    mode.add_argument("--unlabeled", action="store_true")
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv.gz"))
    parser.add_argument("--priors", type=Path, default=Path("outputs/large/wrn_top20_oof_compact.npz"))
    parser.add_argument("--table", type=Path, default=Path("outputs/large/pair_table_no_fold0.npy"))
    parser.add_argument("--unlabeled-paths", type=Path, default=Path("data/unlabeled_paths.txt"))
    parser.add_argument("--sigmas", type=float, nargs="+", default=[.3, .5, .8, 1.2])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[.01, .05, .2])
    parser.add_argument("--gammas", type=float, nargs="+", default=[.5, 1.])
    parser.add_argument("--output", type=Path, default=Path("outputs/large/joint_identity_sim.json"))
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = simulate(args, device) if args.simulate else unlabeled(args, device)
    report["config"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "grid"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
