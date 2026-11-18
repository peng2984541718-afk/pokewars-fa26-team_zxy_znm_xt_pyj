"""Neural matrix completion for battle scores with true species IDs.

Uses the same 80/20 split as evaluate_score_models.py. Early stopping sees only
the first half of the validation rows; the second half is a clean test.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from evaluate_score_models import PairScore


class PairMLP(nn.Module):
    """f(a, b) = 5 + 5 tanh(h(a, b) - h(b, a)), so f(a, b) + f(b, a) = 10."""

    def __init__(self, dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(1026, dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        blocks, width = [], 3 * dim
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout)]
            width = hidden
        blocks.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*blocks)

    def half(self, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((ea, eb, ea * eb), -1)).squeeze(-1)

    def logit(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ea, eb = self.embedding(a), self.embedding(b)
        return self.half(ea, eb) - self.half(eb, ea)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return 10 * torch.sigmoid(self.logit(a, b))


def rmse(p: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((p - y) ** 2)))


@torch.no_grad()
def predict(model, a, b, batch=65536):
    model.eval()
    return torch.cat([model(a[i:i + batch], b[i:i + batch])
                      for i in range(0, len(a), batch)])


def train(model, data, args, lr, weight_decay, epochs, batch_size):
    ta, tb, ty, *rest = data["train"]
    tw = rest[0] if rest else None
    sa, sb, sy = data["select"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(ty) / batch_size)
    total = steps_per_epoch * epochs
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total, pct_start=0.05)
    best, best_epoch, state = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(ty), device=ty.device)
        for part in order.split(batch_size):
            # Every battle also gives the reversed battle with score 10 - y.
            a, b, y = ta[part], tb[part], ty[part]
            flip = torch.rand(len(part), device=ty.device) < 0.5
            a, b = torch.where(flip, b, a), torch.where(flip, a, b)
            y = torch.where(flip, 10 - y, y)
            if args.loss == "bce" and hasattr(model, "logit"):
                # The score is 10 * wins / 1024, so BCE is the binomial likelihood.
                loss = nn.functional.binary_cross_entropy_with_logits(
                    model.logit(a, b), y / 10, reduction="none")
            else:
                loss = (model(a, b) - y) ** 2
            if tw is None:
                loss = loss.mean()
            else:
                w = tw[part]
                loss = (w * loss).sum() / w.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
        error = rmse(predict(model, sa, sb), sy)
        if error < best - 1e-5:
            best, best_epoch = error, epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(state)
    return best, best_epoch, epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battles", type=Path, default=Path("data/known_battles.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/large/pair_mlp_sweep.json"))
    parser.add_argument("--configs", type=str, default="default")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--save-predictions", type=Path, default=None)
    parser.add_argument("--loss", choices=("mse", "bce"), default="mse")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    with np.load(args.battles) as z:
        a, b, y = z["a"].astype(np.int64), z["b"].astype(np.int64), z["y"].astype(np.float32)
    permutation = np.random.default_rng(3780).permutation(len(y))
    cut = int(.8 * len(y))
    train_idx, valid_idx = permutation[:cut], permutation[cut:]
    select_idx, test_idx = valid_idx[:len(valid_idx) // 2], valid_idx[len(valid_idx) // 2:]
    tensor = lambda idx: tuple(torch.as_tensor(v[idx], device=device) for v in (a, b, y))
    data = {"train": tensor(train_idx), "select": tensor(select_idx),
            "test": tensor(test_idx), "valid": tensor(valid_idx)}

    grids = {
        "default": [
            {"kind": "pairscore", "rank": 64, "lr": .02, "wd": .01, "batch": 8192},
            {"kind": "mlp", "dim": 32, "hidden": 512, "layers": 2, "dropout": .1, "lr": 2e-3, "wd": 1e-2, "batch": 2048},
            {"kind": "mlp", "dim": 64, "hidden": 512, "layers": 2, "dropout": .1, "lr": 2e-3, "wd": 1e-2, "batch": 2048},
            {"kind": "mlp", "dim": 64, "hidden": 1024, "layers": 3, "dropout": .1, "lr": 1e-3, "wd": 1e-2, "batch": 2048},
            {"kind": "mlp", "dim": 128, "hidden": 1024, "layers": 3, "dropout": .2, "lr": 1e-3, "wd": 5e-2, "batch": 2048},
            {"kind": "mlp", "dim": 16, "hidden": 512, "layers": 3, "dropout": .1, "lr": 2e-3, "wd": 1e-2, "batch": 2048},
        ],
    }
    configs = grids[args.configs] if args.configs in grids else json.loads(args.configs)
    results = []
    ensemble = {}
    for config in configs:
        for seed in args.seeds:
            torch.manual_seed(seed)
            start = time.monotonic()
            if config["kind"] == "pairscore":
                model = PairScore(config["rank"]).to(device)
            else:
                model = PairMLP(config["dim"], config["hidden"], config["layers"],
                                config["dropout"]).to(device)
            best, best_epoch, epochs_run = train(model, data, args, config["lr"], config["wd"],
                                                 args.epochs, config["batch"])
            test_pred = predict(model, *data["test"][:2])
            valid_pred = predict(model, *data["valid"][:2])
            key = json.dumps(config, sort_keys=True)
            ensemble.setdefault(key, []).append(test_pred)
            record = {**config, "loss": args.loss, "seed": seed, "select_rmse": best,
                      "test_rmse": rmse(test_pred, data["test"][2]),
                      "full_valid_rmse": rmse(valid_pred, data["valid"][2]),
                      "best_epoch": best_epoch, "epochs_run": epochs_run,
                      "seconds": round(time.monotonic() - start, 1)}
            record["test_score_if_ids_perfect"] = 100 * (1 - record["test_rmse"] / 4.382900214041612)
            results.append(record)
            print(json.dumps(record), flush=True)
    summary = {key: rmse(torch.stack(preds).mean(0), data["test"][2])
               for key, preds in ensemble.items() if len(preds) > 1}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"runs": results, "seed_ensemble_test_rmse": summary,
                                       "split": "evaluate_score_models 80/20, seed 3780; select=first half of validation"},
                                      indent=2) + "\n")
    if args.save_predictions is not None:
        np.savez_compressed(args.save_predictions, test_idx=test_idx,
                            **{f"pred_{i}": torch.stack(p).mean(0).cpu().numpy()
                               for i, p in enumerate(ensemble.values())})


if __name__ == "__main__":
    main()
