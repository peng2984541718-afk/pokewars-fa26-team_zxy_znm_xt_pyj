"""Compare score models on held-out battles with known true species IDs.

This pilot uses only rows whose two image identities are known from labeled
training images. It never trains on held-out battle scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from torch import nn


class PairScore(nn.Module):
    def __init__(self, rank: int):
        super().__init__()
        self.strength = nn.Embedding(1026, 1)
        nn.init.zeros_(self.strength.weight)
        self.rank = rank
        if rank:
            self.left = nn.Embedding(1026, rank)
            self.right = nn.Embedding(1026, rank)
            nn.init.normal_(self.left.weight, std=.03)
            nn.init.normal_(self.right.weight, std=.03)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        logit = self.strength(a).squeeze(-1)-self.strength(b).squeeze(-1)
        if self.rank:
            logit = logit + ((self.left(a)*self.right(b) -
                              self.left(b)*self.right(a)).sum(-1) /
                             self.rank**.5)
        return 5 + 5*torch.tanh(logit)


def predict(model: PairScore, a: np.ndarray, b: np.ndarray,
            device: str) -> np.ndarray:
    model.eval()
    result = np.empty(len(a), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(a), 8192):
            end = min(start+8192, len(a))
            left = torch.as_tensor(a[start:end], dtype=torch.long, device=device)
            right = torch.as_tensor(b[start:end], dtype=torch.long, device=device)
            result[start:end] = model(left, right).cpu().numpy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-zip", type=Path, default=Path("data/competition.zip"))
    parser.add_argument("--image-oof", type=Path, default=Path(
        "outputs/large/meta_rerank_three_arch_top20_oof.npz"))
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/large/score_model_pilot.json"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--ranks", type=int, nargs="+", default=(0, 8, 16, 32))
    parser.add_argument("--save-predictions", type=Path, default=None,
                        help="Save row-level validation predictions for the last rank")
    parser.add_argument("--save-model", type=Path, default=None,
                        help="Save the best validation model from the last rank")
    parser.add_argument("--selection-half", action="store_true",
                        help="Use only the first half of held-out score rows for early stopping")
    args = parser.parse_args()
    with ZipFile(args.data_zip) as archive:
        rows = pd.read_csv(archive.open("data/data/train.csv"))
    identity = {}
    for side in (1, 2):
        paths = rows[f"noisy_image{side}"]
        ids = rows[f"pokemon_id{side}"]
        for path, value in zip(paths[ids.notna()], ids[ids.notna()].astype(int)):
            identity[path] = value
    paths = sorted(identity)
    with np.load(args.image_oof) as z:
        predictions = np.zeros(len(paths), dtype=np.int16)
        predictions[z["indices"]] = z["meta_top1"]
        actual_ids = np.array([identity[path] for path in paths])
        if not np.array_equal(actual_ids[z["indices"]], z["labels"]):
            raise ValueError("OOF image indices do not align with labeled paths")
    path_index = pd.Index(paths)
    first = path_index.get_indexer(rows.noisy_image1)
    second = path_index.get_indexer(rows.noisy_image2)
    known = (first >= 0) & (second >= 0)
    first, second = first[known], second[known]
    a, b = actual_ids[first], actual_ids[second]
    predicted_a, predicted_b = predictions[first], predictions[second]
    y = rows.result.to_numpy(dtype=np.float32)[known]
    rng = np.random.default_rng(3780)
    permutation = rng.permutation(len(y))
    cut = int(.8*len(y))
    train_indices, valid_indices = permutation[:cut], permutation[cut:]
    selection_indices = (valid_indices[:len(valid_indices)//2]
                         if args.selection_half else valid_indices)
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    report = {"rows":len(y), "validation_rows":len(valid_indices),
              "image_top1_on_validation":float(np.mean(np.r_[
                  predicted_a[valid_indices] == a[valid_indices],
                  predicted_b[valid_indices] == b[valid_indices]])),
              "models":{}}
    for rank in args.ranks:
        torch.manual_seed(3780)
        model = PairScore(rank).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=.01)
        ta = torch.as_tensor(a[train_indices], dtype=torch.long, device=device)
        tb = torch.as_tensor(b[train_indices], dtype=torch.long, device=device)
        ty = torch.as_tensor(y[train_indices], device=device)
        best = (float("inf"), -1)
        best_state = None
        history = []
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(len(ta), device=device)
            for part in order.split(8192):
                optimizer.zero_grad(set_to_none=True)
                loss = ((model(ta[part], tb[part])-ty[part])**2).mean()
                loss.backward()
                optimizer.step()
            valid_pred = predict(model, a[selection_indices], b[selection_indices], device)
            rmse = float(np.sqrt(np.mean((valid_pred-y[selection_indices])**2)))
            history.append(rmse)
            if rmse < best[0]-1e-4:
                best = (rmse, epoch+1)
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch+1-best[1] >= 6:
                break
        model.load_state_dict(best_state)
        if args.save_model is not None and rank == args.ranks[-1]:
            args.save_model.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, args.save_model)
        true_pred = predict(model, a[valid_indices], b[valid_indices], device)
        true_rmse = float(np.sqrt(np.mean((true_pred-y[valid_indices])**2)))
        pa = predict(model, predicted_a[valid_indices], predicted_b[valid_indices], device)
        predicted_rmse = float(np.sqrt(np.mean((pa-y[valid_indices])**2)))
        report["models"][str(rank)] = {
            "true_id_rmse":true_rmse, "model_selection_rmse":best[0],
            "oof_image_id_rmse":predicted_rmse,
            "oof_image_id_score":100*(1-predicted_rmse/4.382900214041612),
            "best_epoch":best[1], "trained_epochs":len(history)}
        print(rank, report["models"][str(rank)], flush=True)
        if args.save_predictions is not None and rank == args.ranks[-1]:
            args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.save_predictions,
                train_row_indices=np.flatnonzero(known)[valid_indices],
                image1_indices=first[valid_indices], image2_indices=second[valid_indices],
                true_id1=a[valid_indices].astype(np.uint16),
                true_id2=b[valid_indices].astype(np.uint16),
                predicted_id1=predicted_a[valid_indices].astype(np.uint16),
                predicted_id2=predicted_b[valid_indices].astype(np.uint16),
                result=y[valid_indices],
                score_true_ids=true_pred,
                score_predicted_ids=pa)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
