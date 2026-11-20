"""Pilot posterior averaging over image top-20 candidates for score RMSE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_score_models import PairScore


DENOMINATOR = 4.382900214041612


def rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted-actual)**2)))


def score(value: float) -> float:
    return 100*(1-value/DENOMINATOR)


@torch.inference_mode()
def score_table(checkpoint: Path) -> np.ndarray:
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    model = PairScore(64).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    table = np.empty((1026,1026), dtype=np.float32)
    right = torch.arange(1026, device=device).repeat(64)
    for start in range(0, 1026, 64):
        count = min(64, 1026-start)
        left = torch.arange(start, start+count, device=device).repeat_interleave(1026)
        values = model(left, right[:len(left)])
        table[start:start+count] = values.cpu().numpy().reshape(count, 1026)
    return table


def expected_scores(table: np.ndarray, ids1: np.ndarray, ids2: np.ndarray,
                    q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    output = np.empty(len(ids1), dtype=np.float32)
    for start in range(0, len(ids1), 1024):
        end = min(start+1024, len(ids1))
        pair = table[ids1[start:end, :, None], ids2[start:end, None, :]]
        output[start:end] = np.einsum(
            "bi,bij,bj->b", q1[start:end], pair, q2[start:end], optimize=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=Path(
        "outputs/large/score_model_error_rows.npz"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "outputs/large/wrn_top20_oof_compact.npz"))
    parser.add_argument("--image-oof", type=Path, default=Path(
        "outputs/large/meta_rerank_three_arch_top20_oof.npz"))
    parser.add_argument("--score-model", type=Path, default=Path(
        "outputs/large/score_model_rank64_holdout.pt"))
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/large/soft_score_pilot.json"))
    parser.add_argument("--clean-selection", action="store_true",
                        help="Checkpoint used only the first validation half for early stopping")
    parser.add_argument("--save-predictions", type=Path, default=None)
    args = parser.parse_args()
    with np.load(args.rows) as z:
        y = z["result"].astype(np.float32)
        first = z["image1_indices"]
        second = z["image2_indices"]
        hard_predictions = z["score_predicted_ids"]
    with np.load(args.candidates) as z:
        ids = z["candidates"].astype(np.int16)
        raw = z["probabilities"].astype(np.float32)
    with np.load(args.image_oof) as z:
        meta = np.zeros(len(ids), dtype=np.int16)
        meta[z["indices"]] = z["meta_top1"]
    if ids.shape != (len(meta),20) or len(y) != 36960:
        raise ValueError("Image OOF features or score-row validation mismatch")
    if not np.all((ids == meta[:,None]).any(1)):
        raise ValueError("Meta top-1 falls outside WRN top-20 candidates")
    table = score_table(args.score_model)
    first_ids, second_ids = ids[first], ids[second]
    first_meta = first_ids == meta[first,None]
    second_meta = second_ids == meta[second,None]
    cut = len(y)//2
    tune, test = slice(0,cut), slice(cut,None)
    baseline = {"tune_rmse":rmse(hard_predictions[tune],y[tune]),
                "test_rmse":rmse(hard_predictions[test],y[test])}
    candidates = []
    for gamma in (.5, 1., 2., 4., 8.):
        likelihood = np.maximum(raw,1e-10)**gamma
        likelihood /= likelihood.sum(1,keepdims=True)
        for alpha in (0., .25, .5, .75, .9, 1.):
            q = alpha*(ids==meta[:,None])+(1-alpha)*likelihood
            prediction = expected_scores(table,first_ids,second_ids,q[first],q[second])
            for shrink in (.8, .9, 1., 1.1):
                adjusted = np.clip(5+shrink*(prediction-5),0,10)
                candidates.append({
                    "gamma":gamma,"meta_top1_mass":alpha,"shrink":shrink,
                    "tune_rmse":rmse(adjusted[tune],y[tune]),
                    "test_rmse":rmse(adjusted[test],y[test])})
        print(f"Gamma {gamma} complete",flush=True)
    selected = min(candidates,key=lambda value:value["tune_rmse"])
    gamma = selected["gamma"]
    alpha = selected["meta_top1_mass"]
    likelihood = np.maximum(raw,1e-10)**gamma
    likelihood /= likelihood.sum(1,keepdims=True)
    q = alpha*(ids==meta[:,None])+(1-alpha)*likelihood
    expected = expected_scores(table,first_ids,second_ids,q[first],q[second])
    adjusted = np.clip(5+selected["shrink"]*(expected-5),0,10)
    meta_pos = (ids==meta[:,None]).argmax(1)
    meta_conf = raw[np.arange(len(raw)),meta_pos]
    top1_conf = raw[:,0]
    gate_results = []
    for name, image_conf in (("wrn_prob_of_meta_top1",meta_conf),
                             ("wrn_top1_prob",top1_conf)):
        pair_conf = np.minimum(image_conf[first],image_conf[second])
        thresholds = np.unique(np.quantile(pair_conf[tune],np.linspace(.05,.95,19)))
        for threshold in thresholds:
            gated = np.where(pair_conf<threshold,adjusted,hard_predictions)
            gate_results.append({"feature":name,"threshold":float(threshold),
                                 "soft_fraction":float(np.mean(pair_conf<threshold)),
                                 "tune_rmse":rmse(gated[tune],y[tune]),
                                 "test_rmse":rmse(gated[test],y[test])})
    selected_gate = min(gate_results,key=lambda value:value["tune_rmse"])
    if args.save_predictions is not None:
        args.save_predictions.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.save_predictions,
                            row_positions=np.arange(cut,len(y)), actual=y[test],
                            hard=hard_predictions[test], soft=adjusted[test])
    report={"rows":len(y),"tune_rows":cut,"test_rows":len(y)-cut,
            "baseline_hard_meta_top1":baseline,
            "selected_on_tune":selected,
            "baseline_test_score":score(baseline["test_rmse"]),
            "soft_test_score":score(selected["test_rmse"]),
            "test_score_gain":score(selected["test_rmse"])-score(baseline["test_rmse"]),
            "selected_confidence_gate":selected_gate,
            "gated_test_score":score(selected_gate["test_rmse"]),
            "note":("Score-model early stop and posterior hyperparameters used the first validation half only. The rank-64 architecture was chosen in an earlier pilot using this data, so this is still an exploratory estimate."
                    if args.clean_selection else
                    "Exploratory: score-model checkpoint was selected using the full score validation set; posterior hyperparameters were selected on the first half only.")}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
