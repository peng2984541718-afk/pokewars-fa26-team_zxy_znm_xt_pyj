"""One-fold score pilot with fold-safe image ranker and top-20 score averaging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

from evaluate_score_models import PairScore, predict
from evaluate_soft_score import DENOMINATOR, expected_scores, rmse
from image_experiment import SEED, stratified_folds


def model_table(model: PairScore, device: str) -> np.ndarray:
    table = np.empty((1026,1026), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,1026,64):
            end = min(start+64,1026)
            a = torch.arange(start,end,device=device).repeat_interleave(1026)
            b = torch.arange(1026,device=device).repeat(end-start)
            table[start:end] = model(a,b).cpu().numpy().reshape(end-start,1026)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-zip", type=Path, default=Path("data/competition.zip"))
    parser.add_argument("--ranker-scores", type=Path, default=Path(
        "outputs/large/meta_top20_fold0_scores.npz"))
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/large/score_fold0_rerank_soft.json"))
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    with ZipFile(args.data_zip) as archive:
        rows = pd.read_csv(archive.open("data/data/train.csv"))
    identity = {}
    for side in (1,2):
        for path,value in zip(rows[f"noisy_image{side}"],rows[f"pokemon_id{side}"]):
            if pd.notna(value):
                identity[path]=int(value)
    paths = sorted(identity)
    true = np.array([identity[path] for path in paths],dtype=np.int16)
    folds = stratified_folds(true-1,3,SEED)
    path_index = pd.Index(paths)
    first = path_index.get_indexer(rows.noisy_image1)
    second = path_index.get_indexer(rows.noisy_image2)
    known = (first>=0)&(second>=0)
    first,second=first[known],second[known]
    a,b=true[first],true[second]
    y=rows.result.to_numpy(dtype=np.float32)[known]
    pair_fold0 = (folds[first]==0)&(folds[second]==0)
    rng=np.random.default_rng(3780)
    train_rows=np.flatnonzero(~pair_fold0)
    rng.shuffle(train_rows)
    cut=int(.9*len(train_rows))
    fit,early=train_rows[:cut],train_rows[cut:]
    pair_rows=np.flatnonzero(pair_fold0)
    rng.shuffle(pair_rows)
    split=len(pair_rows)//2
    tune,test=pair_rows[:split],pair_rows[split:]
    device="mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(3780)
    model=PairScore(64).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.02,weight_decay=.01)
    fit_a=torch.as_tensor(a[fit],dtype=torch.long,device=device)
    fit_b=torch.as_tensor(b[fit],dtype=torch.long,device=device)
    fit_y=torch.as_tensor(y[fit],device=device)
    best=(float("inf"),0)
    state=None
    for epoch in range(1,args.epochs+1):
        model.train()
        order=torch.randperm(len(fit),device=device)
        for batch in order.split(8192):
            optimizer.zero_grad(set_to_none=True)
            loss=((model(fit_a[batch],fit_b[batch])-fit_y[batch])**2).mean()
            loss.backward()
            optimizer.step()
        early_prediction=predict(model,a[early],b[early],device)
        error=rmse(early_prediction,y[early])
        if error<best[0]-1e-4:
            best=(error,epoch)
            state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        if epoch-best[1]>=6:
            break
    model.load_state_dict(state)
    model.eval()
    table=model_table(model,device)
    print(f"Pair model best epoch {best[1]}, early-stop RMSE {best[0]:.4f}",flush=True)
    with np.load(args.ranker_scores) as z:
        valid_idx=z["valid_indices"]
        candidates=z["candidates"].astype(np.int16)
        wrn=z["wrn_probs"].astype(np.float32)
        ranker=z["ranker_scores"].astype(np.float32)
        meta=z["meta_top1"].astype(np.int16)
        if not np.array_equal(true[valid_idx],z["labels"]):
            raise ValueError("Fold-0 ranker scores do not match labeled image indices")
    position=np.full(len(paths),-1,dtype=np.int32)
    position[valid_idx]=np.arange(len(valid_idx))
    if np.any(position[first[pair_rows]]<0) or np.any(position[second[pair_rows]]<0):
        raise ValueError("A fold-0 pair contains an image outside fold 0")
    one=position[first[pair_rows]]
    two=position[second[pair_rows]]
    ids1,ids2=candidates[one],candidates[two]
    actual=y[pair_rows]
    hard=table[meta[one],meta[two]]
    oracle=table[a[pair_rows],b[pair_rows]]
    baseline={"tune_rmse":rmse(hard[:split],actual[:split]),
              "test_rmse":rmse(hard[split:],actual[split:])}
    wrn=np.maximum(wrn,1e-12)
    wrn/=wrn.sum(1,keepdims=True)
    ranker=np.maximum(ranker,1e-12)
    hard_onehot=(candidates==meta[:,None]).astype(np.float32)
    trials=[]
    for gamma in (.5,1.,2.,4.,8.):
        r=ranker**gamma
        r/=r.sum(1,keepdims=True)
        for ranker_fraction in (0.,.5,1.):
            soft=ranker_fraction*r+(1-ranker_fraction)*wrn
            for hard_mass in (0.,.25,.5):
                posterior=hard_mass*hard_onehot+(1-hard_mass)*soft
                prediction=expected_scores(table,ids1,ids2,
                                           posterior[one],posterior[two])
                trials.append({"gamma":gamma,"ranker_fraction":ranker_fraction,
                               "hard_mass":hard_mass,
                               "tune_rmse":rmse(prediction[:split],actual[:split]),
                               "test_rmse":rmse(prediction[split:],actual[split:])})
        print(f"Gamma {gamma} complete",flush=True)
    selected=min(trials,key=lambda item:item["tune_rmse"])
    r=ranker**selected["gamma"]
    r/=r.sum(1,keepdims=True)
    soft=selected["ranker_fraction"]*r+(1-selected["ranker_fraction"])*wrn
    posterior=selected["hard_mass"]*hard_onehot+(1-selected["hard_mass"])*soft
    selected_prediction=expected_scores(table,ids1,ids2,
                                        posterior[one],posterior[two])
    wrong=(meta[one[split:]]!=a[pair_rows[split:]])|(
        meta[two[split:]]!=b[pair_rows[split:]])
    actual_test=actual[split:]
    hard_test=hard[split:]
    soft_test=selected_prediction[split:]
    rng_boot=np.random.default_rng(3780)
    paired_gains=np.empty(2000,dtype=np.float64)
    hard_squared=(hard_test-actual_test)**2
    soft_squared=(soft_test-actual_test)**2
    for sample_id in range(len(paired_gains)):
        sample=rng_boot.integers(len(test),size=len(test))
        paired_gains[sample_id]=100*(np.sqrt(hard_squared[sample].mean())-
                                      np.sqrt(soft_squared[sample].mean()))/DENOMINATOR
    group_results={}
    for name,mask in (("both_images_correct",~wrong),("at_least_one_image_wrong",wrong)):
        group_results[name]={"count":int(mask.sum()),
                             "hard_rmse":rmse(hard_test[mask],actual_test[mask]),
                             "soft_rmse":rmse(soft_test[mask],actual_test[mask]),
                             "hard_absolute_error_at_least_8":float(np.mean(
                                 np.abs(hard_test[mask]-actual_test[mask])>=8)),
                             "soft_absolute_error_at_least_8":float(np.mean(
                                 np.abs(soft_test[mask]-actual_test[mask])>=8))}
    report={
        "image_fold":0,"known_battles":len(y),
        "score_model_fit_rows":len(fit),"score_model_early_stop_rows":len(early),
        "fold0_pair_battles":len(pair_rows),"tune_rows":split,"test_rows":len(test),
        "ranker_fold0_top1":float(np.mean(meta==true[valid_idx])),
        "pair_model_best_epoch":best[1],"pair_model_early_stop_rmse":best[0],
        "pair_model_true_ids_test_rmse":rmse(oracle[split:],actual[split:]),
        "hard_meta":baseline,
        "selected_soft":selected,
        "hard_test_score":100*(1-baseline["test_rmse"]/DENOMINATOR),
        "soft_test_score":100*(1-selected["test_rmse"]/DENOMINATOR),
        "test_gain_points":100*(baseline["test_rmse"]-selected["test_rmse"])/DENOMINATOR,
        "paired_bootstrap_95_gain_points":np.quantile(paired_gains,[.025,.975]).tolist(),
        "test_error_groups":group_results,
        "test_wrong_image_battle_fraction":float(wrong.mean()),
        "method":"Image ranker fit on folds 1+2; score model excludes all fold-0/fold-0 battles; soft parameters selected on first half of fold-0/fold-0 battles; final metrics on second half.",
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    np.savez_compressed(args.output.with_suffix(".npz"),
                        train_row_positions=pair_rows[split:],
                        actual=actual_test,hard=hard_test,soft=soft_test,
                        image_identity_wrong=wrong)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
