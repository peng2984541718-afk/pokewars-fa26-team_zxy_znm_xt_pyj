import argparse,json
from pathlib import Path
import count_prior_inference as model
out=Path('outputs/count_prior_pilot');out.mkdir(exist_ok=True)
trials=[]
for strength in [0.,.05,.15,.3]:
    model.COUNT_STRENGTH=strength;model.COUNT_DIAGNOSTICS.clear()
    args=argparse.Namespace(train_csv=Path('data/train.csv.gz'),unlabeled_paths=Path('data/unlabeled_paths.txt'),table=Path('outputs/large/pair_semi_nofold0_new_table_round1.npy'),unlabeled_priors=Path('outputs/large/phase2_imp_ranker_unlabeled_f0only.npz'),fold0_scores=Path('outputs/large/phase2_imp_ranker_fold0_scores.npz'),scales=[1.05],debias=1.,cavity_iters=12,use_escape=False,output=out/f'strength{strength}.json')
    model.run_fold0(args,'cuda')
    trials.append({'strength':strength,**json.loads(args.output.read_text())['selected'],'counts':model.COUNT_DIAGNOSTICS[-1]})
report={'trials':trials,'selected_on_tune':min(trials,key=lambda t:t['tune_rmse']),'method':'Weak iterative class-count log-bias, clipped +/-0.5; target 2*battles/1025, applied jointly to both ordered sides. Existing debias retained. Historical development fold.'}
(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
