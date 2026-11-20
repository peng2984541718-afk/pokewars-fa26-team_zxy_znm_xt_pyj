"""Development-only noise_weighted error overlap and conservative posterior fusion."""
import json,sys,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from image_experiment import SEED,stratified_folds
try:
    from stage1_joint_inference import run_fold0
except ModuleNotFoundError:
    from joint_constrained_inference import run_fold0
root=Path('outputs/stage2_channels_v1');out=root/'noise_integration';out.mkdir(exist_ok=True)
y=np.load('labeled_ids.npy');indices=np.flatnonzero(stratified_folds(y,3,SEED)==0);truth=y[indices]
a=np.load(root/'control_fold0_probs.npy').astype(np.float32);b=np.load(root/'noise_weighted_fold0_probs.npy').astype(np.float32)
ca=a.argmax(1)==truth;cb=b.argmax(1)==truth
z=np.load('outputs/large/stage1_aligned_baseline.npz');split=int(z['split']);sq=(z['prediction']-z['actual'])**2
worst=np.arange(split,len(sq));worst=worst[np.argsort(-sq[worst])[:100]]
lookup={int(v):i for i,v in enumerate(indices)};hard=np.unique(np.concatenate([z['image_indices1'][worst],z['image_indices2'][worst]]));hi=np.array([lookup[int(v)] for v in hard])
report={'count':len(truth),'fixed_control_errors':int((~ca&cb).sum()),'new_errors':int((ca&~cb).sum()),'hard100_unique_images':len(hi),'hard100_control_correct':int(ca[hi].sum()),'hard100_noise_weighted_correct':int(cb[hi].sum()),'hard100_fixed':int((~ca[hi]&cb[hi]).sum()),'hard100_new_errors':int((ca[hi]&~cb[hi]).sum())}
old=np.load('outputs/large/phase2_imp_ranker_fold0_scores.npz');d={k:old[k] for k in old.files};idx=np.array([lookup[int(v)] for v in d['valid_indices']]);cand=d['candidates'].astype(int)-1
q=b[idx[:,None],cand];q=q/np.maximum(q.sum(1,keepdims=True),1e-12)
trials=[]
for weight in [.1,.25]:
    dd=d.copy();dd['posteriors']=(1-weight)*d['posteriors']+weight*q
    pack=out/f'noise_weighted_w{weight}.npz';np.savez_compressed(pack,**dd)
    dest=out/f'noise_weighted_w{weight}.json'
    args=argparse.Namespace(train_csv=Path('data/train.csv.gz'),unlabeled_paths=Path('data/unlabeled_paths.txt'),table=Path('outputs/large/pair_semi_nofold0_new_table_round1.npy'),unlabeled_priors=Path('outputs/large/phase2_imp_ranker_unlabeled_f0only.npz'),fold0_scores=pack,scales=[1.05],debias=1.,cavity_iters=12,use_escape=False,output=dest)
    run_fold0(args,'cuda');trials.append({'weight':weight,**json.loads(dest.read_text())['selected']})
base=json.loads(Path('outputs/large/stage1_aligned_baseline.json').read_text())['selected'];trials.insert(0,{'weight':0.,**base})
report['trials']=trials;report['selected_on_tune']=min(trials,key=lambda t:t['tune_rmse']);report['limitation']='Historical development fold; fixed candidate pool and unlabeled priors; not retraining the feature ranker or an unbiased leaderboard estimate.'
(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
