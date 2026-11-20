"""Fit conditional template correction on folds1/2; evaluate fold0 development pipeline."""
import sys,json,argparse
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage1_joint_inference import run_fold0
out=Path('outputs/template_learned_v1');z=np.load('outputs/large/phase2_imp_ranker_oof.npz')
def features(p,s):
    p=p/np.maximum(p.sum(1,keepdims=True),1e-12);lp=np.log(p.clip(1e-9));s=s-s.max(1,keepdims=True)
    entropy=-(p*lp).sum(1,keepdims=True)/np.log(p.shape[1]);confidence=p.max(1,keepdims=True)
    return lp.astype(np.float32),np.stack([s,s*entropy,s*confidence,s*lp.clip(-10,0)/10,lp/10,lp*entropy/10],-1).astype(np.float32)
def load(fold):
    t=z['folds']==fold;v=np.load(out/f'f{fold}/template_scores.npz');assert np.array_equal(v['valid_indices'],z['indices'][t]);ids=z['candidates'][t];p=z['posteriors'][t];lab=z['labels'][t]
    hit=(ids==lab[:,None]);keep=hit.any(1);lp,x=features(p,v['scores'])
    return tuple(torch.as_tensor(v,device='cuda') for v in (lp[keep],x[keep],hit[keep].argmax(1)))
a=load(1);b=load(2)
def fit(data,steps,valid=None):
    lp,x,y=data;w=torch.zeros(6,device='cuda',requires_grad=True);opt=torch.optim.Adam([w],lr=.05);best=0;bestloss=float(torch.nn.functional.cross_entropy(valid[0],valid[2])) if valid else None;records=[]
    for i in range(1,steps+1):
        opt.zero_grad();loss=torch.nn.functional.cross_entropy(lp+(x*w).sum(-1),y)+.01*w.square().sum();loss.backward();opt.step()
        if valid:
            with torch.no_grad():v=float(torch.nn.functional.cross_entropy(valid[0]+(valid[1]*w).sum(-1),valid[2]))
            if v<bestloss:bestloss=v;best=i
            if i%10==0:records.append({'step':i,'validation_nll':v})
    return w.detach().cpu().numpy(),best,records
_,steps,records=fit(a,80,b)
combined=tuple(torch.cat([a[i],b[i]]) for i in range(3));w,_,_=fit(combined,steps)
d0=np.load('outputs/large/phase2_imp_ranker_fold0_scores.npz');d={k:d0[k] for k in d0.files};t=np.load('outputs/template_pilot_v1/template_scores.npz');assert np.array_equal(t['valid_indices'],d['valid_indices']);lp,x=features(d['posteriors'],t['scores']);logits=lp+(x*w).sum(-1);logits-=logits.max(1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True);d['posteriors']=p;pack=out/'learned_fold0.npz';np.savez_compressed(pack,**d)
args=argparse.Namespace(train_csv=Path('data/train.csv.gz'),unlabeled_paths=Path('data/unlabeled_paths.txt'),table=Path('outputs/large/pair_semi_nofold0_new_table_round1.npy'),unlabeled_priors=Path('outputs/large/phase2_imp_ranker_unlabeled_f0only.npz'),fold0_scores=pack,scales=[1.05],debias=1.,cavity_iters=12,use_escape=False,output=out/'evaluation.json')
run_fold0(args,'cuda')
report={'selected_steps_on_fold2':steps,'weights':w.tolist(),'training_trace':records,'baseline':json.loads(Path('outputs/large/stage1_aligned_baseline.json').read_text())['selected'],'learned':json.loads((out/'evaluation.json').read_text())['selected'],'limitation':'Historical development CV, inherits non-nested OOF and full-table leakage risks documented in stage1; not independent confirmation.'}
(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
