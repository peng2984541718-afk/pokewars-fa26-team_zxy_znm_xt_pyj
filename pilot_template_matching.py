"""Fold0 development pilot: clean multi-template masked cosine matching."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from image_experiment import SEED,stratified_folds
from degradation import clean_impulses_gpu,laplacian_noise
from evaluate_wrn_tta import transform
try:
    from stage1_joint_inference import run_fold0
except ModuleNotFoundError:
    from joint_constrained_inference import run_fold0

@torch.inference_mode()
def main():
    torch.set_float32_matmul_precision('high')
    out=Path('outputs/template_pilot_v1');out.mkdir(exist_ok=True)
    images=np.load('labeled_32x32.npy',mmap_mode='r');labels=np.load('labeled_ids.npy');folds=stratified_folds(labels,3,SEED)
    tr=np.flatnonzero(folds!=0);noise=np.empty(len(tr),np.float32)
    for st in range(0,len(tr),4096):noise[st:st+4096]=laplacian_noise(images[tr[st:st+4096]])
    prototypes=[]
    for c in range(1025):
        pool=np.flatnonzero(labels[tr]==c);prototypes.append(tr[pool[np.argsort(noise[pool])[:3]]])
    prototypes=np.array(prototypes);assert (folds[prototypes]!=0).all()
    np.save(out/'template_source_indices.npy',prototypes)
    templates=clean_impulses_gpu(torch.as_tensor(np.array(images[prototypes.reshape(-1)]),device='cuda')).float().reshape(1025,3,1024)/255
    z=np.load('outputs/large/phase2_imp_ranker_fold0_scores.npz');d={k:z[k] for k in z.files};vi=d['valid_indices'];cand=d['candidates'].astype(int)-1
    assert np.all(folds[vi]==0)
    score=np.empty(cand.shape,np.float32)
    for st in range(0,len(vi),64):
        en=min(st+64,len(vi));raw=torch.as_tensor(np.array(images[vi[st:en]]),device='cuda')
        x=clean_impulses_gpu(raw).float().unsqueeze(1)/255;mask=(((raw!=25)&(raw!=245)).float().unsqueeze(1))
        # Carry image and confidence mask through exactly the same geometry.
        stack=torch.cat([x,mask],1);ref=templates[torch.as_tensor(cand[st:en],device='cuda')]
        best=torch.full((en-st,40),-1.,device='cuda')
        for base in (stack,stack.flip(-1)):
            for view in (None,'rotate_minus5','rotate_plus5','left','right','up','down'):
                v=base if view is None else transform(base,view)
                q=v[:,0].flatten(1);w=v[:,1].flatten(1).clamp(0,1)
                sw=w.sum(1).clamp_min(1);qm=(q*w).sum(1)/sw
                qc=(q-qm[:,None])*w
                rm=(ref*w[:,None,None,:]).sum(-1)/sw[:,None,None]
                rc=ref-rm[:,:,:,None]
                numer=(rc*qc[:,None,None,:]).sum(-1)
                den=((rc.square()*w[:,None,None,:]).sum(-1)*(qc.square()/w.clamp_min(1e-6)).sum(-1)[:,None,None]).clamp_min(1e-10).sqrt()
                best=torch.maximum(best,(numer/den).amax(2))
        score[st:en]=best.cpu().numpy()
        if st%4096==0:print('matched',en,'/',len(vi),flush=True)
    np.savez_compressed(out/'template_scores.npz',valid_indices=vi,candidates=d['candidates'],scores=score)
    trials=[]
    baseline=json.loads(Path('outputs/large/stage1_aligned_baseline.json').read_text())['selected'];trials.append({'beta':0,**baseline})
    for beta in (2.,4.,8.):
        lp=np.log(np.clip(d['posteriors'],1e-12,1))+beta*score;lp-=lp.max(1,keepdims=True);post=np.exp(lp);post/=post.sum(1,keepdims=True)
        dd=d.copy();dd['posteriors']=post;pack=out/f'beta{beta}.npz';np.savez_compressed(pack,**dd)
        dest=out/f'beta{beta}.json'
        args=argparse.Namespace(train_csv=Path('data/train.csv.gz'),unlabeled_paths=Path('data/unlabeled_paths.txt'),table=Path('outputs/large/pair_semi_nofold0_new_table_round1.npy'),unlabeled_priors=Path('outputs/large/phase2_imp_ranker_unlabeled_f0only.npz'),fold0_scores=pack,scales=[1.05],debias=1.,cavity_iters=12,use_escape=False,output=dest)
        run_fold0(args,'cuda');trials.append({'beta':beta,**json.loads(dest.read_text())['selected']})
    result={'trials':trials,'selected_on_tune':min(trials,key=lambda t:t['tune_rmse']),'limitations':'Historical development fold; exact image-fold exclusion only, no guaranteed same-source sprite isolation. Fixed top40, 3 least-noisy training templates per class, 14 geometric views. No ranker retraining.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
