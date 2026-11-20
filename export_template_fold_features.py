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
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,choices=[0,1,2],required=True);args=parser.parse_args();fold=args.fold
    out=Path(f'outputs/template_learned_v1/f{fold}');out.mkdir(parents=True,exist_ok=True)
    images=np.load('labeled_32x32.npy',mmap_mode='r');labels=np.load('labeled_ids.npy');folds=stratified_folds(labels,3,SEED)
    tr=np.flatnonzero(folds!=fold);noise=np.empty(len(tr),np.float32)
    for st in range(0,len(tr),4096):noise[st:st+4096]=laplacian_noise(images[tr[st:st+4096]])
    prototypes=[]
    for c in range(1025):
        pool=np.flatnonzero(labels[tr]==c);prototypes.append(tr[pool[np.argsort(noise[pool])[:3]]])
    prototypes=np.array(prototypes);assert (folds[prototypes]!=fold).all()
    np.save(out/'template_source_indices.npy',prototypes)
    templates=clean_impulses_gpu(torch.as_tensor(np.array(images[prototypes.reshape(-1)]),device='cuda')).float().reshape(1025,3,1024)/255
    z=np.load('outputs/large/phase2_imp_ranker_oof.npz');take=z['folds']==fold;d={k:z[k][take] for k in ['candidates','posteriors']};vi=z['indices'][take];cand=d['candidates'].astype(int)-1
    assert np.all(folds[vi]==fold)
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
    print('FEATURES COMPLETE',fold,flush=True)
if __name__=='__main__':main()
