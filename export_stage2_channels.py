"""Equal six-view TTA for the two pilot arms; preserves full probabilities."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from train_stage2_channels import make_model,expand_stem,pack_inputs,preprocess
from evaluate_wrn_tta import transform
from image_experiment import SEED,stratified_folds
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['control','three_channel','hflip','noise_weighted'],required=True);args=p.parse_args()
out=Path('outputs/stage2_channels_v1');three=args.mode=='three_channel'
labels=np.load('labeled_ids.npy');indices=np.flatnonzero(stratified_folds(labels,3,SEED)==0)
images=pack_inputs(torch.as_tensor(np.load('labeled_32x32.npy')[indices],device='cuda'),three)
model=make_model('wrn28_10',False).cuda()
if three:expand_stem(model)
model.load_state_dict(torch.load(out/f'{args.mode}_best.pt',map_location='cuda',weights_only=True));model.eval()
probs=np.empty((len(indices),1025),np.float32)
with torch.inference_mode():
    for start in range(0,len(indices),256):
        x=preprocess(images[start:start+256]);total=0
        for base in (x,x.flip(-1)):
            for view in (None,'rotate_minus5','rotate_plus5'):
                v=base if view is None else transform(base,view)
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(v)
                total=total+logits.float().softmax(1)
        probs[start:start+len(x)]=(total/6).cpu().numpy()
np.save(out/f'{args.mode}_fold0_probs.npy',probs.astype(np.float16))
order=np.argsort(-probs,axis=1);y=labels[indices]
report={'mode':args.mode,'top1':float((order[:,0]==y).mean()),'top10':float((order[:,:10]==y[:,None]).any(1).mean()),'count':len(y),'protocol':'historical fold0 development comparison, not independent holdout'}
(out/f'{args.mode}_tta.json').write_text(json.dumps(report,indent=2));print(report)
