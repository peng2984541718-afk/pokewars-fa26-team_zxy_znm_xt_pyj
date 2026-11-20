import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from train_stage2_channels import WideResNet,pack_inputs,preprocess,expand_stem
raw=torch.randint(0,256,(4,32,32),dtype=torch.uint8);raw[:,5,5]=25;raw[:,6,6]=245
m=WideResNet(width=1).eval()
with torch.no_grad():
    expected=m(preprocess(pack_inputs(raw,False)))
    expand_stem(m)
    packed=pack_inputs(raw,True)
    actual=m(preprocess(packed))
torch.testing.assert_close(expected,actual,atol=1e-6,rtol=1e-5)
assert torch.equal(packed[:,0],raw)
assert torch.equal(packed[:,2]>0,(raw==25)|(raw==245))
print('PASS: initialization preserves baseline logits; raw pixels and mask intact')
