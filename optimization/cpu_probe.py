"""CPU equivalence probe with actual48-block Pairformer and stochastic dropout."""
import os,sys,json,copy
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['LAYERNORM_TYPE']='torch'
root=Path('/root/tianbali/protenix_8h20_20261103_round31');sys.path.insert(0,str(root/'reference'))
import torch
import torch.utils.checkpoint
import protenix.model.modules.pairformer as module
from partial_pair_checkpoint import install
torch.set_num_threads(2);torch.manual_seed(20261031)
model=module.PairformerStack(n_blocks=48,n_heads=4,c_z=16,c_s=32,dropout=.25,blocks_per_ckpt=1).train()
# Default output projections start at zero; randomize to test nontrivial downstream gradients.
with torch.no_grad():
 for name,p in model.named_parameters():
  if p.ndim>=2 and torch.count_nonzero(p)==0:p.normal_(0,.015)
s=torch.randn(4,32);z=torch.randn(4,4,16);state=copy.deepcopy(model.state_dict())
original=module.checkpoint_blocks;results=[];reference=None
for prefix in [0,8,10,12]:
 module.checkpoint_blocks=original;install(prefix)
 model.load_state_dict(state);model.zero_grad(set_to_none=True)
 a=s.clone().requires_grad_();b=z.clone().requires_grad_();torch.manual_seed(12345)
 outputs=model(a,b,pair_mask=None,triangle_attention='torch',triangle_multiplicative='torch')
 sum(o.square().mean() for o in outputs).backward()
 values=[o.detach() for o in outputs]+[a.grad,b.grad]+[p.grad for p in model.parameters() if p.grad is not None]
 assert all(torch.isfinite(v).all() for v in values)
 snapshot=([v.clone() for v in values],torch.get_rng_state().clone())
 if reference is None:reference=snapshot
 exact=all(torch.equal(a,b) for a,b in zip(reference[0],values)) and len(reference[0])==len(values)
 rng_exact=torch.equal(reference[1],snapshot[1])
 results.append({'prefix':prefix,'outputs_and_gradients_exact':exact,'rng_exact':rng_exact,'tensor_count':len(values)})
 assert exact and rng_exact,results[-1]
assert not torch.cuda.is_initialized()
report={'status':'PASS','cases':results,'gpu_initialized':False,'scope':'Actual small-channel48-block CPU Pairformer with dropout; not native-width CUDA performance or full-training acceptance.'}
Path(__file__).with_name('CPU_PROBE.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
