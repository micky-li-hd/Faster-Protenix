import os,sys,json,copy
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['LAYERNORM_TYPE']='torch'
sys.path.insert(0,str(Path(__file__).resolve().parent/'reference'))
import torch
import torch.utils.checkpoint
import protenix.model.modules.transformer as module
from partial_diffusion_checkpoint import install
torch.set_num_threads(2);torch.manual_seed(20261027)
model=module.DiffusionTransformer(c_a=32,c_s=16,c_z=8,n_blocks=24,n_heads=4,drop_path_rate=.1,blocks_per_ckpt=1).train()
with torch.no_grad():
 for p in model.parameters():
  if p.ndim>=2 and torch.count_nonzero(p)==0:p.normal_(0,.015)
values=[torch.randn(2,4,32),torch.randn(2,4,16),torch.randn(1,4,4,8)];state=copy.deepcopy(model.state_dict());reference=None;rows=[];original=module.checkpoint_blocks
for prefix in [0,8,16,24]:
 module.checkpoint_blocks=original;stats=install(prefix);model.load_state_dict(state);model.zero_grad(set_to_none=True)
 inputs=[v.clone().requires_grad_() for v in values];torch.manual_seed(12345)
 y=model(*inputs);y.square().mean().backward()
 tensors=[y.detach()]+[v.grad for v in inputs]+[p.grad for p in model.parameters() if p.grad is not None];after=torch.get_rng_state().clone()
 assert all(torch.isfinite(x).all() for x in tensors)
 if reference is None:reference=([x.clone() for x in tensors],after)
 exact=len(tensors)==len(reference[0]) and all(torch.equal(a,b) for a,b in zip(tensors,reference[0]));rng=torch.equal(after,reference[1])
 row={'prefix':prefix,'outputs_all_input_parameter_gradients_exact':exact,'rng_exact':rng,'tensor_count':len(tensors),'stats':stats};rows.append(row);assert exact and rng,row
 if prefix:assert stats['matched_gradient_calls']==1 and stats['retained_block_calls']==prefix
module.checkpoint_blocks=original
assert not torch.cuda.is_initialized()
r={'status':'PASS_CPU_24BLOCK_DIFFUSION_PREFIX','cases':rows,'cuda_initialized':False,'scope':'Actual24-block small-channel DiffusionTransformer with nonzero output weights and stochastic drop path. Not native-width GPU or training validation.'}
Path(__file__).with_name('DIFFUSION_CPU_CHECK.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
