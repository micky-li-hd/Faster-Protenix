"""Actual upstream fallback and scoped compiler-option construction, no CUDA."""
import os,sys,json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['LAYERNORM_TYPE']='torch'
sys.path.insert(0,'/root/tianbali/protenix_8h20_20261031/reference')
import torch
import torch._inductor.config as config
import protenix.model.modules.primitives as module
from guarded_attention import install
torch.set_num_threads(2);torch.manual_seed(20261027)
original=module._attention
before={k:getattr(config,k) for k in ['pattern_matcher','max_autotune']};before['cudagraphs']=config.triton.cudagraphs
stats=install(module,torch)
after={k:getattr(config,k) for k in ['pattern_matcher','max_autotune']};after['cudagraphs']=config.triton.cudagraphs
assert before==after,'Per-compiler options leaked globally'
rows=[]
for efficient in [True,False]:
 for bias_present in [True,False]:
  raw=[torch.randn(2,4,8,16) for _ in range(3)]+[torch.randn(1,4,8,8)]
  inputs=[x.requires_grad_() for x in raw];bias=inputs[-1] if bias_present else None
  rng=torch.get_rng_state().clone();a=original(*inputs[:3],bias,efficient,False);a.square().mean().backward()
  ref=[a.detach().clone()]+[x.grad.clone() for x in inputs if x.grad is not None]
  for x in inputs:x.grad=None
  b=module._attention(*inputs[:3],bias,efficient,False);b.square().mean().backward()
  values=[b.detach()]+[x.grad for x in inputs if x.grad is not None]
  assert len(ref)==len(values) and all(torch.equal(x,y) for x,y in zip(ref,values))
  assert torch.equal(rng,torch.get_rng_state())
  rows.append({'efficient':efficient,'bias_present':bias_present,'outputs_all_input_gradients_rng_exact':True})
assert stats['calls']==stats['bypassed_calls']==4 and stats['eligible_calls']==0
assert not torch.cuda.is_initialized()
report={'status':'PASS_CPU_FALLBACK_AND_OPTION_SCOPE','cases':rows,'stats':stats,'global_inductor_options_unchanged':before==after,'cuda_initialized':False,'helper_sha256':hashlib.sha256((root/'guarded_attention.py').read_bytes()).hexdigest(),'accepted':False,'scope':'CUDA compiled wrapper and checkpoint interaction still untested; do not promote on CPU fallback proof.'}
(root/'CPU_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
