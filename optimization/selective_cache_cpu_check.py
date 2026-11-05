"""Actual 48-block Pairformer/dropout equivalence for bounded selective caching."""
import os,sys,json,copy,gc
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['LAYERNORM_TYPE']='torch'
sys.path.insert(0,'/root/tianbali/protenix_8h20_20261103_round31/reference')
import torch
import protenix.model.modules.pairformer as module
from selective_pair_cache import install
torch.set_num_threads(2);torch.manual_seed(20261031)
model=module.PairformerStack(n_blocks=48,n_heads=4,c_z=16,c_s=32,dropout=.25,blocks_per_ckpt=1).train()
with torch.no_grad():
 for p in model.parameters():
  if p.ndim>=2 and torch.count_nonzero(p)==0:p.normal_(0,.015)
s=torch.randn(4,32);z=torch.randn(4,4,16);state=copy.deepcopy(model.state_dict());original=module.checkpoint_blocks;cases=[]
for autocast in [False,True]:
 reference=None
 for budget in [0,1024,65536,1048576]:
  module.checkpoint_blocks=original;stats=install(budget)
  model.load_state_dict(state);model.zero_grad(set_to_none=True);a=s.clone().requires_grad_();b=z.clone().requires_grad_();torch.manual_seed(12345)
  with torch.autocast('cpu',dtype=torch.bfloat16,enabled=autocast):outputs=model(a,b,pair_mask=None,triangle_attention='torch',triangle_multiplicative='torch');loss=sum(o.square().mean() for o in outputs)
  loss.backward();values=[o.detach() for o in outputs]+[a.grad,b.grad]+[p.grad for p in model.parameters() if p.grad is not None]
  assert all(torch.isfinite(x).all() for x in values);rng=torch.get_rng_state().clone()
  if reference is None:reference=([x.clone() for x in values],rng)
  exact=len(values)==len(reference[0]) and all(torch.equal(x,y) for x,y in zip(values,reference[0]));rng_exact=torch.equal(rng,reference[1])
  case={'cpu_autocast_bf16':autocast,'budget_bytes_per_block':budget,'outputs_input_and_parameter_gradients_exact':exact,'rng_exact':rng_exact,'stats':stats};cases.append(case)
  Path(__file__).with_name('SELECTIVE_CACHE_CPU_PROGRESS.json').write_text(json.dumps(cases,indent=2)+'\n')
  assert exact and rng_exact,case
  if budget:assert stats['shape_eligible_calls']>0 and stats['shape_rejected_calls']>0 and stats['cache_snapshot_copies']==stats['saved_op_calls'] and stats['stack_invocations']==1 and stats['contexts']==48 and stats['saved_op_calls']>0 and stats['cached_recompute_hits']>0 and stats['max_context_saved_bytes']<=budget
  del outputs,loss,values,a,b;gc.collect()
assert not torch.cuda.is_initialized()
result={'status':'PASS_CPU_PRIORITY_CACHE','cases':cases,'gpu_initialized':False,'accepted':False,'scope':'Native small-channel48-block CPU probe only. CUDA memory, speed, and frozen8/128 acceptance remain required.'}
Path(__file__).with_name('SELECTIVE_CACHE_CPU_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'cases':len(cases)}))
