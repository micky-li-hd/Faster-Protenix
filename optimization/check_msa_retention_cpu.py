"""Actual small-width four-block MSA +48-block Pairformer, CPU only."""
import os,sys,json,copy,hashlib,functools
from pathlib import Path
r=Path(__file__).resolve().parent
assert os.environ['CUDA_VISIBLE_DEVICES']==''
os.environ['LAYERNORM_TYPE']='torch';sys.path.insert(0,str(r/'reference'))
import torch
import torch.utils.checkpoint
import protenix.model.modules.pairformer as module
from partial_msa_checkpoint import install as install_msa
from partial_pair_checkpoint import install as install_pair
assert torch.__version__.split('+')[0]=='2.7.1';torch.set_num_threads(2);torch.manual_seed(20261102)
msa=torch.nn.ModuleList([module.MSABlock(c_m=32,c_z=16,c_hidden=4,is_last_block=i==3,msa_dropout=.15,pair_dropout=.25,msa_chunk_size=4) for i in range(4)]).train()
pair=module.PairformerStack(n_blocks=48,n_heads=4,c_z=16,c_s=32,dropout=.25,blocks_per_ckpt=1).train()
model=torch.nn.ModuleDict({'msa':msa,'pair':pair})
with torch.no_grad():
 for p in model.parameters():
  if p.ndim>=2 and torch.count_nonzero(p)==0:p.normal_(0,.015)
state=copy.deepcopy(model.state_dict());values=[torch.randn(8,4,32),torch.randn(4,4,16),torch.randn(4,32)]
blocks=[functools.partial(b,pair_mask=None,triangle_attention='torch',triangle_multiplicative='torch',inplace_safe=False) for b in msa]
original=module.checkpoint_blocks;reference=None;rows=[]
try:
 for pp in [0,4]:
  for mp in [0,2,4]:
   for order in ['msa_then_pair','pair_then_msa']:
    module.checkpoint_blocks=original
    if order=='msa_then_pair':ms,restore=install_msa(mp);msa_wrapper=module.checkpoint_blocks;ps=install_pair(pp)
    else:ps=install_pair(pp);ms,restore=install_msa(mp)
    model.load_state_dict(state);model.zero_grad(set_to_none=True);m,z,s=[x.clone().requires_grad_(True) for x in values];torch.manual_seed(314159)
    _,mz=module.checkpoint_blocks(blocks,(m,z),1)
    out_s,out_z=pair(s,mz,pair_mask=None,triangle_attention='torch',triangle_multiplicative='torch')
    (out_s.square().mean()+out_z.square().mean()).backward()
    tensors={'s':out_s.detach(),'z':out_z.detach(),'msa_z':mz.detach(),'m_grad':m.grad,'z_grad':z.grad,'s_grad':s.grad,**{n+'_grad':p.grad for n,p in model.named_parameters() if p.grad is not None}}
    assert all(torch.isfinite(v).all() for v in tensors.values());rng=torch.get_rng_state().clone()
    if reference is None:reference=({n:v.clone() for n,v in tensors.items()},rng)
    assert tensors.keys()==reference[0].keys();differences=[n for n,v in tensors.items() if not torch.equal(v,reference[0][n])]
    assert not differences and torch.equal(rng,reference[1]),differences
    assert ms['matched_gradient_calls']==int(mp>0) and ms['retained_block_calls']==mp
    assert ps['matched_gradient_calls']==int(pp>0)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in state.items())
    before=dict(ms)
    with torch.no_grad():module.checkpoint_blocks(blocks,(values[0],values[1]),1)
    assert ms==before
    # Restore nested wrappers in reverse installation order.
    if order=='msa_then_pair':module.checkpoint_blocks=msa_wrapper;restore()
    if order=='pair_then_msa':restore()
    module.checkpoint_blocks=original
    row={'pair_prefix':pp,'msa_prefix':mp,'order':order,'tensor_count':len(tensors),'all_exact':True,'rng_exact':True,'no_grad_bypass':True,'msa_stats':ms,'pair_stats':ps};rows.append(row)
    (r/'MSA_CPU_PROGRESS.json').write_text(json.dumps(rows,indent=2)+'\n')
 assert not torch.cuda.is_initialized()
 report={'status':'PASS_REAL_MSA_PAIR_CPU_RETENTION','cases':rows,'cuda_initialized':False,'accepted':False,'scope':'FP32 CPU small-width actual4MSABlocks+48PairformerBlocks with stochastic dropout. Not native-width CUDA memory,performance or full training acceptance.'}
 (r/'MSA_CPU_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'cases':len(rows),'tensors_per_case':len(tensors)}))
finally:module.checkpoint_blocks=original
