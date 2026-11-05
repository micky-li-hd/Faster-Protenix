"""Actual four-block MSA +48-block Pairformer composition, CPU only."""
import os,sys,json,functools
from pathlib import Path
r=Path(__file__).resolve().parent
assert os.environ['CUDA_VISIBLE_DEVICES']==''
os.environ['LAYERNORM_TYPE']='torch';sys.path.insert(0,str(r/'reference'))
import torch
import torch.utils.checkpoint
import protenix.model.modules.pairformer as module
from partial_msa_checkpoint import install as install_msa
from partial_pair_checkpoint import install as install_pair
from trim_padding import install as install_padding
torch.set_num_threads(2);torch.manual_seed(20261027)
msa=torch.nn.ModuleList([module.MSABlock(c_m=32,c_z=16,c_hidden=4,is_last_block=i==3,msa_dropout=.15,pair_dropout=.25,msa_chunk_size=4) for i in range(4)]).train()
for child in msa.modules():
    if isinstance(child,module.MSAStack):child.msa_max_size=16
pair=module.PairformerStack(n_blocks=48,n_heads=4,c_z=16,c_s=32,dropout=.25,blocks_per_ckpt=1).train()
model=torch.nn.ModuleDict({'msa':msa,'pair':pair})
with torch.no_grad():
    for p in model.parameters():
        if p.ndim>=2 and torch.count_nonzero(p)==0:p.normal_(0,.015)
values=[torch.randn(5,4,32),torch.randn(4,4,16),torch.randn(4,32)]
blocks=[functools.partial(b,pair_mask=None,triangle_attention='torch',triangle_multiplicative='torch',inplace_safe=False,chunk_size=None) for b in msa]
original=module.checkpoint_blocks;reference=None;rows=[]
for pp,mp,padding in [(0,0,False),(4,4,False),(4,4,True),(8,4,True),(8,2,True)]:
    module.checkpoint_blocks=original
    ps=install_pair(pp);ms,restore_msa=install_msa(mp);stats=restore_padding=None
    if padding:stats,restore_padding=install_padding()
    try:
        model.zero_grad(set_to_none=True);m,z,s=[v.clone().requires_grad_(True) for v in values];torch.manual_seed(314159)
        _,mz=module.checkpoint_blocks(blocks,(m,z),1)
        out_s,out_z=pair(s,mz,pair_mask=None,triangle_attention='torch',triangle_multiplicative='torch')
        (out_s.square().mean()+out_z.square().mean()).backward()
        tensors={'s':out_s.detach(),'z':out_z.detach(),'msa_z':mz.detach(),'m_grad':m.grad,'z_grad':z.grad,'s_grad':s.grad,**{n+'_grad':p.grad for n,p in model.named_parameters() if p.grad is not None}}
        rng=torch.get_rng_state().clone()
        if reference is None:reference=({n:v.clone() for n,v in tensors.items()},rng)
        assert tensors.keys()==reference[0].keys()
        bad=[n for n,v in tensors.items() if not torch.equal(v,reference[0][n])]
        assert not bad,(pp,mp,padding,bad)
        assert all(torch.isfinite(v).all() for v in tensors.values()) and torch.equal(rng,reference[1])
        if padding:assert stats['matched_calls']>0 and stats['omitted_rows_per_stage']>0
        rows.append({'pair_prefix':pp,'msa_prefix':mp,'padding':padding,'tensor_count':len(tensors),'all_exact':True,'rng_exact':True,'stats':stats})
    finally:
        if restore_padding:restore_padding()
        restore_msa();module.checkpoint_blocks=original
assert not torch.cuda.is_initialized()
report={'status':'PASS_MSA_PADDING_RETENTION_COMPOSITION','accepted':False,'cases':rows,'cuda_initialized':False,'scope':'Small-width actual4MSA+48Pairformer CPU proof; native GPU training still required.'}
(r/'MSA_PADDING_COMPOSITION_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'cases':len(rows),'tensors_per_case':len(tensors)}))
