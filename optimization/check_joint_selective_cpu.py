"""Actual four-block MSA +48-block Pairformer composition, CPU only."""
import os,sys,json,functools,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent
assert os.environ['CUDA_VISIBLE_DEVICES']==''
os.environ['LAYERNORM_TYPE']='torch';sys.path.insert(0,str(r/'reference'))
for name,item in json.loads((r/'expected_source_fixture.json').read_text())['source'].items():
    assert hashlib.sha256((Path(str(r/'reference'))/name).read_bytes()).hexdigest()==item['sha256'],name
import torch
import torch.utils.checkpoint
import protenix.model.modules.pairformer as module
import protenix.model.modules.transformer as tm
from selective_diffusion_cache import install as install_sac
from partial_diffusion_checkpoint import install as install_diff
from partial_msa_checkpoint import install as install_msa
from partial_pair_checkpoint import install as install_pair
from trim_padding import install as install_padding
torch.set_num_threads(2);torch.manual_seed(20261027)
msa=torch.nn.ModuleList([module.MSABlock(c_m=32,c_z=16,c_hidden=4,is_last_block=i==3,msa_dropout=.15,pair_dropout=.25,msa_chunk_size=4) for i in range(4)]).train()
for child in msa.modules():
    if isinstance(child,module.MSAStack):child.msa_max_size=16
pair=module.PairformerStack(n_blocks=48,n_heads=4,c_z=16,c_s=32,dropout=.25,blocks_per_ckpt=1).train()
diff=tm.DiffusionTransformer(c_a=32,c_s=32,c_z=16,n_blocks=24,n_heads=4,drop_path_rate=.1,blocks_per_ckpt=1).train()
model=torch.nn.ModuleDict({'msa':msa,'pair':pair,'diff':diff})
with torch.no_grad():
    for p in model.parameters():
        if p.ndim>=2 and torch.count_nonzero(p)==0:p.normal_(0,.015)
values=[torch.randn(5,4,32),torch.randn(4,4,16),torch.randn(4,32)]
blocks=[functools.partial(b,pair_mask=None,triangle_attention='torch',triangle_multiplicative='torch',inplace_safe=False,chunk_size=None) for b in msa]
a_value=torch.randn(2,4,32)
original=module.checkpoint_blocks;original_diff=tm.checkpoint_blocks;reference=None;rows=[]
for pp,mp,dp,sac,padding in [(0,0,0,None,False),(4,4,24,None,True),(16,4,0,"mm",True),(19,4,0,"mm",True)]:
    module.checkpoint_blocks=original;tm.checkpoint_blocks=original_diff
    ds=install_diff(dp);cache=undo_sac=None
    if sac:cache,undo_sac=install_sac(sac)
    ps=install_pair(pp);ms,restore_msa=install_msa(mp);stats=restore_padding=None
    if padding:stats,restore_padding=install_padding()
    try:
        model.zero_grad(set_to_none=True);m,z,s=[v.clone().requires_grad_(True) for v in values];torch.manual_seed(314159)
        _,mz=module.checkpoint_blocks(blocks,(m,z),1)
        out_s,out_z=pair(s,mz,pair_mask=None,triangle_attention='torch',triangle_multiplicative='torch')
        a=a_value.clone().requires_grad_(True)
        out_a=diff(a,out_s.unsqueeze(0).expand(2,-1,-1),out_z.unsqueeze(0))
        (out_a.square().mean()+out_s.square().mean()+out_z.square().mean()).backward()
        tensors={'a':out_a.detach(),'a_grad':a.grad,'s':out_s.detach(),'z':out_z.detach(),'msa_z':mz.detach(),'m_grad':m.grad,'z_grad':z.grad,'s_grad':s.grad,**{n+'_grad':p.grad for n,p in model.named_parameters() if p.grad is not None}}
        rng=torch.get_rng_state().clone()
        if reference is None:reference=({n:v.clone() for n,v in tensors.items()},rng)
        assert tensors.keys()==reference[0].keys()
        bad=[n for n,v in tensors.items() if not torch.equal(v,reference[0][n])]
        assert not bad,(pp,mp,padding,bad)
        assert all(torch.isfinite(v).all() for v in tensors.values()) and torch.equal(rng,reference[1])
        if padding:assert stats['matched_calls']>0 and stats['omitted_rows_per_stage']>0
        if sac:
            assert cache['contexts']==24 and cache['matched_stack_calls']==1 and cache['cached_recompute_hits']>0 and cache['mutation_check']
        rows.append({'diff_prefix':dp,'selective_policy':sac,'selective_stats':cache,'pair_prefix':pp,'msa_prefix':mp,'padding':padding,'tensor_count':len(tensors),'all_exact':True,'rng_exact':True,'stats':stats})
    finally:
        if restore_padding:restore_padding()
        if undo_sac:undo_sac()
        restore_msa();module.checkpoint_blocks=original;tm.checkpoint_blocks=original_diff
assert not torch.cuda.is_initialized()
report={'status':'PASS_MSA_PAIR_DIFFUSION_SELECTIVE_COMPOSITION','accepted':False,'cases':rows,'cuda_initialized':False,'scope':'Small-width actual4MSA+48Pairformer+24Diffusion CPU composition; native GPU training still required.'}
(r/'JOINT_SELECTIVE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'cases':len(rows),'tensors_per_case':len(tensors)}))
