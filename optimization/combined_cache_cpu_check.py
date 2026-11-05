"""Native encoder within a small sampler fixture; not the full Protenix model."""
import os,sys,copy,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['LAYERNORM_TYPE']='torch'
root=Path(__file__).resolve().parent
reference=root/'reference'
if not reference.exists():reference=root.parent/'protenix_8h20_joint_retention_prepare/reference'
sys.path.insert(0,str(reference))
import torch
from protenix.model.modules.transformer import AtomAttentionEncoder,AttentionPairBias
import global_sampling_cache,projection_cache
from protenix.model.utils import expand_at_dim
import local_conditioning_cache as cache

torch.set_num_threads(2);torch.manual_seed(20261102)
class Sampler(torch.nn.Module):
 def __init__(self):
  super().__init__();self.encoder=AtomAttentionEncoder(has_coords=True,c_token=32,c_atom=16,c_atompair=8,c_s=16,c_z=8,n_blocks=1,n_heads=4,n_queries=4,n_keys=8)
  self.attention=AttentionPairBias(c_a=32,c_s=16,c_z=8,n_heads=4,has_s=True,cross_attention_mode=False)
 def sample_diffusion(self,**kwargs):
  results=[]
  for i,r in enumerate(kwargs['coords']):
   args={k:v for k,v in kwargs.items() if k not in ('coords','fail_at','nested','pair_z')}
   args['s']=expand_at_dim(args['s'],dim=-3,n=1)
   encoded=self.encoder(**args,r_l=r)
   projected=self.attention(encoded[0],args['s'],kwargs['pair_z'])
   results.append((*encoded,projected))
   if i==kwargs.get('fail_at'):raise RuntimeError('fixture exception')
  return results
m=Sampler()
with torch.no_grad():
 for p in m.parameters():
  if p.ndim>=2 and not torch.count_nonzero(p):p.normal_(0,.02)
b=copy.deepcopy(m)
# Native training mini-rollout temporarily evaluates the diffusion child.
m.encoder.eval();b.encoder.eval();m.attention.eval();b.attention.eval();assert m.training and not m.encoder.training
kw={name:None for name in ('ref_pos','ref_charge','ref_mask','ref_atom_name_chars','ref_element','d_lm','v_lm','pad_info')}
kw.update(atom_to_token_idx=torch.arange(8)//2,s=torch.randn(1,4,16),z=torch.randn(1,1,4,4,8),c_l=torch.randn(1,8,16),p_lm=torch.randn(1,1,2,4,8,8),coords=[torch.randn(1,1,8,3) for _ in range(20)])
kw['pair_z']=torch.randn(1,1,4,4,8)
with torch.no_grad():expected=b.sample_diffusion(**kw)
restore_encoder,source=cache.install(AtomAttentionEncoder)
restore_global,global_stats=global_sampling_cache.install(Sampler,AttentionPairBias,('cpu',))
restore_sampler,stats=cache.install_sampler(Sampler,('cpu',))
try:
 with torch.no_grad():
  for _ in range(2):
   actual=m.sample_diffusion(**kw)
   assert all(torch.equal(x,y) for a,e in zip(actual,expected) for x,y in zip(a,e))
   assert cache._scope.get() is None
  assert global_stats['invocations']==2 and global_stats['hits']==38 and global_stats['misses']==2,global_stats
  assert projection_cache._scope.get() is None
  assert stats['invocations']==2 and stats['hits']==76 and stats['misses']==4,stats
  try:m.sample_diffusion(**kw,fail_at=2)
  except RuntimeError as e:assert str(e)=='fixture exception'
  else:raise AssertionError('Expected exception')
  assert cache._scope.get() is None
  assert stats['invocations']==3 and stats['misses']==6
  # Outer scope is restored after an inner scope exits, including errors.
  with cache.cache_scope(('cpu',)) as outer:
   m.sample_diffusion(**kw)
   assert cache._scope.get() is outer and not outer['entries']
  before=copy.deepcopy(stats);global_before=copy.deepcopy(global_stats)
  m.eval();m.sample_diffusion(**kw);assert stats==before and global_stats==global_before
 m.train()
 m.sample_diffusion(**{**kw,'coords':kw['coords'][:1]});assert stats==before and global_stats==global_before
 assert cache._scope.get() is None
finally:restore_sampler();restore_encoder();restore_global()
report={'status':'PASS_COMBINED_CACHE_CPU_SCOPE','native_encoder_source_sha256':source,'new_expanded_s_view_each_step':True,'two_independent_scopes_each38hits2misses':True,'exception_and_nested_scope_cleanup':True,'evaluation_and_grad_bypass':True,'training_parent_eval_child_reuse':True,'stats':stats,'global_stats':global_stats,'accepted':False,'scope':'Synthetic sampler calls native encoder and native expand_at_dim; actual full-model sampler continuity remains unverified.'}
(root/'COMBINED_CACHE_CPU_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
