"""Read-only startup observers around the native training entry point."""
import os,sys,json,time,math
from pathlib import Path
root=Path('/root/tianbali/protenix_8h20_20261103_round31')
sys.path.insert(0,str(root/'reference'))
from recipe import argv
spec=json.loads((root/'recipe.json').read_text())
spec['overrides'].update(max_steps=int(os.environ['RSI_STEPS']),log_interval=1)
spec['overrides']['enable_efficient_fusion']=os.environ.get('RSI_FUSION','0')=='1'
policies=json.loads((root/'performance_policies.json').read_text())
policy=os.environ.get('RSI_POLICY','reference')
assert policy in policies, ('Unknown performance policy',policy)
policy_config=dict(policies[policy])
silu_recompute=policy_config.pop('_silu_recompute',False)
affine16=policy_config.pop('_affine16',False)
cpu_placement=policy_config.pop('_cpu_placement',None)
assert cpu_placement in (None,'gpu_local_disjoint')
retained_diffusion_prefix=policy_config.pop('_retained_diffusion_prefix',0)
diffusion_selective_policy=policy_config.pop('_diffusion_selective_policy',None)
assert diffusion_selective_policy in (None,'mm')
assert not diffusion_selective_policy or retained_diffusion_prefix==0
retained_pair_prefix=policy_config.pop('_retained_pair_prefix',0)
retained_msa_prefix=policy_config.pop('_retained_msa_prefix',0)
trim_msa_padding=policy_config.pop('_trim_msa_padding',False)
head_gate=policy_config.pop('_head_gate',False)
head_gate_backward=policy_config.pop('_head_gate_backward',False)
assert not head_gate_backward or head_gate
hybrid_layernorm=policy_config.pop('_hybrid_layernorm',False)
mainloop_affine=policy_config.pop('_mainloop_affine',False)
pair_transpose=policy_config.pop('_pair_transpose',False)
assert not mainloop_affine or hybrid_layernorm
assert not affine16 or mainloop_affine
compile_attention=policy_config.pop('_compile_attention',False)
compile_layernorm=policy_config.pop('_compile_layernorm',False)
retain_allocator=policy_config.pop('_retain_allocator',False)
selective_cache_bytes=policy_config.pop('_selective_pair_cache_bytes',0)
global_sampling_projection_cache=policy_config.pop('_global_sampling_projection_cache',False)
local_conditioning_cache=policy_config.pop('_local_conditioning_cache',False)
ddp_gradient_bucket_view=policy_config.pop('_ddp_gradient_bucket_view',False)
spec['overrides'].update(policy_config)
rank=int(os.environ['RANK']);out=Path(os.environ['RSI_OUT']);out.mkdir(exist_ok=True)
sys.argv=[str(root/'reference/runner/train.py')]+argv(spec,out/'native',20261027)
os.environ['PROTENIX_ROOT_DIR']=spec['data_root']
import faulthandler
fault_log=open(out/f'rank{rank}.fault.log','x',buffering=1)
faulthandler.enable(file=fault_log)
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(out/'cache'/f'rank{rank}')
os.environ['TRITON_CACHE_DIR']=str(out/'triton'/f'rank{rank}')
from cpu_placement import install as install_cpu_placement
cpu_placement_stats,restore_cpu_placement=install_cpu_placement(cpu_placement,rank,root/'CPU_PLACEMENT_PLAN.json')
import torch
import torch.utils.checkpoint
from input_fingerprint import fingerprint, rng_fingerprint
import runner.train as training
from partial_pair_checkpoint import install
partial_checkpoint_stats=install(retained_pair_prefix)
from partial_msa_checkpoint import install as install_msa
msa_checkpoint_stats,restore_msa=install_msa(retained_msa_prefix)
from partial_diffusion_checkpoint import install as install_diffusion
diffusion_checkpoint_stats=install_diffusion(retained_diffusion_prefix)
diffusion_selective_stats={"policy":None}
if diffusion_selective_policy:
 from selective_diffusion_cache import install as install_diffusion_selective
 diffusion_selective_stats,restore_diffusion_selective=install_diffusion_selective(diffusion_selective_policy)
selective_cache_stats={'bytes_per_block':0,'stack_invocations':0,'saved_op_calls':0,'cached_recompute_hits':0,'cache_snapshot_copies':0}
if selective_cache_bytes:
 from selective_pair_cache import install as install_selective_cache
 selective_cache_stats=install_selective_cache(selective_cache_bytes)
allocator_stats={'policy':'disabled','skipped_empty_cache':0}
if retain_allocator:
 from retain_allocator import install as install_allocator
 allocator_stats=install_allocator(training.AF3Trainer)
compile_stats={'requested_torch_compile':False}
if compile_layernorm:
 from protenix.model.triangular.layers import OpenFoldLayerNorm
 from compiled_layernorm import install as install_ln
 compile_stats=install_ln(OpenFoldLayerNorm,torch)
attention_stats={'requested_torch_compile':False,'calls':0,'eligible_calls':0,'bypassed_calls':0,'shapes':{}}
if compile_attention:
 import protenix.model.modules.primitives as attention_module
 from guarded_attention import install as install_attention
 attention_stats=install_attention(attention_module,torch)
hybrid_stats={'enabled':False};affine_stats={'enabled':False}
if hybrid_layernorm:
 from protenix.model.triangular.layers import OpenFoldLayerNorm
 if mainloop_affine:
  from dual_cold_install import install as install_dual
  hybrid_stats,affine_stats=install_dual(out,OpenFoldLayerNorm)
 else:
  from cold_build import install as install_hybrid
  hybrid_stats={'enabled':True,**install_hybrid(out,OpenFoldLayerNorm)}
affine16_stats={'enabled':False}
if affine16:
 from guarded_affine16 import install as install_affine16
 import ordered_affine
 restore_affine16,affine16_stats=install_affine16(OpenFoldLayerNorm,ordered_affine._EXTENSION)
 affine16_stats['enabled']=True
silu_stats={'enabled':False}
if silu_recompute:
 from protenix.model.modules.transformer import ConditionedTransitionBlock
 from guarded_silu import install as install_silu
 restore_silu,silu_stats=install_silu(ConditionedTransitionBlock)
(out/f'rank{rank}.policy.json').write_text(json.dumps({"policy":policy,"silu_recompute":silu_recompute,"affine16":affine16,"cpu_placement":cpu_placement,"ddp_gradient_bucket_view":ddp_gradient_bucket_view,"local_conditioning_cache":local_conditioning_cache,"global_sampling_projection_cache":global_sampling_projection_cache,"hybrid_layernorm":hybrid_layernorm,"mainloop_affine":mainloop_affine,"pair_transpose":pair_transpose,"retained_pair_prefix":retained_pair_prefix,"retained_msa_prefix":retained_msa_prefix,"trim_msa_padding":trim_msa_padding,"head_gate":head_gate,"head_gate_backward":head_gate_backward,"retained_diffusion_prefix":retained_diffusion_prefix,"diffusion_selective_policy":diffusion_selective_policy,"compile_layernorm":compile_layernorm,"compile_attention":compile_attention,"retain_allocator":retain_allocator,"selective_cache_bytes":selective_cache_bytes,"config_overrides":policy_config}))
global_projection_stats={'enabled':False}
if global_sampling_projection_cache:
 from global_sampling_cache import install as install_global_projection
 from protenix.model.protenix import Protenix
 from protenix.model.modules.transformer import AttentionPairBias
 restore_global_projection,global_projection_stats=install_global_projection(Protenix,AttentionPairBias)
 global_projection_stats['enabled']=True
local_conditioning_stats={'enabled':False}
if local_conditioning_cache:
 from local_conditioning_cache import install as install_local_encoder, install_sampler as install_local_sampler
 from protenix.model.protenix import Protenix
 from protenix.model.modules.transformer import AtomAttentionEncoder
 restore_local_encoder,local_encoder_source=install_local_encoder(AtomAttentionEncoder)
 restore_local_sampler,local_conditioning_stats=install_local_sampler(Protenix)
 local_conditioning_stats.update(enabled=True,native_encoder_source_sha256=local_encoder_source)
ddp_view_stats={'enabled':False}
if ddp_gradient_bucket_view:
 from bucket_view import install as install_bucket_view
 restore_bucket_view,ddp_view_stats=install_bucket_view(training)
ddp_storage_rows=[]
original_loss=training.AF3Trainer.get_loss
original_update=training.AF3Trainer.update
original_step=training.AF3Trainer.train_step
rows=open(out/f'rank{rank}.jsonl','x',buffering=1)
def get_loss(self,*args,**kwargs):
 result=original_loss(self,*args,**kwargs)
 if kwargs.get('mode')=='train':
  loss,components,_=result
  self._probe_loss=loss.detach()
  self._probe_components={k:v.detach() if isinstance(v,torch.Tensor) else v for k,v in components.items() if 'loss' in k}
 return result
def update(self):
 grads=[p.grad for p in self.model.parameters() if p.grad is not None]
 # Store scalars only: no retained gradient tensors or extra device work.
 storages={g.untyped_storage().data_ptr():g.untyped_storage().nbytes() for g in grads}
 views=sum(g._base is not None for g in grads)
 ddp_storage_rows.append({'step':self.step,'gradient_count':len(grads),'view_count':views,'gradient_bytes':sum(g.numel()*g.element_size() for g in grads),'unique_gradient_storage_bytes':sum(storages.values()),'unique_gradient_storages':len(storages),'allocated_after_backward':torch.cuda.memory_allocated(),'reserved_after_backward':torch.cuda.memory_reserved(),'ddp_gradient_as_bucket_view':bool(self.model.gradient_as_bucket_view)})
 if ddp_gradient_bucket_view and self.step>=2:assert views==len(grads)>0
 self._probe_grad_count=len(grads)
 self._probe_grad_finite=torch.stack([torch.isfinite(g).all() for g in grads]).all().item() if grads else False
 return original_update(self)
def train_step(self,batch):
 audit_start=time.perf_counter()
 rng_before_sha256=rng_fingerprint()
 input_sha256=fingerprint(batch)
 assert rng_fingerprint()==rng_before_sha256, 'Input observer changed RNG'
 if self.step==0:
  assert fingerprint(batch)==input_sha256, 'Input observer changed batch'
  (out/f'rank{rank}.initial.json').write_text(json.dumps({'model_sha256':fingerprint(self.model.state_dict()),'optimizer_sha256':fingerprint(self.optimizer.state_dict())}))
 input_audit_seconds=time.perf_counter()-audit_start
 torch.cuda.synchronize();start=time.perf_counter()
 result=original_step(self,batch)
 torch.cuda.synchronize();duration=time.perf_counter()-start
 rng_after_sha256=rng_fingerprint()
 loss=float(self._probe_loss.cpu())
 assert all(not isinstance(v,torch.Tensor) or v.numel()==1 for v in self._probe_components.values()), 'Nonscalar loss component must not be silently omitted'
 components={k:float(v.cpu()) if isinstance(v,torch.Tensor) and v.numel()==1 else float(v) for k,v in self._probe_components.items() if not isinstance(v,torch.Tensor) or v.numel()==1}
 row={'rng_after_sha256':rng_after_sha256,'input_sha256':input_sha256,'rng_before_sha256':rng_before_sha256,'input_audit_seconds':input_audit_seconds,'rank':rank,'step':self.step,'loss':loss,'components':components,'observed_step_seconds':duration,'grad_count':self._probe_grad_count,'grad_finite':self._probe_grad_finite,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'lr':self.optimizer.param_groups[0]['lr']}
 rows.write(json.dumps(row,allow_nan=False)+'\n')
 assert math.isfinite(loss) and row['grad_finite'], 'Nonfinite loss or gradients'
 return result
training.AF3Trainer.get_loss=get_loss
training.AF3Trainer.update=update
training.AF3Trainer.train_step=train_step
pair_transpose_stats={'enabled':False}
if pair_transpose:
 from protenix.model.modules.pairformer import PairformerBlock
 from guarded_pair_transpose import install as install_pair_transpose
 restore_pair_transpose,pair_transpose_stats=install_pair_transpose(PairformerBlock)
msa_padding_stats={'enabled':False}
if trim_msa_padding:
 from trim_padding import install as install_msa_padding
 msa_padding_stats,restore_msa_padding=install_msa_padding()
 msa_padding_stats['enabled']=True
head_gate_stats={'enabled':False}
if head_gate:
 from protenix.model.triangular.layers import Attention
 if head_gate_backward:
  from guarded_head_gate_backward import install as install_head_gate
 else:
  from guarded_head_gate import install as install_head_gate
 restore_head_gate,head_gate_stats=install_head_gate(Attention)
training.main()
if head_gate:
 assert head_gate_stats['eligible_calls']>=int(os.environ['RSI_STEPS'])
 restore_head_gate()
(out/f'rank{rank}.head_gate_stats.json').write_text(json.dumps(head_gate_stats))
if trim_msa_padding:
 assert msa_padding_stats['matched_calls']>=int(os.environ['RSI_STEPS']) and msa_padding_stats['omitted_rows_per_stage']>0
 restore_msa_padding()
(out/f'rank{rank}.msa_padding_stats.json').write_text(json.dumps(msa_padding_stats))
if pair_transpose:
 assert pair_transpose_stats['eligible_calls']>0
 restore_pair_transpose()
(out/f'rank{rank}.pair_transpose_stats.json').write_text(json.dumps(pair_transpose_stats))
if ddp_gradient_bucket_view:
 assert ddp_view_stats['instances']==1
 restore_bucket_view()
(out/f'rank{rank}.ddp_storage.json').write_text(json.dumps({'installation':ddp_view_stats,'steps':ddp_storage_rows}))
if local_conditioning_cache:
 assert local_conditioning_stats['invocations']==int(os.environ['RSI_STEPS'])
 assert local_conditioning_stats['hits']==38*int(os.environ['RSI_STEPS']) and local_conditioning_stats['misses']==2*int(os.environ['RSI_STEPS'])
 restore_local_sampler();restore_local_encoder()
(out/f'rank{rank}.local_conditioning_stats.json').write_text(json.dumps(local_conditioning_stats))
if global_sampling_projection_cache:
 assert global_projection_stats['invocations']==int(os.environ['RSI_STEPS']) and global_projection_stats['hits']>0
 restore_global_projection()
(out/f'rank{rank}.global_projection_stats.json').write_text(json.dumps(global_projection_stats))
assert not hybrid_layernorm or hybrid_stats['calls']['eligible_calls']>0
(out/f'rank{rank}.hybrid_layernorm_stats.json').write_text(json.dumps(hybrid_stats))
assert not mainloop_affine or (affine_stats['calls']['eligible_width64']>0 and affine_stats['calls']['eligible_width128']>0 and affine_stats['calls']['eligible_backward']==affine_stats['calls']['eligible_width64']+affine_stats['calls']['eligible_width128'])
(out/f'rank{rank}.affine_stats.json').write_text(json.dumps(affine_stats))
assert not compile_attention or attention_stats['eligible_calls']>0
(out/f'rank{rank}.attention_stats.json').write_text(json.dumps(attention_stats))
from attention_codegen_audit import capture
capture(out,rank,compile_attention)
from diffusion_selective_audit import verify_stats as verify_diffusion_selective
verify_diffusion_selective(diffusion_selective_stats,diffusion_selective_policy,int(os.environ['RSI_STEPS']))
(out/f'rank{rank}.diffusion_selective_stats.json').write_text(json.dumps(diffusion_selective_stats))
if diffusion_selective_policy:restore_diffusion_selective()
assert retained_diffusion_prefix==0 or diffusion_checkpoint_stats["matched_gradient_calls"]>=int(os.environ["RSI_STEPS"])
(out/f'rank{rank}.diffusion_prefix_stats.json').write_text(json.dumps(diffusion_checkpoint_stats))
assert retained_pair_prefix==0 or partial_checkpoint_stats["matched_gradient_calls"]>=int(os.environ["RSI_STEPS"])
assert not compile_layernorm or (compile_stats['eligible_calls']>0 and compile_stats['graphs'])
assert not retain_allocator or allocator_stats['skipped_empty_cache']==int(os.environ['RSI_STEPS'])
assert not selective_cache_bytes or (selective_cache_stats['stack_invocations']>=int(os.environ['RSI_STEPS']) and selective_cache_stats['cached_recompute_hits']>0)
(out/f'rank{rank}.selective_cache_stats.json').write_text(json.dumps(selective_cache_stats))
(out/f'rank{rank}.allocator_stats.json').write_text(json.dumps(allocator_stats))
(out/f'rank{rank}.compile_stats.json').write_text(json.dumps(compile_stats))
assert retained_msa_prefix==0 or msa_checkpoint_stats['matched_gradient_calls']>=int(os.environ['RSI_STEPS'])
(out/f'rank{rank}.msa_prefix_stats.json').write_text(json.dumps(msa_checkpoint_stats))
(out/f'rank{rank}.policy_stats.json').write_text(json.dumps(partial_checkpoint_stats))
(out/f'rank{rank}.memory.json').write_text(json.dumps({'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'scope':'Full native run including checkpoint and evaluation'}))
from cpu_placement_audit import finish_receipt
finish_receipt(root,out,rank,cpu_placement_stats,restore_cpu_placement)
if affine16:
 restore_affine16()
 affine16_stats['restored']=True
from affine16_audit import verify_stats as verify_affine16
verify_affine16(affine16_stats,affine16)
(out/f'rank{rank}.affine16_stats.json').write_text(json.dumps(affine16_stats))
if silu_recompute:restore_silu()
from guarded_silu import verify_stats as verify_silu
verify_silu(silu_stats,silu_recompute)
(out/f'rank{rank}.silu_stats.json').write_text(json.dumps(silu_stats))
rows.close()
(out/f'rank{rank}.complete').write_text('native training, checkpoint and evaluation returned\n')
