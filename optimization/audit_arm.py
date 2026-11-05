"""Audit a completed bounded run including both native checkpoints."""
import argparse,hashlib,json,math,re,sys,csv
from pathlib import Path
import torch
import compare_traces
from extract_eval import extract
p=argparse.ArgumentParser();p.add_argument('tag');p.add_argument('--steps',type=int,required=True);a=p.parse_args()
root=Path('/root/tianbali/protenix_8h20_20261103_round31');run=root/'runs'/a.tag
assert run.parent==root/'runs'
state=json.loads((root/'plans'/a.tag/'STATE.json').read_text())
assert state['status']=='COMPLETED'
assert json.loads((run/'cleanup.json').read_text())['status']=='CLEAN'
assert state['runs'][0]['private_shm_cleanup']['status']=='CLEAN'
from native_build_audit import verify as verify_native_build
hybrid_requested=json.loads((run/'rank0.policy.json').read_text())['hybrid_layernorm']
assert hybrid_requested==state['runs'][0]['requested_native_compile']
hybrid_build=verify_native_build(root,run,hybrid_requested,state['runs'][0]['nonce'])
from affine_build_audit import verify as verify_affine_build
mainloop_requested=json.loads((run/'rank0.policy.json').read_text())['mainloop_affine']
affine_build=verify_affine_build(root,run,mainloop_requested,state['runs'][0]['nonce'])
compare_traces.STEPS=a.steps
traces=compare_traces.load(run)
for rank,rows in traces.items():
 assert (run/f'rank{rank}.complete').is_file()
 policy=json.loads((run/f'rank{rank}.policy.json').read_text())
 from guarded_silu import verify_stats as verify_silu
 verify_silu(json.loads((run/f'rank{rank}.silu_stats.json').read_text()),policy['silu_recompute'])
 from affine16_audit import verify_stats as verify_affine16
 verify_affine16(json.loads((run/f'rank{rank}.affine16_stats.json').read_text()),policy['affine16'])
 from cpu_placement_audit import verify_receipt
 verify_receipt(root,json.loads((run/f'rank{rank}.cpu_placement.json').read_text()),policy['cpu_placement'],rank)
 assert policy['hybrid_layernorm']==hybrid_requested
 assert policy['mainloop_affine']==mainloop_requested==json.loads((root/'performance_policies.json').read_text())[policy['policy']].get('_mainloop_affine',False)
 paircopy=policy['pair_transpose']
 assert paircopy==json.loads((root/'performance_policies.json').read_text())[policy['policy']].get('_pair_transpose',False)
 copy_stats=json.loads((run/f'rank{rank}.pair_transpose_stats.json').read_text())
 assert copy_stats['enabled']==paircopy
 if paircopy:
  assert copy_stats['eligible_calls']>0 and copy_stats['tile']==4
  assert copy_stats['original_forward_sha256']==json.loads((root/'PAIR_TRANSPOSE_INTEGRATION.json').read_text())['forward_sha256']
 else:assert copy_stats=={'enabled':False}
 msa_prefix=policy['retained_msa_prefix']
 assert msa_prefix==json.loads((root/'performance_policies.json').read_text())[policy['policy']].get('_retained_msa_prefix',0)
 msa_stats=json.loads((run/f'rank{rank}.msa_prefix_stats.json').read_text())
 assert msa_stats['retained_prefix']==msa_prefix
 assert msa_stats['retained_block_calls']==msa_prefix*msa_stats['matched_gradient_calls']
 assert msa_stats['matched_gradient_calls']>=(a.steps if msa_prefix else 0)
 if not msa_prefix:assert msa_stats['matched_gradient_calls']==0
 padding=policy['trim_msa_padding']
 assert padding==json.loads((root/'performance_policies.json').read_text())[policy['policy']].get('_trim_msa_padding',False)
 padding_stats=json.loads((run/f'rank{rank}.msa_padding_stats.json').read_text())
 assert padding_stats['enabled']==padding
 if padding:assert padding_stats['matched_calls']>=a.steps and padding_stats['omitted_rows_per_stage']>0
 else:assert padding_stats=={'enabled':False}
 head_gate_backward=policy['head_gate_backward']
 assert head_gate_backward==json.loads((root/'performance_policies.json').read_text())[policy['policy']].get('_head_gate_backward',False)
 head_gate=policy['head_gate']
 assert head_gate==json.loads((root/'performance_policies.json').read_text())[policy['policy']].get('_head_gate',False)
 head_stats=json.loads((run/f'rank{rank}.head_gate_stats.json').read_text())
 assert head_stats['enabled']==head_gate
 assert head_stats.get('backward_fused',False)==head_gate_backward
 if head_gate:assert head_stats['eligible_calls']>=a.steps and head_stats['block']==1024
 else:assert head_stats=={'enabled':False}
 view=policy['ddp_gradient_bucket_view']
 assert view==json.loads((root/'performance_policies.json').read_text())[policy['policy']].get('_ddp_gradient_bucket_view',False)
 storage=json.loads((run/f'rank{rank}.ddp_storage.json').read_text())
 assert storage['installation']==({'enabled':True,'instances':1,'gradient_as_bucket_view':True,'static_graph':True,'find_unused_parameters':False} if view else {'enabled':False})
 assert len(storage['steps'])==a.steps
 for i,record in enumerate(storage['steps']):
  assert record['step']==i and record['ddp_gradient_as_bucket_view']==view
  assert record['gradient_count']==rows[i]['grad_count']>0
  assert record['gradient_bytes']>0 and record['unique_gradient_storage_bytes']>=record['gradient_bytes']
  if i>=2:assert record['view_count']==(record['gradient_count'] if view else 0)
 local_cache=policy['local_conditioning_cache']
 assert local_cache==json.loads((root/'performance_policies.json').read_text())[policy['policy']].get('_local_conditioning_cache',False)
 local_stats=json.loads((run/f'rank{rank}.local_conditioning_stats.json').read_text())
 assert local_stats['enabled']==local_cache
 if local_cache:
  assert local_stats['invocations']==a.steps and local_stats['hits']==38*a.steps and local_stats['misses']==2*a.steps
  assert local_stats['peak_output_bytes']>0
  assert local_stats['native_encoder_source_sha256']==json.loads((root/'operator_evidence/local_conditioning_CPU_CHECK.json').read_text())['cases'][0]['source_sha256']
 else:assert local_stats=={'enabled':False}
 global_cache=policy['global_sampling_projection_cache']
 assert global_cache==json.loads((root/'performance_policies.json').read_text())[policy['policy']].get('_global_sampling_projection_cache',False)
 cache_stats=json.loads((run/f'rank{rank}.global_projection_stats.json').read_text())
 assert cache_stats['enabled']==global_cache
 if global_cache:
  assert cache_stats['invocations']==a.steps and cache_stats['hits']>0 and cache_stats['misses']>0 and cache_stats['bypasses']==0
  assert cache_stats['peak_output_bytes']>0
  source_proof=json.loads((root/'operator_evidence/global_sampling_CPU_CHECK.json').read_text())['cases'][0]['training_stats']
  assert cache_stats['standard_source_sha256']==source_proof['standard_source_sha256'] and cache_stats['local_source_sha256']==source_proof['local_source_sha256']
 else:assert cache_stats=={'enabled':False}
 selective=json.loads((run/f'rank{rank}.selective_cache_stats.json').read_text())
 assert selective['bytes_per_block']==policy['selective_cache_bytes']
 if policy['selective_cache_bytes']:
  assert selective['stack_invocations']>=a.steps and selective['contexts']==48*selective['stack_invocations']
  assert selective['cache_snapshot_copies']==selective['saved_op_calls']>0 and selective['cached_recompute_hits']>0
  assert selective['shape_eligible_calls']>0 and selective['shape_rejected_calls']>0
  assert selective['max_context_saved_bytes']<=policy['selective_cache_bytes'] and selective['cache_mutation_check'] is True
 else:
  assert selective['saved_op_calls']==selective['cached_recompute_hits']==selective['cache_snapshot_copies']==0
 allocator=json.loads((run/f'rank{rank}.allocator_stats.json').read_text())
 assert allocator['skipped_empty_cache']==(a.steps if policy['retain_allocator'] else 0)
 if policy['retain_allocator']:
  proof=json.loads((root/'ALLOCATOR_CPU_CHECK.json').read_text())
  assert allocator['original_train_step_sha256']==proof['source_policy']['original_train_step_sha256']
 compilation=json.loads((run/f'rank{rank}.compile_stats.json').read_text())
 assert compilation['requested_torch_compile']==policy['compile_layernorm']
 assert (policy['compile_layernorm'] or policy['compile_attention'])==state['runs'][0]['requested_torch_compile']
 attention=json.loads((run/f'rank{rank}.attention_stats.json').read_text())
 assert attention['requested_torch_compile']==policy['compile_attention']
 assert attention['calls']==attention['eligible_calls']+attention['bypassed_calls']
 assert not policy['compile_attention'] or attention['eligible_calls']>0
 from attention_codegen_audit import verify
 verify(run,rank,policy['compile_attention'])
 if policy['compile_layernorm']:
  assert compilation['eligible_calls']>0 and compilation['graphs']
  assert compilation['calls']==compilation['eligible_calls']+compilation['bypassed_calls']
  assert compilation['dispatch']=={'last_dimension':128,'minimum_rows':147456}
 stats=json.loads((run/f'rank{rank}.policy_stats.json').read_text())
 assert stats['retained_prefix']==policy['retained_pair_prefix']
 assert policy['retained_pair_prefix']==0 or stats['matched_gradient_calls']>=a.steps
 from diffusion_selective_audit import verify_stats as verify_diffusion_selective
 verify_diffusion_selective(json.loads((run/f'rank{rank}.diffusion_selective_stats.json').read_text()),policy['diffusion_selective_policy'],a.steps)
 diffusion=json.loads((run/f'rank{rank}.diffusion_prefix_stats.json').read_text())
 assert diffusion['retained_prefix']==policy['retained_diffusion_prefix']
 assert diffusion['retained_block_calls']==diffusion['matched_gradient_calls']*diffusion['retained_prefix']
 assert diffusion['retained_prefix']==0 or diffusion['matched_gradient_calls']>=a.steps
 initial=json.loads((run/f'rank{rank}.initial.json').read_text())
 assert all(re.fullmatch('[0-9a-f]{64}',initial[k]) for k in ['model_sha256','optimizer_sha256'])
 for row in rows:
  assert all(re.fullmatch('[0-9a-f]{64}',row[k]) for k in ['input_sha256','rng_before_sha256','rng_after_sha256'])
  assert math.isfinite(row['input_audit_seconds']) and row['input_audit_seconds']>=0
native_eval=extract((run/'process.log').read_text(),a.steps-1)
(run/'eval.json').write_text(json.dumps(native_eval,indent=2)+'\n')
folders=list((run/'native').glob('*/checkpoints'));assert len(folders)==1
torch.set_num_threads(1)
reports=[]
for suffix in ['', '_ema_0.999']:
 f=folders[0]/f'{a.steps-1}{suffix}.pt'
 obj=torch.load(f,map_location='cpu',weights_only=True,mmap=True);assert obj['step']==a.steps-1
 counts={};bad=[]
 def visit(x,path):
  if isinstance(x,torch.Tensor):
   counts[path.split('/')[0]]=counts.get(path.split('/')[0],0)+1
   if not bool(torch.isfinite(x).all()):bad.append(path)
  elif isinstance(x,dict):
   for k,v in x.items():visit(v,path+'/'+str(k))
  elif isinstance(x,(list,tuple)):
   for i,v in enumerate(x):visit(v,path+'/'+str(i))
  elif isinstance(x,float) and not math.isfinite(x):bad.append(path)
 for part in ['model','optimizer','scheduler']:visit(obj[part],part)
 assert counts.get('model',0)>0 and counts.get('optimizer',0)>0 and not bad
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
 reports.append({'file':str(f.relative_to(run)),'sha256':h.hexdigest(),'bytes':f.stat().st_size,'tensor_counts':counts,'nonfinite':bad,'finite':True,'step':obj['step']});del obj
memory=[json.loads((run/f'rank{rank}.memory.json').read_text()) for rank in range(8)]
assert all(0<m['peak_allocated_bytes']<=m['peak_reserved_bytes'] for m in memory)
gpu_samples={}
with (run/'gpu_utilization.csv').open() as stream:
 for row in csv.reader(stream):
  if len(row)!=7:continue
  index=int(row[1]);total=float(row[3]);used=float(row[4]);util=float(row[5])
  record=gpu_samples.setdefault(index,{'samples':0,'total_mib':total,'peak_used_mib':0,'peak_util_percent':0})
  record['samples']+=1;record['peak_used_mib']=max(record['peak_used_mib'],used);record['peak_util_percent']=max(record['peak_util_percent'],util)
assert set(gpu_samples)==set(range(8)) and all(x['samples']>0 for x in gpu_samples.values())
report={'native_build':hybrid_build,'affine_build':affine_build,'full_run_memory':memory,'gpu_samples':gpu_samples,'status':'PASS','ranks':8,'updates':a.steps,'checkpoints':reports,'evaluation_metrics':len(native_eval['metrics']),'process_seconds':state['runs'][0]['process_seconds'],'scope':'Short-run integrity/finiteness only; does not prove numerical equivalence or convergence'}
with (run/'audit.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report))
