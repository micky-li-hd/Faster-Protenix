"""One-shot nonce-owned CPU preflight followed by immutable training bindings."""
import os,sys,json,secrets,subprocess,time,signal,hashlib
from pathlib import Path
from ownership import OwnedProcesses,OWNERSHIP_ENV
root=Path(__file__).resolve().parent
assert not (root/'CPU_PREFLIGHT_RUN.json').exists()
for n,h in json.loads((root/'PREPARATION_BINDINGS.json').read_text()).items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
proof=json.loads((root/'operator_evidence/diffusion_gpu_OPERATOR_AUDIT.json').read_text())
assert proof['numerical_passed_all_ranks'] and len(proof['ranks'])==8
source_hash=hashlib.sha256((root/'partial_diffusion_checkpoint.py').read_bytes()).hexdigest()
assert json.loads((root/'operator_evidence/diffusion_gpu_bindings.json').read_text())['partial_diffusion_checkpoint.py']==source_hash
attention_proof=json.loads((root/'operator_evidence/attention_gpu_OPERATOR_AUDIT.json').read_text())
assert attention_proof['numerical_passed_all_ranks'] and len(attention_proof['ranks'])==8
assert hashlib.sha256((root/'guarded_attention.py').read_bytes()).hexdigest()==json.loads((root/'operator_evidence/attention_gpu_bindings.json').read_text())['guarded_attention.py']
# Required real isolated cold-build proof must exist before any preflight or GPU launch.
proof=json.loads((root/'operator_evidence/hybrid_cold_CPU_PROOF.json').read_text())
run=json.loads((root/'operator_evidence/hybrid_cold_CPU_RUN.json').read_text())
assert proof['status']=='REAL_COLD_BUILD_8_WORKER_CPU_IMPORT_PASS' and len(proof['rows'])==8
assert run['status']=='COMPLETED' and run['cleanup']=='CLEAN'
assert sum(row['stats']['compiled_this_rank'] for row in proof['rows'])==1
assert all(row['status']=='CPU_FALLBACK_AND_BINARY_IMPORT_PASS' and row['rng_exact'] and row['parameter_identity_preserved'] and not row['cuda_initialized'] for row in proof['rows'])
proof_binding=json.loads((root/'operator_evidence/hybrid_cold_PREPARATION_BINDINGS.json').read_text())
for n in ['cold_build.py','compile_child.py','hybrid_layernorm.py','guarded_layernorm.py','isolated_layernorm.py','BUILD_MANIFEST.json']:
 assert hashlib.sha256((root/n).read_bytes()).hexdigest()==proof_binding[n],n
operator=json.loads((root/'operator_evidence/hybrid_gpu_OPERATOR_AUDIT.json').read_text())
assert operator['numerical_passed_all_ranks'] and len(operator['ranks'])==8
old_operator=json.loads((root/'operator_evidence/hybrid128_OPERATOR_AUDIT.json').read_text())
assert old_operator['numerical_passed_all_ranks'] and len(old_operator['ranks'])==8
b128=json.loads((root/'operator_evidence/hybrid128_bindings.json').read_text());b64=json.loads((root/'operator_evidence/hybrid_gpu_bindings.json').read_text())
for n in ['hybrid_layernorm.py','isolated_layernorm.py']:
 assert hashlib.sha256((root/n).read_bytes()).hexdigest()==b128[n]==b64[n],n
union=root/'operator_evidence/union_cpu'
for n,h in json.loads((union/'PREPARATION_BINDINGS.json').read_text()).items():assert hashlib.sha256((union/n).read_bytes()).hexdigest()==h,n
proof_union=json.loads((union/'DISPATCH_UNION_CHECK.json').read_text());assert proof_union['status']=='PASS_COMBINED_GUARD_EQUALS_UNION' and proof_union['metadata_cases']==1440
expected_guards={'protenix_8h20_bf16_hybrid_prepare':b128['guarded_layernorm.py'],'protenix_8h20_width64_hybrid_prepare':b64['guarded_layernorm.py'],'protenix_8h20_width64_128_hybrid_prepare':hashlib.sha256((root/'guarded_layernorm.py').read_bytes()).hexdigest()}
for folder,h in expected_guards.items():
 matches=[v for path,v in proof_union['sources'].items() if Path(path).parent.name==folder];assert matches==[h],folder
assert (root/'guarded_layernorm.py').read_bytes()==(union/'guarded_layernorm.py').read_bytes()
projection=json.loads((root/'operator_evidence/projection_OPERATOR_AUDIT.json').read_text())
assert projection['numerical_passed_all_ranks'] and projection['cases']==48 and projection['comparisons']==projection['exact']==13440
assert hashlib.sha256((root/'projection_cache.py').read_bytes()).hexdigest()==json.loads((root/'operator_evidence/projection_bindings.json').read_text())['projection_cache.py']
scope_proof=json.loads((root/'operator_evidence/global_sampling_CPU_CHECK.json').read_text())
assert scope_proof['status']=='PASS_GLOBAL_TRAINING_ONLY_CPU_SCOPE' and len(scope_proof['cases'])==4
for n in ['projection_cache.py','global_sampling_cache.py']:assert hashlib.sha256((root/n).read_bytes()).hexdigest()==scope_proof['sources'][n]
native_proof=json.loads((root/'operator_evidence/projection_NATIVE8_REVIEW.json').read_text())
assert native_proof['actual_training_global_hits_on_all_ranks'] and native_proof['comparisons']['observed_vs_matched_parent']['passed']
local_audit=json.loads((root/'operator_evidence/local_conditioning_OPERATOR_AUDIT.json').read_text())
assert len(local_audit['ranks'])==8 and all(not r['failures'] and r['cases']==6 for r in local_audit['ranks'])
assert hashlib.sha256((root/'local_conditioning_cache.py').read_bytes()).hexdigest()==json.loads((root/'operator_evidence/local_conditioning_bindings.json').read_text())['local_conditioning_cache.py']
local_scope=json.loads((root/'operator_evidence/local_conditioning_SCOPE_CPU_CHECK.json').read_text())
assert local_scope['training_parent_eval_child_reuse'] and local_scope['evaluation_and_grad_bypass']
ddp=json.loads((root/'operator_evidence/ddp_RESULT_REVIEW.json').read_text())
assert ddp['all_exact_comparisons']==4608 and ddp['cases']==32
assert hashlib.sha256((root/'bucket_view.py').read_bytes()).hexdigest()==json.loads((root/'operator_evidence/ddp_bindings.json').read_text())['bucket_view.py']
native_bucket=json.loads((root/'operator_evidence/native_bucket_DDP_STORAGE_REVIEW.json').read_text())
assert native_bucket['all8_candidate_gradients_are_views_after_warmup'] and len(native_bucket['ranks'])==8
native_result=json.loads((root/'operator_evidence/native_bucket_ROUND16_RESULT_REVIEW.json').read_text())
assert native_result['all_training_metrics_passed']==36 and native_result['all_supplemental_eval_metrics_passed']==92
combined=json.loads((root/'operator_evidence/combined_RESULT_REVIEW.json').read_text())
assert combined['cases']==576 and combined['comparisons']==24192 and combined['failed_native']==combined['failed_parent_exact']==0
bindings=json.loads((root/'operator_evidence/combined_bindings.json').read_text())
for n in ['combined_layernorm.py','ordered_affine.py']:assert hashlib.sha256((root/n).read_bytes()).hexdigest()==bindings[n],n
assert hashlib.sha256((root/'combined_guarded_layernorm.py').read_bytes()).hexdigest()==bindings['guarded_layernorm.py']
proof=json.loads((root/'operator_evidence/affine_CPU_PROOF.json').read_text());run=json.loads((root/'operator_evidence/affine_CPU_RUN.json').read_text());assert run['status']=='COMPLETED' and run['cleanup']=='CLEAN' and len(proof['rows'])==8
ab=json.loads((root/'operator_evidence/affine_PREPARATION_BINDINGS.json').read_text())
for n in ['cold_build.py','compile_child.py']:assert hashlib.sha256((root/'affine'/n).read_bytes()).hexdigest()==ab[n],n
from affine16_audit import verify_sources
verify_sources(root)
silu_proof=json.loads((root/'operator_evidence/silu_stack_DECISION_REVIEW.json').read_text())
assert silu_proof['all_bit_exact'] and silu_proof['tensor_checks']==80064
silu_bindings=json.loads((root/'operator_evidence/silu_stack_bindings.json').read_text())
assert hashlib.sha256((root/'recompute_silu.py').read_bytes()).hexdigest()==silu_bindings['recompute_silu.py']
state={'status':'RUNNING','pid':os.getpid(),'stages':[],'accepted':False}
def save():(root/'CPU_PREFLIGHT_RUN.json').write_text(json.dumps(state,indent=2)+'\n')
def stop(*_):raise KeyboardInterrupt('preflight interrupted')
for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
save()
try:
 for name in ['validate_runtime.py','verify_source_fixture.py','cpu_probe.py','dispatch_probe.py','allocator_cpu_check.py','selective_cache_cpu_check.py','diffusion_cpu_check.py','joint_cpu_check.py','validate_configs.py','attention_check_dispatch.py','attention_check_cpu.py','pipeline_check.py','hybrid_cpu_check.py','check_build_protocol.py','check_native_build_audit.py','combined_cache_cpu_check.py','check_policy_metadata.py','check_bucket_view_guard.py','check_affine_build_audit.py','check_combined_integration_cpu.py','check_prefix_watch.py','check_dual_cold.py','pair_transpose_cpu/check_cpu.py','check_verified_msa_cpu.py','check_queue_plans.py','check_padding_composition_cpu.py','check_head_gate_cpu.py','check_head_gate_backward_cpu.py','check_joint_selective_cpu.py','check_cpu_placement.py','check_affine16_cpu.py','check_silu_cpu.py']:
  nonce=secrets.token_hex(32);OwnedProcesses.supported();owner=OwnedProcesses(nonce,root/(name+'.cleanup.json'));proc=None
  row={'name':name,'nonce':nonce,'status':'RUNNING'};state['stages'].append(row);save();start=time.monotonic()
  try:
   env={**os.environ,OWNERSHIP_ENV:nonce,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
   with (root/(name+'.log')).open('x') as log:
    proc=subprocess.Popen([str(root/'assets/env/bin/python'),str(root/name)],env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    row['pid']=proc.pid;save();assert proc.wait(timeout=1800 if name=='check_msa_retention_cpu.py' else 600)==0,name
   row['status']='COMPLETED'
  finally:
   row['cleanup']=owner.cleanup(proc,'Round13 CPU preparation')['status'];row['seconds']=time.monotonic()-start;save()
 subprocess.run([sys.executable,str(root/'freeze_bindings.py')],check=True)
 state['status']='COMPLETED';state['bindings_sha256']=hashlib.sha256((root/'bindings.json').read_bytes()).hexdigest()
except BaseException as error:state.update(status='FAILED',error=repr(error));save();raise
save()
