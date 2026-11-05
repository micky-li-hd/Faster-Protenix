"""Freeze source, fixture, observer and contract before training; no GPU imports."""
import hashlib,json
from pathlib import Path
root=Path('/root/tianbali/protenix_8h20_20261103_round31')
assert not (root/'bindings.json').exists(), 'Existing bindings cannot be silently replaced'
files=[p for p in (root/'reference').rglob('*.py') if '__pycache__' not in p.parts and not p.name.startswith('._')]
files += [p for p in (root/'assets/data/researchboost_fixture').rglob('*') if p.is_file() and not p.name.startswith('._')]
files += [root/name for name in ['recipe.json','recipe.py','native_entry.py','input_fingerprint.py','SHORT_COMPARISON.json','runtime_inventory.json','bounded.py','ownership.py','compare_traces.py','compare_inputs.py','extract_eval.py','audit_arm.py','summarize_rcr.py','runtime_env.sh','performance_policies.json','MEMORY_PLAN.json','partial_pair_checkpoint.py','cpu_probe.py','validate_configs.py','CPU_PROBE.json','compiled_layernorm.py','dispatch_probe.py','DISPATCH_PROBE.json','retain_allocator.py','allocator_cpu_check.py','ALLOCATOR_CPU_CHECK.json','LOCAL_PATCH_GUARD_CHECK.json','PREPARATION_BINDINGS.json','CONFIG_VALIDATION.json','smoke_screen.py','tuning_queue.py','rcr_queue.py','selective_pair_cache.py','selective_cache_cpu_check.py','SELECTIVE_CACHE_CPU_CHECK.json']]
files += [root/'JOINT_SOURCE_BINDINGS.json',root/'JOINT_CPU_CHECK.json',root/'DIFFUSION_CPU_CHECK.json',root/'isolated_shm.py',root/'source_fixture_check.json']
files += list(root.glob('*.py'))
files += [root/n for n in ['CPU_CHECK.json','DISPATCH_CHECK.json','PIPELINE_CHECK.json']]
files += [p for p in (root/'operator_evidence').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += list((root/'plans').glob('*/PLAN.json'))
files += [p for p in (root/'candidate_source').iterdir() if p.is_file()]
files += [root/n for n in ['BUILD_MANIFEST.json','HYBRID_CPU_CHECK.json','BUILD_PROTOCOL_CHECK.json','NATIVE_BUILD_AUDIT_CHECK.json']]
files += [root/n for n in ['expected_source_fixture.json','COMBINED_CACHE_CPU_CHECK.json','POLICY_METADATA_CHECK.json','GUARD_CHECK.json','LOCAL_INTEGRATION_REVIEW.json']]
files += [p for directory in ['affine','combined_cpu'] for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += [root/'analysis'/n for n in ['queue_readiness.py','watch_live_prefixes.py','run_prefix_watch.py','check_queue_readiness.py','PREFIX_WATCH_BINDINGS.json','QUEUE_READINESS_CHECK.json']]
files += [root/n for n in ['AFFINE_BUILD_AUDIT_CHECK.json','COMBINED_ADAPTER_CHECK.json','PREFIX_WATCH_CHECK.json','DUAL_COLD_CHECK.json']]
files += [p for p in (root/'pair_transpose_cpu').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += [root/'PAIR_TRANSPOSE_INTEGRATION.json']
files += [root/'MSA_CPU_CHECK.json',root/'MSA_INTEGRATION.json',root/'MSA_CPU_REUSE_CHECK.json',root/'QUEUE_PLAN_CHECK.json']
files += [root/'MSA_PADDING_COMPOSITION_CHECK.json',root/'MSA_PADDING_INTEGRATION.json']
assert json.loads((root/'HEAD_GATE_CPU_CHECK.json').read_text())['status']=='HEAD_GATE_NATIVE_CPU_INTEGRATION_PASS'
files += [root/'HEAD_GATE_CPU_CHECK.json',root/'HEAD_GATE_INTEGRATION.json']
assert json.loads((root/'HEAD_GATE_BACKWARD_CPU_CHECK.json').read_text())['status']=='HEAD_GATE_NATIVE_CPU_INTEGRATION_PASS'
files += [root/'HEAD_GATE_BACKWARD_CPU_CHECK.json',root/'HEAD_GATE_BACKWARD_INTEGRATION.json']
files += [root/'JOINT_SELECTIVE_CHECK.json',root/'CPU_PLACEMENT_PLAN.json',root/'CPU_PLACEMENT_CHECK.json']
files += [root/'AFFINE16_CPU_CHECK.json']
files += [root/'SILU_CPU_CHECK.json']
result={}
for p in sorted(files):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 result[str(p.relative_to(root))]=h.hexdigest()
with (root/'bindings.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({'status':'FROZEN','files':len(result)}))
