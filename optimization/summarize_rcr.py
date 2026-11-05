"""Eight-H20 R-C-R summary enforcing the frozen contract and actual input checks."""
import argparse,json,statistics,copy
from pathlib import Path
import yaml
from compare_traces import load,timing,compare
from compare_inputs import compare_inputs
from extract_eval import compare as eval_compare
root=Path('/root/tianbali/protenix_8h20_20261103_round31')
def read(tag):
 folder=root/'runs'/tag
 audit=json.loads((folder/'audit.json').read_text())
 assert audit['status']=='PASS' and audit['ranks']==8 and audit['updates']==128
 assert len(audit['checkpoints'])==2 and all(x['finite'] and x['step']==127 for x in audit['checkpoints'])
 state=json.loads((root/'plans'/tag/'STATE.json').read_text());assert state['status']=='COMPLETED' and state['runs'][0]['cleanup']=='CLEAN'
 assert json.loads((folder/'cleanup.json').read_text())['status']=='CLEAN'
 assert state['runs'][0]['private_shm_cleanup']['status']=='CLEAN'
 for rank in range(8):
  policy=json.loads((folder/f'rank{rank}.policy.json').read_text());stats=json.loads((folder/f'rank{rank}.diffusion_prefix_stats.json').read_text())
  from guarded_silu import verify_stats as verify_silu
  verify_silu(json.loads((folder/f'rank{rank}.silu_stats.json').read_text()),policy['silu_recompute'])
  from affine16_audit import verify_stats as verify_affine16
  verify_affine16(json.loads((folder/f'rank{rank}.affine16_stats.json').read_text()),policy['affine16'])
  from cpu_placement_audit import verify_receipt
  verify_receipt(root,json.loads((folder/f'rank{rank}.cpu_placement.json').read_text()),policy['cpu_placement'],rank)
  from diffusion_selective_audit import verify_stats
  verify_stats(json.loads((folder/f'rank{rank}.diffusion_selective_stats.json').read_text()),policy['diffusion_selective_policy'],128)
  assert stats['retained_prefix']==policy['retained_diffusion_prefix']
  assert stats['retained_block_calls']==stats['matched_gradient_calls']*stats['retained_prefix']
  assert stats['retained_prefix']==0 or stats['matched_gradient_calls']>=128
 from native_build_audit import verify as verify_native_build
 hybrid_requested=json.loads((folder/'rank0.policy.json').read_text())['hybrid_layernorm']
 assert hybrid_requested==state['runs'][0]['requested_native_compile']
 verify_native_build(root,folder,hybrid_requested,state['runs'][0]['nonce'])
 from affine_build_audit import verify as verify_affine_build
 verify_affine_build(root,folder,json.loads((folder/'rank0.policy.json').read_text())['mainloop_affine'],state['runs'][0]['nonce'])
 traces=load(folder)
 for r in range(8):assert (folder/f'rank{r}.complete').is_file()
 initial=[json.loads((folder/f'rank{r}.initial.json').read_text()) for r in range(8)]
 configs=list((folder/'native').glob('*/config.yaml'));assert len(configs)==1
 config=yaml.safe_load(configs[0].read_text());config.pop('base_dir')
 ev=json.loads((folder/'eval.json').read_text());assert ev['step']==127
 times=timing(traces);times['whole_process_seconds']=state['runs'][0]['process_seconds']
 times['input_audit_seconds_sum_maxrank']=sum(max(traces[r][s]['input_audit_seconds'] for r in range(8)) for s in range(128))
 policy_metadata=[json.loads((folder/f'rank{r}.policy.json').read_text()) for r in range(8)]
 assert all(x==policy_metadata[0] for x in policy_metadata)
 times['requested_native_compile']=hybrid_requested
 times['requested_torch_compile']=policy_metadata[0]['compile_layernorm'] or policy_metadata[0]['compile_attention']
 assert times['requested_torch_compile']==state['runs'][0]['requested_torch_compile']
 assert times['requested_native_compile']==state['runs'][0]['requested_native_compile']
 times['full_run_peak_allocated_bytes']=max(x['peak_allocated_bytes'] for x in audit['full_run_memory'])
 times['full_run_peak_reserved_bytes']=max(x['peak_reserved_bytes'] for x in audit['full_run_memory'])
 times['gpu_samples']=audit['gpu_samples']
 return {'traces':traces,'initial':initial,'config':config,'policy_metadata':policy_metadata[0],'eval':ev,'timing':times,'bindings':state['bindings_sha256']}
p=argparse.ArgumentParser();p.add_argument('r1');p.add_argument('candidate');p.add_argument('r2');p.add_argument('--output',required=True);p.add_argument('--candidate-policy',required=True);a=p.parse_args()
arms=[read(t) for t in [a.r1,a.candidate,a.r2]]
assert arms[0]['config']==arms[2]['config'],'Reference config changed'
policies=json.loads((root/'performance_policies.json').read_text())
assert a.candidate_policy in policies and a.candidate_policy!='reference'
for arm,name in zip(arms,['reference',a.candidate_policy,'reference']):
 declared=dict(policies[name]);silu_recompute=declared.pop('_silu_recompute',False);affine16=declared.pop('_affine16',False);cpu_placement=declared.pop('_cpu_placement',None);diffusion_selective_policy=declared.pop('_diffusion_selective_policy',None);bucket_view=declared.pop('_ddp_gradient_bucket_view',False);global_cache=declared.pop('_global_sampling_projection_cache',False);local_cache=declared.pop('_local_conditioning_cache',False);hybrid=declared.pop('_hybrid_layernorm',False);affine=declared.pop('_mainloop_affine',False);paircopy=declared.pop('_pair_transpose',False);attention=declared.pop('_compile_attention',False);diffusion_prefix=declared.pop('_retained_diffusion_prefix',0);prefix=declared.pop('_retained_pair_prefix',0);msa_prefix=declared.pop('_retained_msa_prefix',0);trim_msa_padding=declared.pop('_trim_msa_padding',False);head_gate=declared.pop('_head_gate',False);head_gate_backward=declared.pop('_head_gate_backward',False);compiled=declared.pop('_compile_layernorm',False);retained=declared.pop('_retain_allocator',False);cache_bytes=declared.pop('_selective_pair_cache_bytes',0)
 assert arm['policy_metadata']=={'policy':name,'silu_recompute':silu_recompute,'affine16':affine16,'cpu_placement':cpu_placement,'ddp_gradient_bucket_view':bucket_view,'global_sampling_projection_cache':global_cache,'local_conditioning_cache':local_cache,'hybrid_layernorm':hybrid,'mainloop_affine':affine,'pair_transpose':paircopy,'retained_pair_prefix':prefix,'retained_msa_prefix':msa_prefix,'trim_msa_padding':trim_msa_padding,'head_gate':head_gate,'head_gate_backward':head_gate_backward,'retained_diffusion_prefix':diffusion_prefix,'diffusion_selective_policy':diffusion_selective_policy,'compile_layernorm':compiled,'compile_attention':attention,'retain_allocator':retained,'selective_cache_bytes':cache_bytes,'config_overrides':declared}
expected=copy.deepcopy(arms[0]['config'])
for key,value in policies[a.candidate_policy].items():
 if key.startswith('_'):continue
 parts=key.split('.');node=expected
 for part in parts[:-1]:node=node[part]
 node[parts[-1]]=value
 if key=='blocks_per_ckpt':
  for module in ['template_embedder','msa_module','pairformer','diffusion_module','confidence_head']:
   expected['model'][module]['blocks_per_ckpt']=value
assert arms[1]['config']==expected,'Candidate configuration differs from declared performance policy' 
assert arms[0]['bindings']==arms[1]['bindings']==arms[2]['bindings'],'Different source/data/observer bindings'
assert arms[0]['initial']==arms[1]['initial']==arms[2]['initial'],'Different initial model/optimizer'
inputs=compare_inputs(*(x['traces'] for x in arms))
comparisons={}
for name,i,j in [('reference_repeat',0,2),('candidate_vs_r1',0,1),('candidate_vs_r2',2,1)]:
 comparisons[name]={'trace':compare(arms[i]['traces'],arms[j]['traces']),'evaluation':eval_compare(arms[i]['eval'],arms[j]['eval'])}
passed=inputs['input_rng_exact'] and all(x['trace']['trace_screen_passed'] and x['evaluation']['evaluation_screen_passed'] for x in comparisons.values())
ratios={}
for metric in ['mean_max_rank_seconds_updates16_127','whole_process_seconds']:
 values=[x['timing'][metric] for x in arms]
 ratios[metric]={'against_r1':values[0]/values[1],'against_r2':values[2]/values[1],'mean_references_over_candidate':statistics.mean([values[0],values[2]])/values[1]}
result={'candidate_policy':a.candidate_policy,'short_run_contract_passed':passed,'input_rng':inputs,'comparisons':comparisons,'ratios':ratios,'timings':[x['timing'] for x in arms],'scope':'Eight-H20 public fixture128updates inside2000step warmup; not convergence or full-corpus quality; instrumented throughput only','target_2x_reached_in_both_metrics':passed and all(x['mean_references_over_candidate']>=2 for x in ratios.values())}
with open(a.output,'x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps({k:result[k] for k in ['short_run_contract_passed','ratios','target_2x_reached_in_both_metrics']}))
