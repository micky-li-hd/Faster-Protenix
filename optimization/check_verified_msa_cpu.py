"""Reuse completed exact CPU proof only for byte-identical tested implementations."""
from pathlib import Path
import hashlib,json,itertools
r=Path(__file__).resolve().parent;p=r/'operator_evidence/msa_round22_cpu'
read=lambda x:json.loads(x.read_text())
manifest=read(p/'bindings.json');run=read(p/'CPU_PREFLIGHT_RUN.json');closure=read(p/'FAILURE_CLOSURE.json')
h=hashlib.sha256((p/'bindings.json').read_bytes()).hexdigest()
assert run['status']=='COMPLETED' and run['bindings_sha256']==closure['bindings_sha256']==h
assert closure['status']=='FAILED_QUEUE_CLEAN_ALL_HANDLES_GONE_8GPU_IDLE'
stage=[x for x in run['stages'] if x['name']=='check_msa_retention_cpu.py'];assert len(stage)==1 and stage[0]['status']=='COMPLETED' and stage[0]['cleanup']=='CLEAN'
for n in ['check_msa_retention_cpu.py','partial_pair_checkpoint.py','partial_msa_checkpoint.py']:
 assert hashlib.sha256((r/n).read_bytes()).hexdigest()==manifest[n],n
for n,h in manifest.items():
 if n.startswith('reference/'):
  assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
proof=read(p/'MSA_CPU_CHECK.json');assert hashlib.sha256((p/'MSA_CPU_CHECK.json').read_bytes()).hexdigest()==manifest['MSA_CPU_CHECK.json']
cases=proof['cases'];assert proof['status']=='PASS_REAL_MSA_PAIR_CPU_RETENTION' and not proof['cuda_initialized'] and len(cases)==12
assert {(c['pair_prefix'],c['msa_prefix'],c['order']) for c in cases}==set(itertools.product([0,4],[0,2,4],['msa_then_pair','pair_then_msa']))
assert all(c['all_exact'] and c['rng_exact'] and c['no_grad_bypass'] and c['tensor_count']==2969 for c in cases)
(r/'MSA_CPU_CHECK.json').write_bytes((p/'MSA_CPU_CHECK.json').read_bytes())
report={'status':'PASS_REUSED_CONTENT_IDENTICAL_CPU_PROOF','source_campaign':'protenix_8h20_20261102_round22','cases':12,'tensors_per_case':2969,'executed_again':False,'accepted':False}
(r/'MSA_CPU_REUSE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
