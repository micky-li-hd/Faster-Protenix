"""Execute actual metadata producer expression and summary assertions on all policies."""
import ast,json,copy,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent;native=ast.parse((r/'native_entry.py').read_text());summary=ast.parse((r/'summarize_rcr.py').read_text())
policies=json.loads((r/'performance_policies.json').read_text())
start=next(i for i,n in enumerate(native.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='policy_config')
end=next(i for i,n in enumerate(native.body) if isinstance(n,ast.Expr) and ast.unparse(n.value).startswith("spec['overrides'].update(policy_config)"))
producer=next(n for n in ast.walk(native) if isinstance(n,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=='config_overrides' for k in n.keys))
loop=next(n for n in summary.body if isinstance(n,ast.For) and ast.unparse(n.target)=='(arm, name)')
program=compile(ast.Module(body=loop.body,type_ignores=[]),'<actual-summary-policy-check>','exec')
rows=[];rejected=0
for name in policies:
 ns={'policies':policies,'policy':name};exec(compile(ast.Module(body=native.body[start:end],type_ignores=[]),'<actual-native-policy-producer>','exec'),ns)
 metadata=eval(compile(ast.Expression(body=producer),'<actual-metadata-dict>','eval'),ns)
 for rank in range(8):
  exec(program,{'policies':policies,'name':name,'arm':{'policy_metadata':metadata}})
  for key in ('global_sampling_projection_cache','local_conditioning_cache','ddp_gradient_bucket_view','mainloop_affine','pair_transpose','retained_msa_prefix','trim_msa_padding','head_gate','head_gate_backward','diffusion_selective_policy','cpu_placement','affine16','silu_recompute'):
   bad=copy.deepcopy(metadata);bad[key]=(int(bad[key])+1 if key=='retained_msa_prefix' else not bad[key])
   try:exec(program,{'policies':policies,'name':name,'arm':{'policy_metadata':bad}})
   except AssertionError:rejected+=1
   else:raise AssertionError('Wrong flag accepted')
 rows.append({'policy':name,'metadata':metadata,'all8_passed':True})
report={'status':'PASS_ACTUAL_POLICY_SCHEMA','valid_records':len(policies)*8,'wrong_flags_rejected':rejected,'rows':rows,'source_hashes':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ('native_entry.py','summarize_rcr.py')},'accepted':False}
(r/'POLICY_METADATA_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
