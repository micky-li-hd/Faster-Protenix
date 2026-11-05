import ast,json
from pathlib import Path
r=Path(__file__).resolve().parent;tree=ast.parse((r/'tuning_queue.py').read_text());declared=json.loads((r/'performance_policies.json').read_text())
policies=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='policies' for t in n.targets))
assert len(policies)==4 and len(set(policies))==4 and set(policies)==set(declared)-{'reference'}
for policy in policies:
 for prefix,steps in [('smoke_',8),('candidate128_',128)]:
  p=r/'plans'/(prefix+policy)/'PLAN.json';plan=json.loads(p.read_text());assert plan['gpus']==list(range(8));job,=plan['jobs']
  assert job['tag']==prefix+policy and job['env']['RSI_POLICY']==policy and job['env']['RSI_STEPS']==str(steps)
  assert job['out']==str(r/'runs'/(prefix+policy)) and job['cwd']==str(r)
report={'status':'PASS_ACTUAL_QUEUE_PLAN_BINDINGS','policies':policies,'plans':8,'accepted':False}
(r/'QUEUE_PLAN_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
