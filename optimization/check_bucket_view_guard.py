import json,types,hashlib,ast
from pathlib import Path
from bucket_view import install
r=Path(__file__).resolve().parent;calls=[]
def factory(*args,**kwargs):calls.append((args,kwargs));return object()
m=types.SimpleNamespace(DDP=factory);restore,stats=install(m);args=(object(),);kwargs={'static_graph':True,'find_unused_parameters':False,'device_ids':[3],'output_device':3,'bucket_cap_mb':25,'broadcast_buffers':True}
m.DDP(*args,**kwargs);assert calls==[(args,{**kwargs,'gradient_as_bucket_view':True})] and stats['instances']==1
for bad in ({'static_graph':False},{'static_graph':True,'find_unused_parameters':True},{'static_graph':True,'gradient_as_bucket_view':False}):
 try:m.DDP(*args,**bad)
 except AssertionError:pass
 else:raise AssertionError('Invalid configuration accepted')
restore();assert m.DDP is factory and len(calls)==1
source=r/'reference/runner/train.py';tree=ast.parse(source.read_text());trainer=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='AF3Trainer');calls=[x for x in ast.walk(trainer) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='DDP'];assert len(calls)==1
keys={k.arg:ast.unparse(k.value) for k in calls[0].keywords};assert keys=={'find_unused_parameters':'self.configs.find_unused_parameters','device_ids':'[DIST_WRAPPER.local_rank]','output_device':'DIST_WRAPPER.local_rank','static_graph':'True'}
assert 'cache_enabled=False' in source.read_text()
d={'status':'PASS_DDP_VIEW_ONLY_GUARD','only_changed_constructor_argument':'gradient_as_bucket_view=True','native_constructor':keys,'native_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((r/'bucket_view.py').read_bytes()).hexdigest(),'invalid_configs_rejected':3,'accepted':False};(r/'GUARD_CHECK.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
