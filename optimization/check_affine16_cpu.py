import os,sys,json,copy,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent;sys.path.insert(0,str(r/'reference'));os.environ['LAYERNORM_TYPE']='torch'
import torch
from protenix.model.triangular.layers import OpenFoldLayerNorm
from guarded_affine16 import install
from affine16_audit import verify_sources,verify_stats
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
source=verify_sources(r);original=OpenFoldLayerNorm.forward;original_init=OpenFoldLayerNorm.__init__;calls=[]
def parent(self,x):calls.append(id(self));return original(self,x)
OpenFoldLayerNorm.forward=parent
restore,stats=install(OpenFoldLayerNorm,object())
modules=[OpenFoldLayerNorm(w) for w in [16,64,128,384]]
assert stats['installed']==1 and 'forward' in modules[0].__dict__ and all('forward' not in x.__dict__ for x in modules[1:])
x=torch.randn(4,16);model=modules[0];clone=copy.deepcopy(model)
with torch.no_grad():clone.weight.mul_(2);clone.bias.add_(.75)
y=clone(x);expected=original(clone,x);assert torch.equal(y,expected) and not torch.equal(y,original(model,x)) and calls[-1]==id(clone)
assert torch.equal(clone.state_dict()['weight'],clone.weight)
ids=[[id(p) for p in m.parameters()] for m in modules];restore();assert OpenFoldLayerNorm.__init__ is original_init and all('forward' not in m.__dict__ for m in modules)
assert ids==[[id(p) for p in m.parameters()] for m in modules]
# An existing deepcopy retains its wrapper, but must still use its own parameters.
before=dict(stats)
assert torch.equal(clone(x),original(clone,x)) and stats==before
OpenFoldLayerNorm.forward=original
verify_stats({'enabled':False},False)
valid={'enabled':True,'restored':True,'eligible':1,'fallback':0,'installed':1};verify_stats(valid,True)
rejected=0
for key,value in [('eligible',0),('installed',0),('restored',False),('enabled',False),('fallback',-1)]:
 bad={**valid,key:value}
 try:verify_stats(bad,True)
 except AssertionError:rejected+=1
 else:raise AssertionError(key)
assert rejected==5 and not torch.cuda.is_initialized()
report={'status':'PASS_AFFINE16_CPU_COMPOSITION_AND_EMA_COPY','accepted':False,'source':source,'deepcopy_uses_own_parameters':True,'parent_fallback_preserved':True,'other_widths_unwrapped':True,'parameter_ids_preserved':True,'restored':True,'malformed_receipts_rejected':rejected,'cuda_initialized':False,'adapter_sha256':hashlib.sha256((r/'guarded_affine16.py').read_bytes()).hexdigest()}
(r/'AFFINE16_CPU_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
