import os,sys,json,copy,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent
assert os.environ['CUDA_VISIBLE_DEVICES']==''
os.environ['LAYERNORM_TYPE']='torch'
sys.path.insert(0,os.environ.get('RSI_REFERENCE',str(r/'reference')))
import torch
import torch.utils.checkpoint as cp
from protenix.model.modules.transformer import ConditionedTransitionBlock
from guarded_silu import install,verify_stats
torch.set_num_threads(2);torch.manual_seed(20261027)
original=ConditionedTransitionBlock.forward
m=ConditionedTransitionBlock(768,384).train()
with torch.no_grad():
 for p in m.parameters():
  if p.ndim>=2 and not bool(torch.count_nonzero(p)):p.normal_(0,.005)
clone=copy.deepcopy(m)
with torch.no_grad():
 for p in clone.parameters():p.add_(.0001)
a=torch.randn(2,3,768,requires_grad=True);s=torch.randn(2,3,384,requires_grad=True);up=torch.randn_like(a)
ids=[id(p) for p in m.parameters()];rows=[]
def run(model,checkpoint):
 y=cp.checkpoint(model,a,s,use_reentrant=False) if checkpoint else model(a,s)
 return (y.detach(),*[g.detach() for g in torch.autograd.grad(y,(a,s,*model.parameters()),up)])
for checkpoint in [False,True]:
 for label,model in [('model',m),('ema_copy_updated',clone)]:
  state=torch.get_rng_state();expected=run(model,checkpoint)
  restore,stats=install(ConditionedTransitionBlock)
  try:
   actual=run(model,checkpoint)
   assert all(torch.isfinite(x).all() and torch.equal(x,y) for x,y in zip(expected,actual))
   assert torch.equal(state,torch.get_rng_state())
  finally:restore()
  verify_stats(stats,True);assert stats['eligible']==(2 if checkpoint else 1)
  rows.append({'checkpoint':checkpoint,'model':label,'tensor_checks':len(actual),'bit_exact':True})
# Copy during installation: the class wrapper binds to copied parameters.
restore,stats=install(ConditionedTransitionBlock)
try:
 copied=copy.deepcopy(m)
 with torch.no_grad():
  for p in copied.parameters():p.add_(.002)
 y=copied(a,s)
 assert torch.equal(y,original(copied,a,s)) and not torch.equal(y,original(m,a,s))
 before=stats['eligible']
 with torch.no_grad():assert torch.equal(m(a,s),original(m,a,s))
 m.eval();assert torch.equal(m(a,s),original(m,a,s));m.train()
 small=ConditionedTransitionBlock(16,8);x=torch.randn(2,3,16);t=torch.randn(2,3,8)
 assert torch.equal(small(x,t),original(small,x,t))
 double=copy.deepcopy(m).double();assert torch.equal(double(a.double(),s.double()),original(double,a.double(),s.double()))
 assert stats['eligible']==before and stats['fallback']==4
finally:restore()
verify_stats(stats,True)
assert ConditionedTransitionBlock.forward is original and ids==[id(p) for p in m.parameters()]
before=copy.deepcopy(stats);assert torch.equal(copied(a,s),original(copied,a,s)) and stats==before
verify_stats({'enabled':False},False);rejected=0
for key,value in [('enabled',False),('eligible',0),('eligible',True),('fallback',-1),('restored',False),('shapes',{})]:
 invalid={**stats,key:value}
 try:verify_stats(invalid,True)
 except AssertionError:rejected+=1
 else:raise AssertionError(key)
assert rejected==6 and not torch.cuda.is_initialized()
report=dict(status='PASS_GUARDED_SILU_CPU_EMA_FALLBACK_RESTORE',accepted=False,cases=rows,malformed_rejected=rejected,cuda_initialized=False,
 adapter_sha256=hashlib.sha256((r/'guarded_silu.py').read_bytes()).hexdigest())
(r/'SILU_CPU_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
