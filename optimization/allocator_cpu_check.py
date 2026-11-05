"""Exercise actual native train_step with deterministic small CPU model and Adam."""
import os,sys,json,copy
from pathlib import Path
from types import SimpleNamespace
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['LAYERNORM_TYPE']='torch'
sys.path.insert(0,str(Path(__file__).resolve().parent/'reference'))
import torch
from runner.train import AF3Trainer
from retain_allocator import install
torch.set_num_threads(2)
original=AF3Trainer.train_step
class Metrics:
 def __init__(self):self.values=[]
 def add(self,k,v,namespace):self.values.append((k,float(v.detach()),namespace))
class Tiny:
 train_step=original
 def __init__(self,accum):
  self.model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Dropout(.25),torch.nn.Linear(8,2))
  self.optimizer=torch.optim.Adam(self.model.parameters(),lr=.001)
  self.lr_scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,2,.9)
  self.configs=SimpleNamespace(dtype='fp32');self.iters_to_accumulate=accum
  self.train_metric_wrapper=Metrics();self.global_step=0;self.step=0
 def model_forward(self,batch,mode):return self.model(batch),None
 def get_loss(self,batch,mode):
  loss=batch.square().mean();return loss,{'loss':loss,'extra':loss+1},None
 def update(self):torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
 def print(self,x):raise AssertionError(x)
def equal(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
cases=[]
for accum in [1,2]:
 torch.manual_seed(99);a=Tiny(accum)
 torch.manual_seed(99);b=Tiny(accum);data=torch.randn(3,4)
 torch.manual_seed(123)
 for i in range(4):a.global_step=i;a.step=i;a.train_step(data)
 expected_rng=torch.get_rng_state().clone()
 Tiny.train_step=original;stats=install(Tiny)
 torch.manual_seed(123)
 for i in range(4):b.global_step=i;b.step=i;b.train_step(data)
 assert equal(a.model.state_dict(),b.model.state_dict())
 assert equal(a.optimizer.state_dict(),b.optimizer.state_dict())
 assert equal(a.lr_scheduler.state_dict(),b.lr_scheduler.state_dict())
 assert equal(a.train_metric_wrapper.values,b.train_metric_wrapper.values)
 assert torch.equal(expected_rng,torch.get_rng_state())
 assert stats['skipped_empty_cache']==4
 cases.append({'accumulation':accum,'steps':4,'parameters_optimizer_scheduler_metrics_rng_exact':True})
 Tiny.train_step=original
assert not torch.cuda.is_initialized()
report={'status':'PASS','cases':cases,'gpu_initialized':False,'source_policy':stats,'accepted':False,'scope':'Actual native train_step CPU control-flow equivalence; no GPU allocator performance claim.'}
Path(__file__).with_name('ALLOCATOR_CPU_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
