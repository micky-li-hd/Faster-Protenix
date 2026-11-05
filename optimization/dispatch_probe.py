"""CPU dispatch boundaries and AOT forward/backward checks; no CUDA initialization."""
import os,sys,json,copy
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['LAYERNORM_TYPE']='torch'
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'reference'))
import torch
from protenix.model.triangular.layers import OpenFoldLayerNorm
from compiled_layernorm import install,eligible
torch.set_num_threads(2)
for shape,want in [((384,128),False),((48,384,128),False),((147455,128),False),((147456,128),True),((384,384,128),True),((384,384,384),False)]:assert eligible(shape)==want
original=OpenFoldLayerNorm.forward;stats=install(OpenFoldLayerNorm,torch,'aot_eager');cases=[]
for shape in [(384,128),(384,384,128)]:
 for dtype in [torch.float32,torch.bfloat16]:
  torch.manual_seed(20261027)
  model=OpenFoldLayerNorm(128).to(dtype=dtype)
  data=torch.randn(shape,dtype=dtype);grad=torch.randn_like(data)
  rng=torch.get_rng_state().clone();state=copy.deepcopy(model.state_dict())
  a=data.clone().requires_grad_();y=original(model,a);y.backward(grad)
  expected=[y.detach(),a.grad.clone()]+[p.grad.clone() for p in model.parameters()]
  model.zero_grad(set_to_none=True)
  b=data.clone().requires_grad_();z=model(b);z.backward(grad)
  actual=[z.detach(),b.grad]+[p.grad for p in model.parameters()]
  assert all(torch.equal(x,y) for x,y in zip(expected,actual))
  assert torch.equal(rng,torch.get_rng_state())
  assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
  cases.append({'shape':shape,'dtype':str(dtype),'eligible':eligible(shape),'forward_input_parameter_grads_exact':True,'rng_exact':True})
assert stats['eligible_calls']==2 and stats['bypassed_calls']==2 and stats['graphs']
assert not torch.cuda.is_initialized()
(root/'DISPATCH_PROBE.json').write_text(json.dumps({'status':'PASS','cases':cases,'stats':stats,'gpu_initialized':False,'scope':'CPU AOT and dispatch only; operator CUDA evidence separate; native8/128 RCR mandatory'},indent=2)+'\n')
print('DISPATCH_PROBE PASS')
