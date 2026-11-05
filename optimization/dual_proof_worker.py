"""Hidden-device proof of actual dual installer and CPU fallback."""
import os,sys,json
from pathlib import Path
os.environ['LAYERNORM_TYPE']='torch'
import torch
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'reference'))
from protenix.model.triangular.layers import OpenFoldLayerNorm
from dual_cold_install import install
out=Path(sys.argv[1]);rank=int(sys.argv[2]);torch.set_num_threads(2)
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
model=OpenFoldLayerNorm(64);x=torch.randn(7,64,requires_grad=True)
params=[x,*model.parameters()];ids=[id(p) for p in model.parameters()];keys=list(model.state_dict());before_y=model(x);before_g=torch.autograd.grad(before_y,params,torch.ones_like(before_y))
rng=torch.get_rng_state().clone();env=dict(os.environ)
forward,affine=install(out,OpenFoldLayerNorm)
y=model(x);g=torch.autograd.grad(y,params,torch.ones_like(y))
assert torch.equal(y,before_y) and all(torch.equal(a,b) for a,b in zip(g,before_g))
assert ids==[id(p) for p in model.parameters()] and keys==list(model.state_dict())
assert torch.equal(rng,torch.get_rng_state()) and env==dict(os.environ) and not torch.cuda.is_initialized()
assert forward['calls']['bypassed_calls']>0 and forward['calls']['eligible_calls']==0
assert affine['calls']=={'eligible_backward':0,'native_backward_fallback':0,'eligible_width64':0,'eligible_width128':0}
(out/f'rank{rank}.json').write_text(json.dumps({'rank':rank,'status':'DUAL_CPU_IMPORT_FALLBACK_PASS','forward':forward,'affine':affine,'rng_exact':True,'environment_unchanged':True,'parameter_identity_preserved':True,'cuda_initialized':False},indent=2)+'\n')
