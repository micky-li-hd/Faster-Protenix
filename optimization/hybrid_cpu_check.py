"""Actual native LayerNorm CPU fallback, dtype, parameter identity and RNG proof."""
import os,sys,json,hashlib,itertools
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['LAYERNORM_TYPE']='torch'
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'reference'))
source=root/'reference/protenix/model/triangular/layers.py';expected=json.loads((root/'expected_source_fixture.json').read_text())['source']['protenix/model/triangular/layers.py']['sha256'];assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
import torch
from protenix.model.triangular.layers import OpenFoldLayerNorm
from guarded_layernorm import install
assert torch.__version__.split('+')[0]=='2.7.1';torch.set_num_threads(2)
original=OpenFoldLayerNorm.forward;stats=install(OpenFoldLayerNorm);results=[]
try:
 for index,(scale,offset,shape,stride,dtype,frozen) in enumerate(itertools.product([True,False],[True,False],[(16,64),(4,4,64),(16,128),(4,4,128)],[1,2],[torch.float32,torch.bfloat16],[False,True])):
  torch.manual_seed(20261101+index);model=OpenFoldLayerNorm(shape[-1],create_scale=scale,create_offset=offset).float();ids=[id(p) for p in model.parameters()];state={n:v.clone() for n,v in model.state_dict().items()}
  if frozen:
   for p in model.parameters():p.requires_grad_(False)
  raw=torch.randn(*shape[:-1],shape[-1]*stride).to(dtype);x=raw[...,::stride];assert x.shape==shape
  def run(fn):
   model.zero_grad(set_to_none=True);a=x.detach().requires_grad_(True);before=torch.get_rng_state().clone();out=fn(model,a);out.float().square().mean().backward();assert torch.equal(before,torch.get_rng_state());return [out.detach(),a.grad,*[p.grad for p in model.parameters() if p.requires_grad]]
  a=run(original);b=run(OpenFoldLayerNorm.forward);assert len(a)==len(b) and all(torch.equal(v,w) for v,w in zip(a,b));assert all(torch.isfinite(v).all() for v in b)
  assert [id(p) for p in model.parameters()]==ids and model.state_dict().keys()==state.keys() and all(torch.equal(state[n],v) for n,v in model.state_dict().items())
  results.append({'scale':scale,'offset':offset,'shape':shape,'stride':stride,'dtype':str(dtype),'affine_frozen':frozen,'tensor_count':len(b),'exact':True})
finally:OpenFoldLayerNorm.forward=original
assert not torch.cuda.is_initialized() and stats['eligible_calls']==0 and stats['bypassed_calls']==len(results)
report={'status':'PASS_NATIVE_CPU_HYBRID_FALLBACK','cases':results,'stats':stats,'gpu_initialized':False,'accepted':False,'scope':'CPU fallback/parameter/RNG proof; actual fused CUDA path covered by separate192-case operator proof, native full training still required.'};(root/'HYBRID_CPU_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'cases':len(results),'stats':stats}))
