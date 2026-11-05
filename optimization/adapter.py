import torch,sys,json,hashlib,os,importlib.util
from pathlib import Path
r=Path(__file__).resolve().parent
sys.path.insert(0,str(r/'cold_source'))
from simt_cold_build import ensure_binary
binary,receipt,timing=ensure_binary(Path(os.environ['RSI_OUT']))
spec=importlib.util.spec_from_file_location('rsi_simt_backward_v1',binary)
extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension)
with (Path(os.environ['RSI_OUT'])/('rank'+os.environ['RANK']+'.simt_build.json')).open('x') as f:
 json.dump(dict(binary=str(binary),nonce=receipt['nonce'],timing=timing,cold_build_seconds=receipt['cold_build_seconds'],codegen_passed=receipt['codegen']['passed']),f,indent=2)
class SimtAttention(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,b):
  assert not torch.is_autocast_enabled('cuda') and not torch.backends.cuda.matmul.allow_tf32 and torch.get_float32_matmul_precision()=='highest'
  assert all(x.dtype==torch.float32 for x in [q,k,v,b])
  y,lse,padded=extension.forward(q,k,v,b,1)
  ctx.save_for_backward(q,k,v,padded,y,lse)
  return y
 @staticmethod
 @torch.autograd.function.once_differentiable
 def backward(ctx,go):
  q,k,v,b,y,lse=ctx.saved_tensors
  return tuple(extension.backward(q,k,v,b,y,lse,go))
attention=SimtAttention.apply
