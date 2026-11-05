"""Isolated experimental affine backward; never loads/builds implicitly."""
from pathlib import Path
import hashlib,importlib.util
import torch
_EXTENSION=None
_COLUMNS=4
def set_columns(columns):
 global _COLUMNS
 assert columns in (1,2,4,8)
 _COLUMNS=columns


def load_verified_extension(path,digest):
 global _EXTENSION
 path=Path(path).resolve()
 assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
 spec=importlib.util.spec_from_file_location('rsi_affine_grouped_load_cuda_v1',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 assert hasattr(module,'affine_backward')
 _EXTENSION=module
 return module

def eligible(x,weight,bias,eps):
 return (x.is_cuda and x.layout==torch.strided and x.dtype in (torch.float32,torch.bfloat16)
  and x.ndim>=2 and x.shape[-1] in (64,128) and 131072<=x.numel()//x.shape[-1]<=4194304 and (x.numel()//x.shape[-1])%32==0
  and eps in (1e-5,1e-3)
  and all(t is None or (t.device==x.device and t.dtype==torch.float32 and tuple(t.shape)==(x.shape[-1],) and t.is_contiguous()) for t in (weight,bias)))

class OrderedAffineFunction(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,weight,bias,eps):
  assert _EXTENSION is not None
  x=x.contiguous();w=weight.to(x.dtype) if weight is not None else None;b=bias.to(x.dtype) if bias is not None else None
  # Match OpenFold BF16 affine cast semantics, preserving original parameters.
  with torch.autocast(device_type='cuda',enabled=False):
   out,mean,rstd=torch.ops.aten.native_layer_norm.default(x,[x.shape[-1]],w,b,eps)
  ctx.save_for_backward(x,weight,bias,mean,rstd)
  ctx.columns=_COLUMNS
  return out
 @staticmethod
 def backward(ctx,dy):
  x,weight,bias,mean,rstd=ctx.saved_tensors;dy=dy.contiguous()
  w=weight.to(x.dtype) if weight is not None else None;b=bias.to(x.dtype) if bias is not None else None
  mask=(ctx.needs_input_grad[0],weight is not None and ctx.needs_input_grad[1],bias is not None and ctx.needs_input_grad[2])
  dx=None
  if mask[0]:
   dx,_,_=torch.ops.aten.native_layer_norm_backward.default(dy,x,[x.shape[-1]],mean,rstd,w,b,(True,False,False))
  dw=db=None
  if mask[1] or mask[2]:
   dw,db=_EXTENSION.affine_backward(x,dy,mean.contiguous(),rstd.contiguous(),mask[1],mask[2],ctx.columns)
   if dw is not None:dw=dw.to(weight.dtype)
   if db is not None:db=db.to(bias.dtype)
  return dx,dw,db,None

def apply(x,weight,bias,eps):
 assert eligible(x,weight,bias,eps)
 return OrderedAffineFunction.apply(x,weight,bias,eps)
