"""Existing fused forward with narrowly gated mainloop affine backward."""
import torch
import ordered_affine
from hybrid_layernorm import HybridLayerNormFunction
_STATS={'eligible_backward':0,'native_backward_fallback':0,'eligible_width64':0,'eligible_width128':0}
class CombinedLayerNormFunction(HybridLayerNormFunction):
 @staticmethod
 def backward(ctx,grad_output):
  x,weight,bias,mean,rstd=ctx.saved_tensors
  assert x.dtype in (torch.float32,torch.bfloat16) and grad_output.dtype==x.dtype
  shape=tuple(ctx.normalized_shape);stats_shape=tuple(x.shape[:-len(shape)])+(1,)*len(shape)
  mask=(ctx.needs_input_grad[0],weight is not None and ctx.needs_input_grad[1],bias is not None and ctx.needs_input_grad[2])
  w=weight.to(x.dtype) if weight is not None else None;b=bias.to(x.dtype) if bias is not None else None
  dy=grad_output.contiguous()
  if shape not in ((64,),(128,)) or not ordered_affine.eligible(x,weight,bias,ctx.eps):
   _STATS['native_backward_fallback']+=1
   dx,dw,db=torch.ops.aten.native_layer_norm_backward.default(dy,x,shape,mean.view(stats_shape),rstd.view(stats_shape),w,b,mask)
  else:
   assert ordered_affine._EXTENSION is not None
   _STATS['eligible_backward']+=1
   _STATS['eligible_width'+str(shape[0])]+=1
   dx=dw=db=None
   if mask[0]:dx=torch.ops.aten.native_layer_norm_backward.default(dy,x,shape,mean.view(stats_shape),rstd.view(stats_shape),w,b,(True,False,False))[0]
   if mask[1] or mask[2]:dw,db=ordered_affine._EXTENSION.affine_backward(x,dy,mean.contiguous(),rstd.contiguous(),mask[1],mask[2],4)
  if dw is not None:dw=dw.to(weight.dtype)
  if db is not None:db=db.to(bias.dtype)
  return dx,dw,db,None,None
