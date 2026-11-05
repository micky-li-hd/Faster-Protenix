"""Experimental FP32/BF16 fused forward/native backward; CUDA extension unvalidated."""
import torch
from isolated_layernorm import FusedLayerNormAffineFunction
class HybridLayerNormFunction(FusedLayerNormAffineFunction):
    @staticmethod
    def backward(ctx,grad_output):
        x,weight,bias,mean,rstd=ctx.saved_tensors
        assert x.dtype in (torch.float32,torch.bfloat16) and grad_output.dtype==x.dtype
        shape=tuple(ctx.normalized_shape)
        stats_shape=tuple(x.shape[:-len(shape)])+(1,)*len(shape)
        mask=(ctx.needs_input_grad[0],weight is not None and ctx.needs_input_grad[1],bias is not None and ctx.needs_input_grad[2])
        # Match OpenFoldLayerNorm's BF16 affine casts and autograd cast-back.
        native_weight=weight.to(x.dtype) if weight is not None else None
        native_bias=bias.to(x.dtype) if bias is not None else None
        dx,dw,db=torch.ops.aten.native_layer_norm_backward.default(
            grad_output.contiguous(),x,shape,mean.view(stats_shape),rstd.view(stats_shape),native_weight,native_bias,mask)
        if dw is not None:dw=dw.to(weight.dtype)
        if db is not None:db=db.to(bias.dtype)
        return dx,dw,db,None,None
