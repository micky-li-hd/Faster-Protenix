"""Explicit isolated extension loader; native source classes preserved byte-for-byte."""
import numbers,hashlib,importlib.util,sys
from typing import Any,Optional,Union
from pathlib import Path
import torch
from torch.nn.parameter import Parameter
fast_layer_norm_cuda_v2=None

def load_verified_extension(path,expected_sha256):
    global fast_layer_norm_cuda_v2
    path=Path(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected_sha256
    assert fast_layer_norm_cuda_v2 is None,'Load one explicitly verified binary per process'
    name='rsi_fast_layer_norm_cuda_v1'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fast_layer_norm_cuda_v2=module
    return module

class FusedLayerNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        input: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        normalized_shape: torch.Size,
        eps: float,
    ) -> torch.Tensor:
        d = input.dtype

        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()

        if weight is None:
            if bias is None:
                output, mean, invvar = fast_layer_norm_cuda_v2.forward_none_affine(
                    input_, ctx.normalized_shape, ctx.eps
                )
            else:
                output, mean, invvar = fast_layer_norm_cuda_v2.forward_with_bias_affine(
                    input_, ctx.normalized_shape, bias.to(d), ctx.eps
                )
        else:
            if bias is None:
                (
                    output,
                    mean,
                    invvar,
                ) = fast_layer_norm_cuda_v2.forward_with_weight_affine(
                    input_, ctx.normalized_shape, weight.to(d), ctx.eps
                )
            else:
                output, mean, invvar = fast_layer_norm_cuda_v2.forward_with_both_affine(
                    input_,
                    ctx.normalized_shape,
                    weight.to(d),
                    bias.to(d),
                    ctx.eps,
                )
        ctx.save_for_backward(input_, weight, bias, mean, invvar)
        return output

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], ...]:
        d = grad_output.dtype
        input_, weight_, bias_, mean, invvar = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if weight_ is None:
            if bias_ is None:
                (
                    grad_input,
                    grad_weight,
                    grad_bias,
                ) = fast_layer_norm_cuda_v2.backward_none_affine(
                    grad_output.contiguous(),
                    mean,
                    invvar,
                    input_,
                    ctx.normalized_shape,
                    ctx.eps,
                )
            else:
                (
                    grad_input,
                    grad_weight,
                    grad_bias,
                ) = fast_layer_norm_cuda_v2.backward_with_bias_affine(
                    grad_output.contiguous(),
                    mean,
                    invvar,
                    input_,
                    ctx.normalized_shape,
                    bias_.to(dtype=d),
                    ctx.eps,
                )
        else:
            if bias_ is None:
                (
                    grad_input,
                    grad_weight,
                    grad_bias,
                ) = fast_layer_norm_cuda_v2.backward_with_weight_affine(
                    grad_output.contiguous(),
                    mean,
                    invvar,
                    input_,
                    ctx.normalized_shape,
                    weight_.to(dtype=d),
                    ctx.eps,
                )
            else:
                (
                    grad_input,
                    grad_weight,
                    grad_bias,
                ) = fast_layer_norm_cuda_v2.backward_with_both_affine(
                    grad_output.contiguous(),
                    mean,
                    invvar,
                    input_,
                    ctx.normalized_shape,
                    weight_.to(dtype=d),
                    bias_.to(dtype=d),
                    ctx.eps,
                )
        return (
            grad_input,
            None if weight_ is None else grad_weight,
            None if bias_ is None else grad_bias,
            None,
            None,
            None,
        )

class FusedLayerNorm(torch.nn.Module):
    """
    Args:
        normalized_shape (int or list or torch.Size) input shape from an expected input of size
        create_scale (bool) If set to False, the layer will not learn an additive weight, Default: True
        create_offset (bool) If set to False, the layer will not learn an additive bias, Default: True
        eps (float) a value added to the denominator for numerical stability. Default: 1e-5
    """

    def __init__(
        self,
        normalized_shape: Union[int, list[int], torch.Size],
        create_scale: bool = True,
        create_offset: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super(FusedLayerNorm, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        if create_scale:
            self.weight = Parameter(torch.ones(*normalized_shape))
        else:
            self.weight = None

        if create_offset:
            self.bias = Parameter(torch.zeros(*normalized_shape))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return FusedLayerNormAffineFunction.apply(
            input, self.weight, self.bias, self.normalized_shape, self.eps
        )
