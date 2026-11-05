"""Diagnostic FP32 bias/softmax fusion; native matmul and native softmax backward."""
import torch

class BiasSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx,logits,bias):
        if logits.is_cuda:
            from softmax_kernel import forward
            probabilities=forward(logits,bias)
        else:
            probabilities=torch.softmax(logits+bias,dim=-1)
        ctx.save_for_backward(probabilities)
        ctx.bias_shape=bias.shape
        return probabilities

    @staticmethod
    def backward(ctx,grad):
        probabilities,=ctx.saved_tensors
        derivative=torch.ops.aten._softmax_backward_data.default(grad.contiguous(),probabilities,-1,probabilities.dtype)
        return (derivative if ctx.needs_input_grad[0] else None,
                derivative.sum_to_size(ctx.bias_shape) if ctx.needs_input_grad[1] else None)

def attention(q,k,v,bias):
    assert q.dtype==k.dtype==v.dtype==bias.dtype==torch.float32
    assert not torch.is_autocast_enabled('cuda') and not torch.backends.cuda.matmul.allow_tf32
    assert torch.get_float32_matmul_precision()=='highest'
    scores=torch.matmul(q,k.transpose(-2,-1))
    return torch.matmul(BiasSoftmax.apply(scores,bias),v)
