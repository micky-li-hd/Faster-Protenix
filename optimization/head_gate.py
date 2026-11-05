"""Candidate: preserve gate multiplication rounding while materializing head layout."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gate_flat(O, G, Y, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
               TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = index % D
    head = index // D % H
    token = index // (D * H) % N
    batch = index // (D * H * N)
    source = ((batch * H + head) * N + token) * D + channel
    mask = index < TOTAL
    left = tl.load(O + source, mask, other=0).to(tl.float32)
    right = tl.load(G + index, mask, other=0).to(tl.float32)
    product = left * right
    tl.store(Y + index, product, mask)


def eligible(o, g):
    if not (o.is_cuda and g.is_cuda and o.device == g.device
            and o.dtype == g.dtype == torch.bfloat16
            and o.ndim == g.ndim == 4 and o.shape == g.shape
            and g.is_contiguous()):
        return False
    b, n, h, d = o.shape
    return (b > 0 and n > 1 and h == 4 and d == 32
            and o.stride() == (h*n*d, d, n*d, 1))


def raw(o, g, block=256):
    assert eligible(o, g)
    b, n, h, d = o.shape
    result = torch.empty((b, n, h*d), device=o.device, dtype=o.dtype)
    with torch.cuda.device(o.device):
        _gate_flat[(triton.cdiv(o.numel(), block),)](
            o, g, result, n, h, d, o.numel(), block, num_warps=4,
            enable_fp_fusion=False)
    return result


class _GateFlat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o, g, block):
        ctx.save_for_backward(o, g)
        return raw(o, g, block)

    @staticmethod
    def backward(ctx, grad):
        o, g = ctx.saved_tensors
        expanded = grad.reshape(o.shape)
        # Keep native multiplication for both VJPs, in the original operand order.
        return expanded * g, expanded * o, None


def gate_flat(o, g, block=256):
    if not eligible(o, g):
        product = o * g
        return product.reshape(*product.shape[:-2], -1)
    return _GateFlat.apply(o, g, block)
