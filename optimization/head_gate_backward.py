"""Unvalidated follow-up: fuse gate VJPs with head-major grad_o storage."""
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
        ctx.block = block
        return raw(o, g, block)

    @staticmethod
    def backward(ctx, grad):
        o, g = ctx.saved_tensors
        if (torch.is_grad_enabled() or grad.dtype != o.dtype or grad.ndim != 3
                or grad.device != o.device):
            expanded = grad.reshape(o.shape)
            return expanded * g, expanded * o, None
        b, n, h, d = o.shape
        assert grad.shape == (b, n, h*d)
        grad_o = torch.empty_strided(o.shape, o.stride(), dtype=o.dtype, device=o.device)
        grad_g = torch.empty(g.shape, dtype=g.dtype, device=g.device)
        with torch.cuda.device(o.device):
            _gate_vjps[(triton.cdiv(o.numel(), ctx.block),)](
                o, g, grad, grad_o, grad_g, n, h, d,
                *grad.stride(), o.numel(), ctx.block, num_warps=4,
                enable_fp_fusion=False)
        return grad_o, grad_g, None


@triton.jit
def _gate_vjps(O, G, DY, DO, DG, N: tl.constexpr, H: tl.constexpr,
               D: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr,
               S2: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = index % D
    head = index // D % H
    token = index // (D * H) % N
    batch = index // (D * H * N)
    source = ((batch * H + head) * N + token) * D + channel
    mask = index < TOTAL
    upstream = tl.load(DY + batch*S0 + token*S1 + (head*D+channel)*S2,
                       mask, other=0).to(tl.float32)
    gate = tl.load(G + index, mask, other=0).to(tl.float32)
    value = tl.load(O + source, mask, other=0).to(tl.float32)
    tl.store(DO + source, upstream * gate, mask)
    tl.store(DG + index, upstream * value, mask)


def gate_flat(o, g, block=256):
    if not eligible(o, g):
        product = o * g
        return product.reshape(*product.shape[:-2], -1)
    return _GateFlat.apply(o, g, block)
