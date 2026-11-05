"""Exact bit-copy candidate for contiguous pair tensors; no training patch."""
import torch
import triton
import triton.language as tl


@triton.jit
def _copy(X, Y, M: tl.constexpr, N: tl.constexpr, C: tl.constexpr,
          TI: tl.constexpr, TJ: tl.constexpr, BC: tl.constexpr):
    i = tl.program_id(0) * TI + tl.arange(0, TI)
    j = tl.program_id(1) * TJ + tl.arange(0, TJ)
    c = tl.arange(0, BC)
    batch = tl.program_id(2)
    offsets = ((batch * M + i[:, None, None]) * N + j[None, :, None]) * C + c[None, None, :]
    target = ((batch * N + j[None, :, None]) * M + i[:, None, None]) * C + c[None, None, :]
    mask = (i[:, None, None] < M) & (j[None, :, None] < N) & (c[None, None, :] < C)
    bits = tl.load(X + offsets, mask, other=0)
    tl.store(Y + target, bits, mask)


def eligible(x):
    return (x.is_cuda and x.dtype in (torch.bfloat16, torch.float32)
            and x.ndim in (3, 4) and x.is_contiguous()
            and all(d > 0 for d in x.shape) and 1 < x.shape[-1] <= 256
            and x.shape[-2] > 1 and x.shape[-3] > 1)


def raw(x, tile=4):
    if not eligible(x):
        return x.transpose(-2, -3).contiguous()
    m, n, c = x.shape[-3:]
    shape = (*x.shape[:-3], n, m, c)
    y = torch.empty(shape, device=x.device, dtype=x.dtype)
    integer = torch.int16 if x.dtype == torch.bfloat16 else torch.int32
    # Integer views preserve NaN payloads, signed zero and all finite bit patterns.
    with torch.cuda.device(x.device):
        _copy[(triton.cdiv(m, tile), triton.cdiv(n, tile), x.numel() // (m*n*c))](
            x.view(integer), y.view(integer), m, n, c, tile, tile,
            triton.next_power_of_2(c), num_warps=4)
    return y


class _Transpose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tile):
        return raw(x, tile)

    @staticmethod
    def backward(ctx, grad):
        # Preserve native transpose-view gradient behavior, including strides.
        return grad.transpose(-2, -3), None


def transpose(x, tile=4):
    if not eligible(x):
        return x.transpose(-2, -3).contiguous()
    return _Transpose.apply(x, tile)
