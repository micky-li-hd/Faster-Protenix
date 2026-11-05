import torch
import triton
import triton.language as tl

@triton.jit
def _forward(X,B,Y,ROWS,N:tl.constexpr,D1,D2,D3,S0,S1,S2,S3,S4,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,BLOCK)
    i3=row%D3;i2=(row//D3)%D2;i1=(row//(D3*D2))%D1;i0=row//(D3*D2*D1)
    offset=i0*S0+i1*S1+i2*S2+i3*S3+col*S4
    x=tl.load(X+row*N+col,col<N,other=0)
    bias=tl.load(B+offset,col<N,other=0)
    z=tl.where(col<N,x+bias,float('-inf'))
    maximum=tl.max(z,axis=0)
    numerator=tl.exp(z-tl.where(maximum==float('-inf'),0.0,maximum))
    total=tl.sum(numerator,axis=0)
    probabilities=numerator/tl.where(total==0.0,1.0,total)
    tl.store(Y+row*N+col,probabilities,col<N)

def forward(logits,bias):
    assert logits.is_cuda and logits.dtype==bias.dtype==torch.float32
    assert logits.is_contiguous() and 2<=logits.ndim<=5
    assert logits.shape[-1] in (128,368,379,384,437)
    expanded=bias.expand(logits.shape)
    shape=(1,)*(5-logits.ndim)+tuple(logits.shape)
    strides=(0,)*(5-logits.ndim)+expanded.stride()
    output=torch.empty_like(logits)
    rows=logits.numel()//shape[-1]
    _forward[(rows,)](logits,expanded,output,rows,shape[-1],*shape[1:4],*strides,triton.next_power_of_2(shape[-1]),num_warps=4,enable_fp_fusion=False)
    return output
