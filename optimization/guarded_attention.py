"""Batch48 FP32 ordinary attention only. Requires independent native validation."""
def eligible(q,k,v,bias,*,efficient,inplace_safe,grad_enabled,autocast_enabled,tf32,precision):
    if not efficient or inplace_safe or not grad_enabled or autocast_enabled or tf32 or precision!='highest':return False
    if bias is None:return False
    xs=(q,k,v,bias)
    if any(x.device.type!='cuda' or str(x.dtype)!='torch.float32' for x in xs):return False
    if any(x.device!=q.device for x in xs):return False
    if q.ndim not in (4,5) or any(x.ndim!=q.ndim for x in xs):return False
    if q.shape[0]!=48 or k.shape!=v.shape or q.shape[:-2]!=k.shape[:-2]:return False
    if q.shape[-1]!=k.shape[-1] or bias.shape[-2:]!=(q.shape[-2],k.shape[-2]):return False
    if any(b not in (1,a) for a,b in zip(q.shape[:-2],bias.shape[:-2])):return False
    if any(str(x.layout)!='torch.strided' for x in xs):return False
    # Admit only observed native global/local channel/head geometries.
    return (q.ndim==4 and q.shape[1]==16 and q.shape[-1]==48 and q.shape[-2]==k.shape[-2]) or (q.ndim==5 and q.shape[1]==4 and q.shape[-2:]==(32,32) and k.shape[-2:]==(128,32))

def install(module,torch):
    import hashlib,inspect
    original=module._attention
    assert hashlib.sha256(inspect.getsource(original).strip().encode()).hexdigest()=='ac6ef0d0faa32b4ee7f62df37d33bc133b4724143abb84f2faf33fa7b3add7bc', 'Native attention source changed'
    stats={'requested_torch_compile':True,'calls':0,'eligible_calls':0,'bypassed_calls':0,'shapes':{},'scope':'Batch48 grad-enabled FP32/highest non-autocast global/local ordinary attention; native acceptance pending.'}
    def explicit(q,k,v,bias):
        return torch.matmul(torch.softmax(torch.matmul(q,k.transpose(-2,-1))+bias,dim=-1),v)
    # Per-compiler options prevent changing unrelated model compiler settings.
    torch._dynamo.config.suppress_errors=False
    torch._dynamo.config.cache_size_limit=64
    torch._dynamo.config.accumulated_cache_size_limit=256
    torch._dynamo.config.fail_on_recompile_limit_hit=True
    compiled=torch.compile(explicit,fullgraph=True,dynamic=False,options={'pattern_matcher':False,'max_autotune':False,'triton.cudagraphs':False})
    def attention(q,k,v,attn_bias=None,use_efficient_implementation=True,inplace_safe=False):
        stats['calls']+=1
        if not eligible(q,k,v,attn_bias,efficient=use_efficient_implementation,inplace_safe=inplace_safe,grad_enabled=torch.is_grad_enabled(),autocast_enabled=torch.is_autocast_enabled('cuda'),tf32=torch.backends.cuda.matmul.allow_tf32,precision=torch.get_float32_matmul_precision()):
            stats['bypassed_calls']+=1
            return original(q,k,v,attn_bias,use_efficient_implementation,inplace_safe)
        stats['eligible_calls']+=1
        key=str((tuple(q.shape),tuple(q.stride()),tuple(k.stride()),tuple(v.stride()),tuple(attn_bias.shape),tuple(attn_bias.stride())))
        stats['shapes'][key]=stats['shapes'].get(key,0)+1
        return compiled(q,k,v,attn_bias)
    module._attention=attention
    return stats
