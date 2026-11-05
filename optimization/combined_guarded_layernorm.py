"""Large FP32/BF16 LayerNorm experiment; preserve original parameter identities and fallback."""
def eligible(module,x):
    if x.device.type!='cuda' or str(x.dtype) not in ('torch.float32','torch.bfloat16') or str(x.layout)!='torch.strided':return False
    if tuple(module.c_in) not in ((64,),(128,)) or x.ndim<2:return False
    width=module.c_in[0]
    if x.shape[-1]!=width or x.numel()//width<131072:return False
    if module.eps not in (1e-5,1e-3):return False
    return all(p is None or (p.device==x.device and str(p.dtype)=='torch.float32' and tuple(p.shape)==(width,) and p.is_contiguous()) for p in (module.weight,module.bias))

def make_forward(original):
    import hashlib,inspect
    from combined_layernorm import CombinedLayerNormFunction
    assert hashlib.sha256(inspect.getsource(original).strip().encode()).hexdigest()=='693a0dfa960b68e2f768ca5d833f75f93cc87e21563bed0dd640dae72268698e','Native LayerNorm source changed'
    stats={'eligible_calls':0,'bypassed_calls':0}
    def forward(self,x):
        if not eligible(self,x):
            stats['bypassed_calls']+=1
            return original(self,x)
        stats['eligible_calls']+=1
        return CombinedLayerNormFunction.apply(x,self.weight,self.bias,self.c_in,self.eps)
    return forward,stats

def install(cls):
    forward,stats=make_forward(cls.forward)
    cls.forward=forward
    return stats
