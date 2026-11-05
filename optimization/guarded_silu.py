"""Scoped native FP32 SiLU recomputation; no parameter/cache ownership changes."""
import torch
from recompute_silu import conditioned_forward

def install(cls):
    original=cls.forward
    stats={'enabled':True,'eligible':0,'fallback':0,'shapes':{},'restored':False}
    def forward(self,a,s):
        eligible=(torch.is_grad_enabled() and self.training and self.c_a==768 and self.c_s==384 and self.n==2
                  and a.dtype==s.dtype==torch.float32)
        if not eligible:
            stats['fallback']+=1
            return original(self,a,s)
        stats['eligible']+=1
        key=str((tuple(a.shape),tuple(s.shape),str(a.dtype),str(a.device)))
        stats['shapes'][key]=stats['shapes'].get(key,0)+1
        return conditioned_forward(self,a,s,'recompute_silu')
    cls.forward=forward
    def restore():
        assert cls.forward is forward,'Unexpected conditioned-transition patch replaced this adapter'
        cls.forward=original
        stats['restored']=True
    return restore,stats

def verify_stats(stats,enabled):
    assert stats['enabled'] is enabled
    if not enabled:
        assert stats=={'enabled':False}
        return
    assert stats['restored'] is True
    for k in ['eligible','fallback']:assert type(stats[k]) is int and stats[k]>=0
    assert stats['eligible']>0 and isinstance(stats['shapes'],dict) and stats['shapes']
    assert all(type(v) is int and v>0 for v in stats['shapes'].values())
    assert sum(stats['shapes'].values())==stats['eligible']
