"""Narrow global attention adapter. Unsupported calls delegate to parent policy."""
def eligible(q,k,v,b,*,efficient,inplace_safe,grad_enabled,autocast_enabled,tf32,precision):
 if not efficient or inplace_safe or not grad_enabled or autocast_enabled or tf32 or precision!='highest':return False
 if b is None:return False
 xs=(q,k,v,b)
 if any(x.ndim!=4 or x.device.type!='cuda' or str(x.dtype)!='torch.float32' or str(x.layout)!='torch.strided' for x in xs):return False
 if any(x.device!=q.device for x in xs):return False
 if q.shape[0]!=48 or q.shape[1]!=16 or q.shape[2] not in (368,379,384,437) or q.shape[3]!=48:return False
 if k.shape!=q.shape or v.shape!=q.shape or b.shape!=(1,16,q.shape[2],q.shape[2]):return False
 if any(x.stride(-1)!=1 for x in [q,k,v]):return False
 return True

def install(module,torch,candidate):
 original=module._attention
 stats=dict(enabled=True,calls=0,eligible_calls=0,bypassed_calls=0,shapes={},restored=False)
 def attention(q,k,v,attn_bias=None,use_efficient_implementation=True,inplace_safe=False):
  stats['calls']+=1
  if not eligible(q,k,v,attn_bias,efficient=use_efficient_implementation,inplace_safe=inplace_safe,grad_enabled=torch.is_grad_enabled(),autocast_enabled=torch.is_autocast_enabled('cuda'),tf32=torch.backends.cuda.matmul.allow_tf32,precision=torch.get_float32_matmul_precision()):
   stats['bypassed_calls']+=1
   return original(q,k,v,attn_bias,use_efficient_implementation,inplace_safe)
  stats['eligible_calls']+=1
  key=str((tuple(q.shape),tuple(q.stride()),tuple(k.stride()),tuple(v.stride()),tuple(attn_bias.stride())))
  stats['shapes'][key]=stats['shapes'].get(key,0)+1
  return candidate(q,k,v,attn_bias)
 module._attention=attention
 def restore():
  assert module._attention is attention
  module._attention=original;stats['restored']=True
 return restore,stats
