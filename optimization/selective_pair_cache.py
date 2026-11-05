"""Bound snapshots of reduction-heavy native Pairformer matmul outputs."""
from collections import defaultdict,deque

def install(bytes_per_block):
 import torch
 import torch.utils.checkpoint as cp
 import protenix.model.modules.pairformer as module
 assert torch.__version__.startswith('2.7.1'), 'Snapshot mode is pinned to tested PyTorch2.7.1 internals'
 assert type(bytes_per_block) is int and bytes_per_block>=0
 original=module.checkpoint_blocks
 ops={torch.ops.aten.mm.default,torch.ops.aten.bmm.default,torch.ops.aten.addmm.default}
 stats={'bytes_per_block':bytes_per_block,'stack_invocations':0,'contexts':0,'saved_op_calls':0,'cached_recompute_hits':0,'estimated_saved_bytes_total':0,'max_context_saved_bytes':0,'cache_mutation_check':True,'cache_snapshot_copies':0,'shape_eligible_calls':0,'shape_rejected_calls':0}
 def context_fn():
  stats['contexts']+=1
  decisions=defaultdict(deque);reserved=0
  def policy(ctx,op,*args,**kwargs):
   nonlocal reserved
   if op not in ops:return cp.CheckpointPolicy.PREFER_RECOMPUTE
   if ctx.is_recompute:
    assert decisions[op],('Missing recorded cache decision',str(op))
    save=decisions[op].popleft()
    if save:stats['cached_recompute_hits']+=1
   else:
    a,b=args[1:3] if op==torch.ops.aten.addmm.default else args[:2]
    elements=a.shape[-2]*b.shape[-1]
    if op==torch.ops.aten.bmm.default:elements*=a.shape[0]
    size=elements*a.element_size()
    eligible=(a.shape[-1]>=max(a.shape[-2],b.shape[-1]) if op==torch.ops.aten.bmm.default else a.shape[-1]>b.shape[-1])
    stats['shape_eligible_calls' if eligible else 'shape_rejected_calls']+=1
    save=eligible and reserved+size<=bytes_per_block
    decisions[op].append(save)
    if save:
     reserved+=size;stats['saved_op_calls']+=1;stats['estimated_saved_bytes_total']+=size
     stats['max_context_saved_bytes']=max(stats['max_context_saved_bytes'],reserved)
   return cp.CheckpointPolicy.MUST_SAVE if save else cp.CheckpointPolicy.PREFER_RECOMPUTE
  forward,recompute=cp.create_selective_checkpoint_contexts(policy,allow_cache_entry_mutation=False)
  class SnapshotMode(type(forward)):
   def __torch_dispatch__(self,func,types,args=(),kwargs=None):
    before=len(self.storage.get(func,()))
    result=super().__torch_dispatch__(func,types,args,kwargs)
    entries=self.storage.get(func,())
    if len(entries)>before:
     assert func in ops and isinstance(entries[-1],cp._VersionWrapper)
     value=entries[-1].get_val(False)
     assert isinstance(value,torch.Tensor)
     entries[-1]=cp._VersionWrapper(value.clone())
     stats['cache_snapshot_copies']+=1
    return result
  return SnapshotMode(policy,forward.storage),recompute
 def checkpoint_blocks(blocks,args,blocks_per_ckpt):
  target=(bytes_per_block>0 and torch.is_grad_enabled() and blocks_per_ckpt is not None and len(blocks)==48 and all(isinstance(getattr(b,'func',None),module.PairformerBlock) for b in blocks))
  if not target:return original(blocks,args,blocks_per_ckpt)
  assert blocks_per_ckpt==1,'Only frozen native per-block checkpointing is supported'
  stats['stack_invocations']+=1
  values=args if type(args) is tuple else (args,)
  for block in blocks:
   values=cp.checkpoint(block,*values,use_reentrant=False,context_fn=context_fn)
   if type(values) is not tuple:values=(values,)
  return values
 module.checkpoint_blocks=checkpoint_blocks
 return stats
