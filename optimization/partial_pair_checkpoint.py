"""Preserve exact48-block Pairformer computation; cache a prefix of activations."""
def install(uncached_prefix):
 import torch
 import protenix.model.modules.pairformer as module
 assert type(uncached_prefix) is int and 0<=uncached_prefix<48
 original=module.checkpoint_blocks
 stats={"matched_gradient_calls":0,"retained_prefix":uncached_prefix}
 def checkpoint_blocks(blocks,args,blocks_per_ckpt):
  if uncached_prefix==0 or not torch.is_grad_enabled() or blocks_per_ckpt is None:
   return original(blocks,args,blocks_per_ckpt)
  target=(len(blocks)==48 and all(isinstance(getattr(block,'func',None),module.PairformerBlock) for block in blocks))
  if not target:return original(blocks,args,blocks_per_ckpt)
  stats['matched_gradient_calls']+=1
  assert blocks_per_ckpt==1,'This prototype is frozen for original per-block checkpointing'
  prefix=original(blocks[:uncached_prefix],args,None)
  return original(blocks[uncached_prefix:],prefix,blocks_per_ckpt)
 module.checkpoint_blocks=checkpoint_blocks
 return stats
