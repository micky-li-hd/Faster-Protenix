"""Retain a prefix of the native24-block diffusion transformer, leaving every operation unchanged."""
def install(retained_prefix):
 import torch
 import protenix.model.modules.transformer as module
 assert type(retained_prefix) is int and 0<=retained_prefix<=24
 original=module.checkpoint_blocks
 stats={'retained_prefix':retained_prefix,'matched_gradient_calls':0,'retained_block_calls':0}
 def checkpoint_blocks(blocks,args,blocks_per_ckpt):
  if retained_prefix==0 or not torch.is_grad_enabled() or blocks_per_ckpt is None:
   return original(blocks,args,blocks_per_ckpt)
  target=len(blocks)==24 and all(isinstance(getattr(b,'func',None),module.DiffusionTransformerBlock) for b in blocks)
  if not target:return original(blocks,args,blocks_per_ckpt)
  assert blocks_per_ckpt==1,'Frozen per-block checkpoint setting required'
  stats['matched_gradient_calls']+=1;stats['retained_block_calls']+=retained_prefix
  result=original(blocks[:retained_prefix],args,None)
  return result if retained_prefix==len(blocks) else original(blocks[retained_prefix:],result,blocks_per_ckpt)
 module.checkpoint_blocks=checkpoint_blocks
 return stats
