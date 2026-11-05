"""Experimental four-block MSA retention; preserve sampling and native operators."""
def install(retained_prefix):
 import torch
 import protenix.model.modules.pairformer as module
 assert type(retained_prefix) is int and 0<=retained_prefix<=4
 original=module.checkpoint_blocks
 stats={'retained_prefix':retained_prefix,'matched_gradient_calls':0,'retained_block_calls':0}
 def checkpoint_blocks(blocks,args,blocks_per_ckpt):
  if retained_prefix==0 or not torch.is_grad_enabled() or blocks_per_ckpt is None:
   return original(blocks,args,blocks_per_ckpt)
  target=len(blocks)==4 and all(isinstance(getattr(b,'func',None),module.MSABlock) for b in blocks)
  if not target:return original(blocks,args,blocks_per_ckpt)
  assert blocks_per_ckpt==1,'Frozen native per-block checkpoint setting required'
  stats['matched_gradient_calls']+=1
  stats['retained_block_calls']+=retained_prefix
  result=original(blocks[:retained_prefix],args,None)
  return result if retained_prefix==len(blocks) else original(blocks[retained_prefix:],result,blocks_per_ckpt)
 module.checkpoint_blocks=checkpoint_blocks
 def restore():
  assert module.checkpoint_blocks is checkpoint_blocks
  module.checkpoint_blocks=original
 return stats,restore
