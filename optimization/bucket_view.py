"""Change only DDP gradient storage aliasing; keep native reducer and optimizer."""
import functools

def install(training):
 original=training.DDP
 stats={'enabled':True,'instances':0,'gradient_as_bucket_view':True,'static_graph':True,'find_unused_parameters':False}
 @functools.wraps(original)
 def ddp(*args,**kwargs):
  assert kwargs.get('static_graph') is True
  assert kwargs.get('find_unused_parameters',False) is False
  assert 'gradient_as_bucket_view' not in kwargs
  result=original(*args,**kwargs,gradient_as_bucket_view=True)
  stats['instances']+=1
  return result
 training.DDP=ddp
 def restore():
  assert training.DDP is ddp
  training.DDP=original
 return restore,stats
