def verify_stats(stats,policy,steps):
 assert policy in (None,'mm')
 if policy is None:
  assert stats=={'policy':None}
  return
 assert stats['policy']==policy and stats['mutation_check']
 assert stats['matched_stack_calls']>=steps and stats['contexts']==24*stats['matched_stack_calls']
 assert stats['saved_op_calls']>=stats['cached_recompute_hits']>0
 assert sum(stats['saved_by_op'].values())==stats['saved_op_calls']
 assert sum(stats['hit_by_op'].values())==stats['cached_recompute_hits']
 assert set(stats['saved_by_op'])<= {'aten.mm.default','aten.addmm.default'}
 assert stats['saved_bytes_estimate']>=stats['max_context_saved_bytes_estimate']>0
