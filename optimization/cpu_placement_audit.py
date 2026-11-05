"""Verify child-only placement and preserve per-thread end-state evidence."""
import os,json,hashlib
from pathlib import Path

def verify_receipt(root,receipt,mode,rank):
 root=Path(root);plan=json.loads((root/'CPU_PLACEMENT_PLAN.json').read_text());stats=receipt['initial']
 assert mode in (None,'gpu_local_disjoint') and stats['mode']==mode
 assert receipt['rank']==rank and receipt['plan_sha256']==hashlib.sha256((root/'CPU_PLACEMENT_PLAN.json').read_bytes()).hexdigest()
 assert stats['before'] and len(stats['before'])==len(set(stats['before']))
 if mode:
  expected=plan['ranks'][str(rank)];assert stats['rank']==rank and stats['node']==expected['node']
  assert stats['after']==expected['cpus'] and stats['physical_cores']==expected['physical_cores'] and stats['torch_imported'] is False
 else:assert stats['after']==stats['before']
 assert set(stats['after'])<=set(stats['before'])
 assert receipt['thread_affinity_before_restore']
 for tid,cpus in receipt['thread_affinity_before_restore'].items():
  assert int(tid)>0 and cpus and set(cpus)<=set(stats['after'])
 assert receipt['main_affinity_after_restore']==stats['before'] and receipt['restored'] is True

def finish_receipt(root,out,rank,stats,restore):
 threads={}
 for p in Path('/proc/self/task').iterdir():
  try:threads[p.name]=sorted(os.sched_getaffinity(int(p.name)))
  except ProcessLookupError:continue
 restore()
 report={'rank':rank,'initial':stats,'thread_affinity_before_restore':threads,'main_affinity_after_restore':sorted(os.sched_getaffinity(0)),'restored':True,'plan_sha256':hashlib.sha256((Path(root)/'CPU_PLACEMENT_PLAN.json').read_bytes()).hexdigest()}
 verify_receipt(root,report,stats['mode'],rank)
 (Path(out)/f'rank{rank}.cpu_placement.json').write_text(json.dumps(report,indent=2)+'\n')
