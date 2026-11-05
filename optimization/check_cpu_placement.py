from pathlib import Path
import subprocess,sys,os,json,hashlib,concurrent.futures
r=Path(__file__).resolve().parent
if len(sys.argv)>1:
 from cpu_placement import install
 rank=int(sys.argv[1]);mode=None if sys.argv[2]=='none' else 'gpu_local_disjoint'
 stats,restore=install(mode,rank,r/'CPU_PLACEMENT_PLAN.json')
 try:
  import threading
  seen=[];t=threading.Thread(target=lambda:seen.append(sorted(os.sched_getaffinity(0))));t.start();t.join()
  child=json.loads(subprocess.check_output([sys.executable,'-c','import os,json;print(json.dumps(sorted(os.sched_getaffinity(0))))']))
  assert seen==[stats['after']] and child==stats['after']
  assert 'torch' not in sys.modules
  stats.update(thread_inheritance_exact=True,child_inheritance_exact=True)
 finally:restore()
 stats['restored']=sorted(os.sched_getaffinity(0))==stats['before'];assert stats['restored'];print(json.dumps(stats));raise SystemExit
initial=sorted(os.sched_getaffinity(0));tasks=[(i,m) for m in ['none','enabled'] for i in range(8)]
def run(args):return json.loads(subprocess.check_output([sys.executable,str(__file__),str(args[0]),args[1]],env={**os.environ,'CUDA_VISIBLE_DEVICES':''}))
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(run,tasks))
assert sorted(os.sched_getaffinity(0))==initial
report={'status':'PASS_CHILD_CPU_AFFINITY_INHERITANCE_RESTORE','cases':rows,'parent_affinity_unchanged':True,'cuda_initialized':False,'accepted':False,'bindings_sha256':hashlib.sha256((r/'PREPARATION_BINDINGS.json').read_bytes()).hexdigest()};(r/'CPU_PLACEMENT_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'cases':len(rows),'parent_affinity_unchanged':True,'cuda_initialized':False}))
