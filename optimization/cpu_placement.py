"""Child-process-only CPU placement, before Torch/thread initialization."""
import os,sys,json
from pathlib import Path

def expand(text):
 result=set()
 for part in text.strip().split(','):
  if not part:continue
  v=part.split('-');result.update(range(int(v[0]),int(v[-1])+1))
 return result

def install(mode,rank,plan_path):
 assert mode in (None,'gpu_local_disjoint')
 before=sorted(os.sched_getaffinity(0))
 if mode is None:return {'mode':None,'before':before,'after':before},lambda:None
 assert 'torch' not in sys.modules,'Placement must precede Torch initialization'
 assert len(list(Path('/proc/self/task').iterdir()))==1,'Placement requires a fresh single-thread process'
 assert type(rank) is int and 0<=rank<8
 plan=json.loads(Path(plan_path).read_text());assert plan['mode']==mode
 all_cpus=set();all_cores=set()
 for key,entry in plan['ranks'].items():
  cpus=set(entry['cpus']);cores={tuple(x) for x in entry['physical_cores']};assert len(cpus)==48 and len(cores)==24
  assert not cpus&all_cpus and not cores&all_cores
  assert cpus<=expand((Path('/sys/devices/system/node')/('node'+str(entry['node']))/'cpulist').read_text())
  observed={}
  for cpu in cpus:
   root=Path('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology'
   core=(int((root/'physical_package_id').read_text()),int((root/'core_id').read_text()))
   observed.setdefault(core,set()).add(cpu)
   assert expand((root/'thread_siblings_list').read_text())<=cpus
  assert set(observed)==cores
  all_cpus.update(cpus);all_cores.update(cores)
 selected=plan['ranks'][str(rank)];target=set(selected['cpus']);assert target<=set(before)
 os.sched_setaffinity(0,target);assert set(os.sched_getaffinity(0))==target
 def restore():
  os.sched_setaffinity(0,before);assert sorted(os.sched_getaffinity(0))==before
 return {'mode':mode,'rank':rank,'node':selected['node'],'before':before,'after':sorted(target),'physical_cores':selected['physical_cores'],'torch_imported':False},restore
