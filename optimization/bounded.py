import os,sys,json,fcntl,subprocess,secrets,signal,time,hashlib
from pathlib import Path
from contextlib import ExitStack
from ownership import OwnedProcesses,OWNERSHIP_ENV
p=Path(sys.argv[1]);plan=json.loads(p.read_text());root=p.parent
controller_lock=(root/'controller.lock').open('a')
fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (root/'STATE.json').exists():
 raise RuntimeError('Refusing to overwrite existing experiment state; use a new plan directory')
state={'status':'STARTING','runs':[],'pid':os.getpid(),'accepted':False,'plan_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'plan':plan}
def save(): (root/'STATE.json').write_text(json.dumps(state,indent=2))
def interrupt(s,f): raise KeyboardInterrupt(s)
for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupt)
save()
try:
 experiment_root=Path(__file__).resolve().parent
 bindings=json.loads((experiment_root/'bindings.json').read_text())
 for name,expected in bindings.items():
  h=hashlib.sha256()
  with (experiment_root/name).open('rb') as f:
   for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
  assert h.hexdigest()==expected, ('Binding changed',name)
 state['bindings_sha256']=hashlib.sha256((experiment_root/'bindings.json').read_bytes()).hexdigest()
 save()
 with ExitStack() as stack:
  for lock in ['/root/tianbali/gpu_experiments.lock','/root/tianbali/rsi_infra_protenix/work/optimization_20261021/gpu_experiments.lock']:
   path=Path(lock);path.parent.mkdir(parents=True,exist_ok=True)
   f=stack.enter_context(path.open('a'));fcntl.flock(f,fcntl.LOCK_SH|fcntl.LOCK_NB)
  for gpu in plan['gpus']:
   row=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
   assert float(row[1])<256 and float(row[2])<=5 and row[0] not in apps, ('GPU busy',gpu,row)
   path=Path('/root/tianbali/gpu_locks')/(row[0]+'.lock');path.parent.mkdir(exist_ok=True)
   f=stack.enter_context(path.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  # Recheck after all requested UUID locks are held, closing the occupancy/lock race.
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
  inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu,driver_version','--format=csv,noheader'],text=True)
  (root/'gpu_preflight.txt').write_text(inventory+'\ncompute_apps:\n'+apps)
  for gpu in plan['gpus']:
   uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
   assert uuid not in apps, ('GPU became busy',gpu,uuid)
   occupancy=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
   assert int(occupancy[0])<256 and int(occupancy[1])<=5, ('GPU occupancy changed after locks',gpu,occupancy)
  for job in plan['jobs']:
   out=Path(job['out']);out.mkdir(parents=True,exist_ok=False)
   nonce=secrets.token_hex(32);OwnedProcesses.supported();owner=OwnedProcesses(nonce,out/'cleanup.json')
   env={**os.environ,**job.get('env',{}),OWNERSHIP_ENV:nonce};env.pop('PYTHONPATH',None)
   env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,plan['gpus'])),OMP_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='2',TORCHINDUCTOR_CACHE_DIR=str(out/'cache'),RSI_OUT=str(out))
   host_namespace=os.readlink('/proc/self/ns/mnt');host_device=os.stat('/dev/shm').st_dev
   shm_size=os.statvfs('/dev/shm').f_blocks*os.statvfs('/dev/shm').f_frsize
   command=['unshare','--mount','--propagation','private','--fork',str(experiment_root/'assets/env/bin/python'),str(experiment_root/'isolated_shm.py'),'--expected-parent-mnt',host_namespace,'--size-bytes',str(shm_size),'--receipt',str(out/'private_shm_mount.json'),'--',*job['command']]
   row={'tag':job['tag'],'status':'RUNNING','nonce':nonce,'out':str(out),'command':command,'original_command':job['command'],'requested_torch_compile':bool(job.get('requested_torch_compile',False)),'requested_native_compile':bool(job.get('requested_native_compile',False))};state['runs'].append(row);state['status']='RUNNING';save();child=None;monitor=None;start=time.monotonic()
   try:
    with (out/'process.log').open('w') as log, (out/'gpu_utilization.csv').open('w') as gpu_log:
     monitor=subprocess.Popen(['nvidia-smi','-i',','.join(map(str,plan['gpus'])),'--query-gpu=timestamp,index,uuid,memory.total,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits','--loop-ms=1000'],env=env,stdout=gpu_log,stderr=subprocess.STDOUT,start_new_session=True)
     row['monitor_pid']=monitor.pid;save()
     child=subprocess.Popen(command,env=env,cwd=job['cwd'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=child.pid;save()
     child.wait(timeout=job.get('timeout',2400));assert child.returncode==0,child.returncode
    row['status']='COMPLETED'
   except BaseException as e:
    row.update(status='FAILED',error=repr(e));raise
   finally:
    row['process_seconds']=time.monotonic()-start
    receipt=owner.cleanup(child,'bounded eight-H20 Protenix run')
    row['cleanup']=receipt['status']
    monitor.poll() if monitor is not None else None
    namespace_receipt=out/'private_shm_mount.json';survivors=[]
    if namespace_receipt.exists():
     private_namespace=json.loads(namespace_receipt.read_text())['namespace']
     for entry in Path('/proc').iterdir():
      if not entry.name.isdigit():continue
      try:
       if os.readlink(entry/'ns/mnt')==private_namespace:survivors.append(int(entry.name))
      except (FileNotFoundError,ProcessLookupError):pass
    unchanged=os.readlink('/proc/self/ns/mnt')==host_namespace and os.stat('/dev/shm').st_dev==host_device
    row['private_shm_cleanup']={'status':'CLEAN' if unchanged and not survivors else 'FAILED','namespace_survivors':survivors,'host_mount_unchanged':unchanged,'mount_receipt_exists':namespace_receipt.exists()}
    row['seconds_including_cleanup']=time.monotonic()-start;save()
    assert receipt['status']=='CLEAN' and row['private_shm_cleanup']['status']=='CLEAN'
    if row['status']=='COMPLETED':assert namespace_receipt.exists()

  state['status']='COMPLETED';save()
except BaseException as e:state.update(status='FAILED',error=repr(e));save();raise
