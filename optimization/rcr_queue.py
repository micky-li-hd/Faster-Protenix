"""Finite R-C-R queue; each GPU stage retains bounded.py locks and nonce cleanup."""
import fcntl,hashlib,json,os,signal,subprocess,time,sys
from pathlib import Path
from ownership import open_pidfd,send_pidfd
root=Path('/root/tianbali/protenix_8h20_20261103_round31');out=root/'rcr_queue'
out.mkdir(exist_ok=True)
lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'STATE.json').exists(), 'Existing queue must be inspected, not overwritten'
assert json.loads((root/'runs/smoke_reference/audit.json').read_text())['status']=='PASS'
assert len(sys.argv)==2,'Explicit candidate policy required'
policy=sys.argv[1]
assert policy in json.loads((root/'performance_policies.json').read_text()) and policy!='reference'
tags=['reference128_r1','candidate128_'+policy,'reference128_r2']
assert all(not (root/'plans'/tag/'STATE.json').exists() for tag in tags)
state={'status':'RUNNING','pid':os.getpid(),'stages':[],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'accepted':False}
def save(): (out/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
def stop(*_):raise KeyboardInterrupt('queue interrupted')
for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
def run(name,cmd,timeout,cpu=False):
 row={'name':name,'status':'STARTING','command':cmd};state['stages'].append(row);save()
 env=dict(os.environ)
 if cpu:env['CUDA_VISIBLE_DEVICES']=''
 proc=None;fd=None;started=time.monotonic()
 try:
  with (out/(name+'.log')).open('x') as log:
   proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   fd=open_pidfd(proc.pid);row.update(status='RUNNING',pid=proc.pid);save()
   rc=proc.wait(timeout=timeout)
   assert rc==0,(name,rc)
  row['status']='COMPLETED'
 except BaseException as error:
  row.update(status='FAILED',error=repr(error));raise
 finally:
  if proc is not None and proc.poll() is None:
   # Signal pinned direct child; a GPU controller invokes its own exact-nonce cleanup.
   send_pidfd(fd,signal.SIGTERM)
   proc.wait(timeout=30)
  if fd is not None:os.close(fd)
  row['seconds']=time.monotonic()-started;save()
save()
try:
 for tag in tags:
  state['current_run']=tag;save()
  run(tag,[sys.executable,str(root/'bounded.py'),str(root/'plans'/tag/'PLAN.json')],3700)
  result=json.loads((root/'plans'/tag/'STATE.json').read_text())
  assert result['status']=='COMPLETED' and result['runs'][0]['cleanup']=='CLEAN'
  run(tag+'_audit',[sys.executable,str(root/'audit_arm.py'),tag,'--steps','128'],300,cpu=True)
  assert json.loads((root/'runs'/tag/'audit.json').read_text())['status']=='PASS'
 run('summarize',[sys.executable,str(root/'summarize_rcr.py'),*tags,'--candidate-policy',policy,'--output',str(out/'RCR.json')],180,cpu=True)
 report=json.loads((out/'RCR.json').read_text())
 state.update(status='COMPLETED',short_run_contract_passed=report['short_run_contract_passed'],target_2x_reached_in_both_metrics=report['target_2x_reached_in_both_metrics'])
except BaseException as error:
 state.update(status='FAILED',error=repr(error));save();raise
save()
