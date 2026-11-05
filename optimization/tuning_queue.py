"""Finite R-C-R queue; each GPU stage retains bounded.py locks and nonce cleanup."""
import fcntl,hashlib,json,os,signal,subprocess,time,sys
from pathlib import Path
from ownership import open_pidfd,send_pidfd
root=Path('/root/tianbali/protenix_8h20_20261103_round31');out=root/'tuning_queue'
out.mkdir(exist_ok=True)
lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'STATE.json').exists(), 'Existing queue must be inspected, not overwritten'
policies=['allocator_cache_chunk48_pair4_diff24_width64_128hybridln_global_projection_grouped_load_affine_pair_transpose_msa4_trim_msa_padding_head_gate_backward_affine16', 'allocator_cache_chunk48_pair4_diff24_width64_128hybridln_global_projection_grouped_load_affine_pair_transpose_msa4_trim_msa_padding_head_gate_backward_affine16_silu_pair4', 'allocator_cache_chunk48_pair4_diff24_width64_128hybridln_global_projection_grouped_load_affine_pair_transpose_msa4_trim_msa_padding_head_gate_backward_affine16_silu_pair5', 'allocator_cache_chunk48_pair4_diff24_width64_128hybridln_global_projection_grouped_load_affine_pair_transpose_msa4_trim_msa_padding_head_gate_backward_affine16_silu_pair6']
assert not (root/'plans/smoke_reference/STATE.json').exists()
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
 baseline='smoke_reference'
 run(baseline,[sys.executable,str(root/'bounded.py'),str(root/'plans'/baseline/'PLAN.json')],3700)
 run(baseline+'_audit',[sys.executable,str(root/'audit_arm.py'),baseline,'--steps','8'],300,cpu=True)
 assert json.loads((root/'runs'/baseline/'audit.json').read_text())['status']=='PASS'
 state['startup_passed']=True;save()
 screened=[]
 for policy in policies:
  tag='smoke_'+policy;state['current_run']=tag;save()
  try:
   run(tag,[sys.executable,str(root/'bounded.py'),str(root/'plans'/tag/'PLAN.json')],3700)
   run(tag+'_audit',[sys.executable,str(root/'audit_arm.py'),tag,'--steps','8'],300,cpu=True)
   run(tag+'_screen',[sys.executable,str(root/'smoke_screen.py'),baseline,tag],180,cpu=True)
   report=json.loads((root/'runs'/tag/'smoke_screen.json').read_text())
   screened.append({'policy':policy,**report})
  except Exception as error:
   # Continue only after proving this failed trial relinquished all owned GPU work.
   run_state=json.loads((root/'plans'/tag/'STATE.json').read_text())
   assert run_state.get('runs') and run_state['runs'][-1].get('cleanup')=='CLEAN',('Unclean or resource preflight failure',tag)
   receipt=json.loads((root/'runs'/tag/'cleanup.json').read_text());assert receipt['status']=='CLEAN'
   screened.append({'policy':policy,'screen_passed':False,'error':repr(error)})
  state['screened']=screened;save()
 parent_rows=[r for r in screened if r['policy']=='allocator_cache_chunk48_pair4_diff24_width64_128hybridln_global_projection_grouped_load_affine_pair_transpose_msa4_trim_msa_padding_head_gate_backward_affine16' and r['screen_passed']]
 assert len(parent_rows)==1, 'Matched parent must pass training screen before selecting cache combination'
 parent_seconds=parent_rows[0]['mean_max_rank_seconds_updates2_7'][1]
 viable=[r for r in screened if r['policy']!='allocator_cache_chunk48_pair4_diff24_width64_128hybridln_global_projection_grouped_load_affine_pair_transpose_msa4_trim_msa_padding_head_gate_backward_affine16' and r['screen_passed'] and r['exploratory_ratio']>1.0 and r['mean_max_rank_seconds_updates2_7'][1]<parent_seconds]
 if viable:
  selected=min(viable,key=lambda r:r['mean_max_rank_seconds_updates2_7'][1])['policy']
  state['selected_policy']=selected;save()
  run('rcr_selected',[sys.executable,str(root/'rcr_queue.py'),selected],12000)
  report=json.loads((root/'rcr_queue/RCR.json').read_text())
  state.update(status='COMPLETED',short_run_contract_passed=report['short_run_contract_passed'],target_2x_reached_in_both_metrics=report['target_2x_reached_in_both_metrics'])
 else:
  state.update(status='COMPLETED_NO_FASTER_SCREENED_CANDIDATE')
except BaseException as error:
 state.update(status='FAILED',error=repr(error));save();raise
save()
