"""Eight-update candidate screening; never the128-update acceptance result."""
import json,statistics
from pathlib import Path
import compare_traces
ROOT=Path(__file__).resolve().parent

def screen(reference_tag,candidate_tag):
 compare_traces.STEPS=8
 tags=[reference_tag,candidate_tag];arms=[];audits=[]
 for tag in tags:
  run=ROOT/'runs'/tag
  audit=json.loads((run/'audit.json').read_text())
  assert audit['status']=='PASS' and audit['ranks']==8 and audit['updates']==8
  audits.append(audit);arms.append(compare_traces.load(run))
 initial=[[json.loads((ROOT/'runs'/tag/f'rank{rank}.initial.json').read_text()) for rank in range(8)] for tag in tags]
 exact=initial[0]==initial[1]
 mismatches=[]
 for rank in range(8):
  for step in range(8):
   for field in ['input_sha256','rng_before_sha256','rng_after_sha256','lr','grad_count']:
    if arms[0][rank][step][field]!=arms[1][rank][step][field]:mismatches.append({'rank':rank,'step':step,'field':field})
 metrics=[]
 for field in ['total_loss',*sorted(arms[0][0][0]['components'])]:
  means=[statistics.mean(row['loss'] if field=='total_loss' else row['components'][field] for rows in arm.values() for row in rows) for arm in arms]
  delta=abs(means[1]-means[0]);limit=max(.0001,.05*abs(means[0]));metrics.append({'metric':field,'means':means,'delta':delta,'limit':limit,'passed':delta<=limit})
 times=[statistics.mean(max(arm[r][s]['observed_step_seconds'] for r in range(8)) for s in range(2,8)) for arm in arms]
 result={'tags':tags,'screen_passed':exact and not mismatches and all(m['passed'] for m in metrics),'initial_state_exact':exact,'input_rng_lr_gradient_mismatches':mismatches,'metrics':metrics,'mean_max_rank_seconds_updates2_7':times,'exploratory_ratio':times[0]/times[1],'full_run_peak_allocated_bytes':[max(m['peak_allocated_bytes'] for m in a['full_run_memory']) for a in audits],'full_run_peak_reserved_bytes':[max(m['peak_reserved_bytes'] for m in a['full_run_memory']) for a in audits],'scope':'Exploratory8-update candidate screen only; native eval integrity checked but numerical eval comparison and reference repetition remain required in128-update R-C-R. Not acceptance or convergence.'}
 with (ROOT/'runs'/candidate_tag/'smoke_screen.json').open('x') as f:json.dump(result,f,indent=2)
 return result
if __name__=='__main__':
 import sys
 print(json.dumps(screen(sys.argv[1],sys.argv[2])))
