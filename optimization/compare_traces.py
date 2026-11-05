"""Frozen short-run trace screen, never a standalone acceptance certificate."""
import argparse,json,math,statistics
from pathlib import Path

WORLD=8
STEPS=128

def validate(traces):
    if set(traces)!=set(range(WORLD)):
        raise ValueError('Expected exactly ranks0..7')
    component_keys=None
    for rank,rows in traces.items():
        if len(rows)!=STEPS or [r['step'] for r in rows]!=list(range(STEPS)):
            raise ValueError(f'Incomplete or unordered rank{rank}')
        for row in rows:
            keys=set(row['components'])
            if component_keys is None:component_keys=keys
            if not keys or keys!=component_keys:raise ValueError('Missing or inconsistent loss components')
            values=[row['loss'],row['lr'],row['observed_step_seconds'],row['peak_allocated'],row['peak_reserved'],*row['components'].values()]
            if not all(isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x) for x in values):
                raise ValueError('Nonfinite or nonnumeric measurement')
            if row['rank']!=rank or row['grad_finite'] is not True or row['grad_count']<=0:
                raise ValueError('Rank or gradient integrity failed')
            if row['observed_step_seconds']<=0 or row['peak_allocated']<0:raise ValueError('Invalid time or memory')
    return component_keys

def load(path):
    traces={}
    for f in Path(path).rglob('rank*.jsonl'):
        rank=int(f.stem.removeprefix('rank'))
        if rank in traces:raise ValueError(f'Duplicate rank{rank}')
        traces[rank]=[json.loads(line) for line in f.read_text().splitlines()]
    validate(traces)
    return traces

def timing(traces):
    values=[max(traces[r][s]['observed_step_seconds'] for r in range(WORLD)) for s in range(16,STEPS)]
    return {'mean_max_rank_seconds_updates16_127':statistics.mean(values),'peak_allocated_bytes':max(row['peak_allocated'] for rows in traces.values() for row in rows),'peak_reserved_bytes':max(row['peak_reserved'] for rows in traces.values() for row in rows)}

def compare(reference,candidate):
    keys=validate(reference)
    if validate(candidate)!=keys:raise ValueError('Component schema differs between arms')
    integrity=[]
    for r in range(WORLD):
        for s in range(STEPS):
            for key in ['lr','grad_count']:
                if reference[r][s][key]!=candidate[r][s][key]:integrity.append({'rank':r,'step':s,'field':key})
    windows=[]
    for start in range(0,STEPS,32):
        for field in ['total_loss',*sorted(keys)]:
            def average(arm):
                return statistics.mean(arm[r][s]['loss'] if field=='total_loss' else arm[r][s]['components'][field] for r in range(WORLD) for s in range(start,start+32))
            ref,cand=average(reference),average(candidate)
            delta=abs(cand-ref);limit=max(.0001,.05*abs(ref))
            windows.append({'steps':[start,start+31],'metric':field,'reference':ref,'candidate':cand,'absolute_delta':delta,'limit':limit,'passed':delta<=limit})
    rt,ct=timing(reference),timing(candidate)
    return {'trace_screen_passed':not integrity and all(w['passed'] for w in windows),'integrity_mismatches':integrity,'windows':windows,'reference_timing':rt,'candidate_timing':ct,'single_reference_exploratory_ratio':rt['mean_max_rank_seconds_updates16_127']/ct['mean_max_rank_seconds_updates16_127'],'acceptance':'NOT_ESTABLISHED_BY_TRACES_ALONE','remaining_checks':['matched source/data/workload and input order','all-node native evaluation and metric comparison','normal/EMA checkpoint validation','all-node CLEAN completion','second reference repeat and R-C-R timing'],'scope':'128-step instrumented 5SAK fixture, entirely within2000-step warmup'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('candidate');p.add_argument('--output',required=True);a=p.parse_args()
    report=compare(load(a.reference),load(a.candidate))
    with open(a.output,'x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'trace_screen_passed':report['trace_screen_passed'],'acceptance':report['acceptance']}))
