"""Supplement existing numerical screens with mandatory per-step input/RNG checks."""
import argparse,json,math,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compare_traces import load,validate,WORLD,STEPS
HASH=re.compile(r'[0-9a-f]{64}')
FIELDS=('input_sha256','rng_before_sha256','rng_after_sha256')
def check(traces):
    validate(traces)
    for rank,rows in traces.items():
        for row in rows:
            for field in FIELDS:
                if not isinstance(row.get(field),str) or not HASH.fullmatch(row[field]):
                    raise ValueError(f'Missing/invalid {field}: rank{rank} step{row["step"]}')
            overhead=row.get('input_audit_seconds')
            if isinstance(overhead,bool) or not isinstance(overhead,(int,float)) or not math.isfinite(overhead) or overhead<0:
                raise ValueError('Missing/invalid input audit overhead')
def compare_inputs(reference,candidate,repeat):
    for arm in [reference,candidate,repeat]:check(arm)
    mismatches=[]
    for tag,left,right in [('R1_C',reference,candidate),('R1_R2',reference,repeat),('C_R2',candidate,repeat)]:
        for rank in range(WORLD):
            for step in range(STEPS):
                for field in FIELDS:
                    if left[rank][step][field]!=right[rank][step][field]:
                        mismatches.append({'comparison':tag,'rank':rank,'step':step,'field':field})
    return {'input_rng_exact':not mismatches,'mismatches':mismatches,'rank_updates_per_arm':WORLD*STEPS,'scope':'Mandatory supplemental provenance screen, not numerical, performance, evaluation, or convergence acceptance. Old traces without fingerprints are invalid inputs.','accepted':False}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('candidate');p.add_argument('repeat');p.add_argument('--output',required=True);a=p.parse_args()
    report=compare_inputs(load(a.reference),load(a.candidate),load(a.repeat))
    with open(a.output,'x') as f:json.dump(report,f,indent=2)
    if not report['input_rng_exact']:sys.exit(1)
