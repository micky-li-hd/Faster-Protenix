import os,sys,json,runpy,hashlib,random,pickle,traceback
from pathlib import Path
r=Path(__file__).resolve().parent
prior=Path('/root/tianbali/protenix_8h20_20261103_round31')
for n,h in json.loads((r/'PRIOR_BINDINGS.json').read_text()).items():
    assert hashlib.sha256((prior/n).read_bytes()).hexdigest()==h,n
sys.path[:0]=[str(prior),str(r/'cold_source')]
import torch
import numpy as np
assert os.environ['RSI_PARALLEL_COLD'] in ('0','1')
restore=None;stats=dict(enabled=False);out=Path(os.environ['RSI_OUT'])
try:
    if os.environ['RSI_PARALLEL_COLD']=='1':
        assert not torch.cuda.is_initialized()
        def rng_digest():
            return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state(),
                torch.get_rng_state().numpy().tobytes()))).hexdigest()
        before=rng_digest();env=dict(os.environ)
        import cold_build
        import affine.cold_build as affine_build
        import simt_cold_build
        from parallel_build import install
        (out/'affine').mkdir(exist_ok=True)
        restore,stats=install({'layernorm':(cold_build,out),
            'affine':(affine_build,out/'affine'),'simt':(simt_cold_build,out)})
        stats.update(rng_before=before,rng_after=rng_digest(),
                     cuda_initialized=torch.cuda.is_initialized(),environment_unchanged=dict(os.environ)==env)
        assert stats['rng_before']==stats['rng_after'] and not stats['cuda_initialized'] and stats['environment_unchanged']
    runpy.run_path(str(r/'training_entry.py'),run_name='__main__')
    if restore is not None:assert all(b['consumed'] for b in stats['builds'].values())
    stats['status']='COMPLETED'
except BaseException:
    stats.update(status='FAILED',error=traceback.format_exc());raise
finally:
    if restore is not None:restore()
    (out/f'rank{os.environ["RANK"]}.parallel_build.json').write_text(json.dumps(stats,indent=2)+'\n')
