import inspect,hashlib,time,pickle,random
from contextlib import nullcontext
import numpy as np
from parallel_ppf import parallel_ppf

def install(cls,torch,enabled,expected_model_sha,expected_ppf_sha):
    original=cls.init_model
    assert hashlib.sha256(inspect.getsource(original).encode()).hexdigest()==expected_model_sha
    stats=dict(enabled=enabled,calls=0,restored=False,records=[],original_sha256=expected_model_sha)
    def rng():
        values=(random.getstate(),np.random.get_state(),torch.get_rng_state().numpy().tobytes(),
                torch.cuda.get_rng_state().cpu().numpy().tobytes() if torch.cuda.is_initialized() else None)
        return hashlib.sha256(pickle.dumps(values,protocol=5)).hexdigest()
    def init_model(self):
        stats['calls']+=1;assert stats['calls']==1
        row=dict(rng_before=rng(),cuda_initialized=torch.cuda.is_initialized());stats['records'].append(row)
        start=time.perf_counter()
        try:
            with parallel_ppf(expected_ppf_sha,8) if enabled else nullcontext({}) as ppf:
                result=original(self)
            row.update(ppf=ppf.copy(),rng_after=rng(),status='COMPLETED')
            return result
        finally:row['seconds']=time.perf_counter()-start
    cls.init_model=init_model
    def restore():
        assert cls.init_model is init_model
        cls.init_model=original;stats['restored']=True
    return restore,stats
