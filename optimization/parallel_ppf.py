from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import hashlib,inspect
import numpy as np
from scipy.stats import truncnorm

@contextmanager
def parallel_ppf(expected_sha,workers=8):
    original=truncnorm._ppf
    assert hashlib.sha256(inspect.getsource(original).encode()).hexdigest()==expected_sha
    stats=dict(calls=0,parallel_calls=0,elements=0,workers=workers,restored=False)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def candidate(q,a,b):
            stats['calls']+=1
            if (not isinstance(q,np.ndarray) or q.ndim!=1 or q.size<262144
                or np.size(a)!=1 or np.size(b)!=1
                or np.asarray(a).reshape(-1)[0]!=-2 or np.asarray(b).reshape(-1)[0]!=2):
                return original(q,a,b)
            stats['parallel_calls']+=1;stats['elements']+=q.size
            edges=np.linspace(0,q.size,workers+1,dtype=np.int64)
            jobs=[pool.submit(original,q[edges[i]:edges[i+1]],a,b) for i in range(workers)]
            return np.concatenate([job.result() for job in jobs])
        truncnorm._ppf=candidate
        try:yield stats
        finally:
            assert truncnorm._ppf is candidate
            truncnorm._ppf=original;stats['restored']=True
