"""Strict value/layout fingerprinting outside compiled model regions; no RNG draws."""
import hashlib
import json
import struct
from collections.abc import Mapping
import random
import numpy as np
import torch

def fingerprint(value):
    h=hashlib.sha256()
    def emit(tag,payload):
        tag=tag.encode();h.update(struct.pack('>Q',len(tag)));h.update(tag)
        h.update(struct.pack('>Q',len(payload)));h.update(payload)
    def visit(x):
        if isinstance(x,torch.Tensor):
            if x.layout != torch.strided or x.is_quantized:
                raise TypeError('Unsupported tensor layout/quantization')
            emit('tensor-metadata',json.dumps([str(x.dtype),list(x.shape),list(x.stride())]).encode())
            emit('tensor-value',x.detach().resolve_conj().resolve_neg().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray):
            if x.dtype.hasobject: raise TypeError('Object arrays need an explicit serializer')
            emit('numpy-metadata',json.dumps([x.dtype.str,list(x.shape),list(x.strides)]).encode())
            emit('numpy-value',x.tobytes(order='C'))
        elif isinstance(x,np.generic):
            visit(np.asarray(x))
        elif isinstance(x,Mapping):
            if any(type(k) not in (str,int) for k in x):raise TypeError('Unsupported mapping key')
            emit('mapping-size',str(len(x)).encode())
            for k in sorted(x,key=lambda k:(type(k).__name__,str(k))):visit(k);visit(x[k])
        elif isinstance(x,(tuple,list)):
            emit(type(x).__name__,str(len(x)).encode())
            for item in x:visit(item)
        elif x is None:emit('none',b'')
        elif type(x) is bool:emit('bool',b'1' if x else b'0')
        elif type(x) is int:emit('int',str(x).encode())
        elif type(x) is float:emit('float',struct.pack('>d',x))
        elif type(x) is str:emit('str',x.encode())
        elif type(x) is bytes:emit('bytes',x)
        else:raise TypeError('Unsupported fingerprint object: '+type(x).__module__+'.'+type(x).__qualname__)
    visit(value)
    return h.hexdigest()

def rng_fingerprint(include_cuda=True):
    state={'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state()}
    if include_cuda:
        if not torch.cuda.is_initialized():raise RuntimeError('CUDA RNG must already be initialized by native training')
        state['torch_cuda']=torch.cuda.get_rng_state(torch.cuda.current_device())
    return fingerprint(state)
