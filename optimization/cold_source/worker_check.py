import json,os,sys
from pathlib import Path
from simt_cold_build import ensure_binary
assert os.environ['CUDA_VISIBLE_DEVICES']==''
root=Path(os.environ['RSI_OUT']);rank=int(sys.argv[1]);before=dict(os.environ)
binary,receipt,timing=ensure_binary(root)
assert os.environ==before and not receipt['cuda_initialized']
import torch,importlib.util
assert not torch.cuda.is_initialized()
spec=importlib.util.spec_from_file_location('rsi_simt_backward_v1',binary);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert callable(module.forward) and callable(module.backward) and not torch.cuda.is_initialized()
(root/f'worker{rank}.json').write_text(json.dumps(dict(rank=rank,binary=str(binary),nonce=receipt['nonce'],binary_sha256=receipt['binary_sha256'],cuda_initialized=receipt['cuda_initialized'],codegen_passed=receipt['codegen']['passed'],timing=timing),indent=2)+'\n')
