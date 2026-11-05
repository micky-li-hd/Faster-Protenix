"""CPU-only cold compilation, invoked under the per-run build lock."""
import json,os,sys,time,subprocess
from pathlib import Path
from cold_build import checked_manifest,sha,NONCE
started=time.monotonic();root=Path(__file__).resolve().parent;folder=Path(sys.argv[1]).resolve()
manifest,digest=checked_manifest(root);nonce=os.environ[NONCE]
assert os.environ['CUDA_VISIBLE_DEVICES']==''
assert json.loads((folder/'started.json').read_text())['nonce']==nonce
assert not (folder/'receipt.json').exists()
prefix=Path(manifest['toolchain']);build=folder/'build';build.mkdir()
os.environ.update(CUDA_HOME=str(prefix),CUDACXX=str(prefix/'bin/nvcc'),TORCH_CUDA_ARCH_LIST='9.0',MAX_JOBS='2',TORCH_EXTENSIONS_DIR=str(folder/'torch_extensions'))
os.environ['PATH']=str(prefix/'bin')+os.pathsep+os.environ['PATH']
import torch
from torch.utils.cpp_extension import load,CUDA_HOME
assert torch.__version__.split('+')[0]=='2.7.1' and torch.version.cuda=='12.8'
assert CUDA_HOME==str(prefix) and not torch.cuda.is_initialized()
version=subprocess.check_output([str(prefix/'bin/nvcc'),'--version'],text=True)
assert 'V12.8.93' in version,version
module=load(name='rsi_affine_grouped_load_cuda_v1',sources=[str(root/n) for n in ['affine_binding.cpp','affine_kernel.cu']],extra_cflags=manifest['cxx_flags'],extra_cuda_cflags=manifest['cuda_flags'],build_directory=str(build),verbose=True)
assert not torch.cuda.is_initialized()
binary=Path(module.__file__).resolve();assert binary.parent==build
report={'status':'COLD_COMPILED_NO_CUDA','nonce':nonce,'manifest_sha256':digest,'binary':str(binary),'binary_sha256':sha(binary),'cuda_flags':manifest['cuda_flags'],'cxx_flags':manifest['cxx_flags'],'cuda_initialized':False,'cold_build_seconds':time.monotonic()-started,'nvcc_version':version,'torch':torch.__version__,'torch_cuda':torch.version.cuda}
with (folder/'receipt.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report))
