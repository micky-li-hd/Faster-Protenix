"""CPU import and source/fixture inventory; does not initialize CUDA."""
import hashlib,importlib.metadata,json,os,platform,sys,subprocess,ctypes
from pathlib import Path
root=Path('/root/tianbali/protenix_8h20_20261103_round31')
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['LAYERNORM_TYPE']='torch'
os.environ['PROTENIX_ROOT_DIR']=str(root/'assets/data')
sys.path.insert(0,str(root/'reference'))
import torch,biotite,rdkit,cuequivariance_torch
assert sys.version_info[:2]==(3,12),sys.version
assert torch.__version__.split('+')[0]=='2.7.1',torch.__version__
import runner.train
# Torch loads solver dynamically; CPU import alone did not cover it.
ctypes.CDLL("libcusolver.so.11")
assert not torch.cuda.is_initialized()
def inventory(base):
 result={}
 for p in sorted(base.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('._'):
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
   result[str(p.relative_to(base))]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
 return result
report={'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')},'source':inventory(root/'reference'),'fixture':inventory(root/'assets/data/researchboost_fixture'),'runtime_cuda_libraries':inventory(root/'cuda12_runtime'),'platform_abi_libraries':inventory(root/'platform_abi'),'torch_cuda_linkage':subprocess.check_output(['ldd',str(Path(torch.__file__).parent/'lib/libtorch_cuda.so')],text=True),'torch_import_path':torch.__file__,'gpu_initialized':torch.cuda.is_initialized(),'status':'CPU_IMPORTS_PASS'}
(root/'runtime_inventory.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['python','torch','torch_cuda','gpu_initialized','status']}))
