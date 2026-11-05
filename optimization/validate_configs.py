"""Use upstream main parser but replace trainer with a CPU config capture stub."""
import os,sys,json,copy
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['LAYERNORM_TYPE']='torch';os.environ['PROTENIX_ROOT_DIR']=str(root/'assets/data')
sys.path.insert(0,str(root/'reference'))
import torch
import runner.train as training
from recipe import argv,plain
policies=json.loads((root/'performance_policies.json').read_text());resolved={}
class Capture:
 def __init__(self,config):self.config=config
 def run(self):resolved[name]=plain(self.config)
training.AF3Trainer=Capture
for name,policy in policies.items():
 spec=json.loads((root/'recipe.json').read_text());overrides={k:v for k,v in policy.items() if not k.startswith('_')}
 spec['overrides'].update(max_steps=128,log_interval=1,enable_efficient_fusion=False);spec['overrides'].update(overrides)
 sys.argv=['train.py']+argv(spec,root/'config_validation_output',20261027)
 training.main()
 assert resolved[name]['model']['pairformer']['n_blocks']==48
 assert resolved[name]['model']['pairformer']['blocks_per_ckpt']==1
 expected=copy.deepcopy(resolved['reference'])
 for key,value in overrides.items():
  node=expected;parts=key.split('.')
  for part in parts[:-1]:node=node[part]
  node[parts[-1]]=value
 assert resolved[name]==expected, name
assert not torch.cuda.is_initialized()
report={'status':'PASS','policies':list(policies),'gpu_initialized':False,'scope':'Upstream parser resolved only declared configuration differences; actual GPU behavior still untested.'}
(root/'CONFIG_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
