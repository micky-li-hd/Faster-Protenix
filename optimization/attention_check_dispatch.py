"""Metadata dispatch contract only; does not validate compiled CUDA math."""
from types import SimpleNamespace as S
from pathlib import Path
import json,copy,hashlib
from guarded_attention import eligible
cuda=S(type='cuda',index=0)
def tensor(shape):return S(shape=tuple(shape),ndim=len(shape),device=cuda,dtype='torch.float32',layout='torch.strided')
def inputs(local=False):
 shapes=[(48,16,384,48)]*3+[(1,16,384,384)]
 if local:shapes=[(48,4,87,32,32),(48,4,87,128,32),(48,4,87,128,32),(1,4,87,32,128)]
 return [tensor(s) for s in shapes]
opts=dict(efficient=True,inplace_safe=False,grad_enabled=True,autocast_enabled=False,tf32=False,precision='highest');rows=[]
def check(name,x,k,expected):
 actual=eligible(*x,**k);assert actual==expected,(name,actual,expected);rows.append({'case':name,'eligible':actual})
for local in [False,True]:
 label='local' if local else 'global';check(label,inputs(local),opts,True)
 for name,value in [('efficient',False),('inplace_safe',True),('grad_enabled',False),('autocast_enabled',True),('tf32',True),('precision','high')]:check(label+'_'+name,inputs(local),{**opts,name:value},False)
 for batch in [1,4]:
  x=inputs(local)
  for t in x[:3]:t.shape=(batch,*t.shape[1:])
  check(label+'_batch'+str(batch),x,opts,False)
 for i in range(4):
  x=inputs(local);x[i].dtype='torch.bfloat16';check(label+'_bf16_'+str(i),x,opts,False)
  x=inputs(local);x[i].device=S(type='cpu',index=None);check(label+'_cpu_'+str(i),x,opts,False)
 x=inputs(local);x[-1]=None;check(label+'_no_bias',x,opts,False)
 x=inputs(local);x[1].shape=(*x[1].shape[:-1],24);check(label+'_mismatch_k_v',x,opts,False)
 x=inputs(local);x[-1].shape=(2,*x[-1].shape[1:]);check(label+'_nonbroadcast_bias',x,opts,False)
x=[tensor((16,384,24)) for _ in range(3)]+[tensor((16,384,384))];check('native_pairformer3d_bypass',x,opts,False)
root=Path(__file__).resolve().parent
report={'status':'PASS_DISPATCH_METADATA_ONLY','cases':rows,'helper_sha256':hashlib.sha256((root/'guarded_attention.py').read_bytes()).hexdigest(),'accepted':False,'next_required':'Validate wrapper with native CUDA shape/grad/checkpoint paths, generated kernels and actual compilation options; native8 and128 RCR required before use.'}
(root/'DISPATCH_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'cases':len(rows)}))
