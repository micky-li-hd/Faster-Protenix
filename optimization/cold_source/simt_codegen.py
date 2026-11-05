"""Check actual compiled PTX and SASS, including constant-only instruction operands."""
import subprocess,re,hashlib
from pathlib import Path

def audit(binary,folder):
 result={}
 for mode in ['ptx','sass']:
  text=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-'+mode,str(binary)],text=True)
  path=Path(folder)/('kernel.'+mode);path.write_text(text)
  constants=[]
  if mode=='ptx':
   assert len(re.findall(r'^\.visible \.entry .*attention_kernel_batched_impl',text,re.M))==2
   assert len(re.findall(r'^\.visible \.entry .*attention_kernel_backward_batched_impl',text,re.M))==1
   bad=re.findall(r'\b(?:mma\.[\w.]*|wgmma\.[\w.]*|tcgen05\.[\w.]*|[\w.]*\.tf32[\w.]*|[\w.]*\.bf16[\w.]*|[\w.]*\.f16[\w.]*)\b',text)
   ops=len(re.findall(r'\b(?:fma|mul|add)[\w.]*\.f32\b',text))
  else:
   bad=[];ops=0
   for line in text.splitlines():
    m=re.match(r'\s*/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][\w.]*)\s+(.*?);',line)
    if not m:continue
    op,args=m.groups()
    if op.split('.')[0] in ['FFMA','FMUL','FADD']:ops+=1
    if op.startswith(('HMMA','IMMA','BMMA','HGMMA','MMA','HFMA','HMUL','HADD')):
     if op=='HFMA2.MMA' and re.fullmatch(r'R\d+, -RZ, RZ, -?[\d.e+-]+, -?[\d.e+-]+\s*',args):constants.append(line.strip())
     else:bad.append(line.strip())
  result[mode]=dict(sha256=hashlib.sha256(text.encode()).hexdigest(),fp32_ops=ops,constant_only_half=constants,prohibited=bad,passed=ops>0 and not bad)
 result['passed']=all(result[x]['passed'] for x in ['ptx','sass']);assert result['passed'];return result
