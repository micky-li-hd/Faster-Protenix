"""Persist and independently verify compiled attention source, without executing it."""
import hashlib,json
from pathlib import Path
FORBIDDEN=('_scaled_dot_product','efficient_attention','flash_attention','cudnn_attention','OpMultiplyAddFastF32','tl.dot(')
def inspect_source(text):
 return {'external_bmm':text.count('extern_kernels.bmm'),'external_mm':text.count('extern_kernels.mm'),'forbidden':[s for s in FORBIDDEN if s in text]}
def capture(out,rank,enabled):
 out=Path(out);sources=[]
 for p in sorted((out/'cache'/f'rank{rank}').rglob('*.py')):
  text=p.read_text();rel=p.relative_to(out/'cache'/f'rank{rank}');dest=out/'attention_sources'/f'rank{rank}'/rel
  dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
  sources.append({'path':str(rel),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),**inspect_source(text)})
 report={'requested_torch_compile':enabled,'sources':sources}
 (out/f'rank{rank}.attention_codegen.json').write_text(json.dumps(report,indent=2)+'\n')
 verify(out,rank,enabled)
def verify(out,rank,enabled):
 out=Path(out);report=json.loads((out/f'rank{rank}.attention_codegen.json').read_text());assert report['requested_torch_compile']==enabled
 if enabled:assert report['sources'] and sum(s['external_bmm']+s['external_mm'] for s in report['sources'])>0
 else:assert not report['sources']
 for s in report['sources']:
  path=Path(s['path']);assert not path.is_absolute() and '..' not in path.parts
  p=out/'attention_sources'/f'rank{rank}'/path;assert hashlib.sha256(p.read_bytes()).hexdigest()==s['sha256']
  actual=inspect_source(p.read_text());assert all(s[k]==v for k,v in actual.items()) and not actual['forbidden']
 return report
