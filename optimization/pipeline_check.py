"""Exercise code-generation evidence verification with corruption and forbidden backends."""
import json,tempfile,hashlib
from pathlib import Path
from attention_codegen_audit import capture,verify
root=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as d:
 r=Path(d);p=r/'cache/rank0/ok.py';p.parent.mkdir(parents=True);p.write_text('extern_kernels.bmm(a,b)\n')
 capture(r,0,True);verify(r,0,True)
 s=r/'attention_sources/rank0/ok.py';s.write_text('changed')
 try:verify(r,0,True)
 except AssertionError:pass
 else:raise AssertionError('Corruption accepted')
 p.write_text('extern_kernels.bmm(a,b)\ntl.dot(a,b)\n')
 try:capture(r,0,True)
 except AssertionError:pass
 else:raise AssertionError('Forbidden kernel accepted')
 capture(r,1,False)
 try:verify(r,1,True)
 except AssertionError:pass
 else:raise AssertionError('Wrong compile declaration accepted')
report={'status':'PASS','checks':['valid bmm source verified','modified generated file rejected','forbidden kernel rejected','disabled policy empty cache accepted','wrong compile declaration rejected']}
(root/'PIPELINE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
