"""Local black-box process/locking proof with a synthetic compiler; no CUDA claim."""
import concurrent.futures,json,os,secrets,subprocess,sys,tempfile,shutil,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
fake='''import json,os,sys,time
from pathlib import Path
from cold_build import checked_manifest,sha,NONCE
r=Path(__file__).resolve().parent;folder=Path(sys.argv[1]);m,d=checked_manifest(r)
assert os.environ['CUDA_VISIBLE_DEVICES']==''
with (folder/'compiler_invocation').open('x') as f:f.write('one child')
time.sleep(.2)
b=folder/'build';b.mkdir();p=b/'mock.so';p.write_bytes(b'mock binary not CUDA')
data={'status':'COLD_COMPILED_NO_CUDA','nonce':os.environ[NONCE],'manifest_sha256':d,'binary':str(p),'binary_sha256':sha(p),'cuda_flags':m['cuda_flags'],'cxx_flags':m['cxx_flags'],'cuda_initialized':False}
(folder/'receipt.json').write_text(json.dumps(data))
'''
worker='''import os,json,sys
from cold_build import ensure_binary
before=dict(os.environ);p,d,s=ensure_binary(sys.argv[1]);assert dict(os.environ)==before
print(json.dumps(s))
'''
with tempfile.TemporaryDirectory() as t:
    r=Path(t);shutil.copy2(root/'cold_build.py',r/'cold_build.py');(r/'compile_child.py').write_text(fake);(r/'worker.py').write_text(worker)
    (r/'source.txt').write_text('source');source_hash=hashlib.sha256((r/'source.txt').read_bytes()).hexdigest()
    (r/'BUILD_MANIFEST.json').write_text(json.dumps({'sources':{'source.txt':source_hash},'cuda_flags':['strict'],'cxx_flags':['strict']}))
    out=r/'run';out.mkdir();env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7','PROTENIX_RESEARCHBOOST_LAUNCH_NONCE':os.environ.get('PROTENIX_RESEARCHBOOST_LAUNCH_NONCE',secrets.token_hex(32))}
    def run(rank):
        p=subprocess.run([sys.executable,str(r/'worker.py'),str(out)],env={**env,'RANK':str(rank)},text=True,capture_output=True,timeout=20)
        assert p.returncode==0,p.stderr
        return json.loads(p.stdout)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(run,range(8)))
    assert sum(x['compiled_this_rank'] for x in rows)==1
    assert len({x['binary_sha256'] for x in rows})==1
    failures=[]
    def rejected(label,runout,runenv):
        p=subprocess.run([sys.executable,str(r/'worker.py'),str(runout)],env=runenv,text=True,capture_output=True,timeout=20)
        assert p.returncode!=0,label;failures.append({'case':label,'rejected':True})
    receipt_path=out/'cold_layernorm/receipt.json';valid_receipt=json.loads(receipt_path.read_text());bad_receipt={**valid_receipt,'nonce':secrets.token_hex(32)};receipt_path.write_text(json.dumps(bad_receipt))
    rejected('foreign nonce receipt',out,env)
    receipt_path.write_text(json.dumps(valid_receipt))
    (out/'cold_layernorm/build/mock.so').write_bytes(b'corrupt')
    rejected('binary corruption',out,env)
    failed=r/'failed_run';(failed/'cold_layernorm').mkdir(parents=True);(failed/'cold_layernorm/started.json').write_text('{}')
    rejected('interrupted build must not rebuild',failed,env)
    (r/'source.txt').write_text('changed source')
    rejected('source mismatch',out,env)
report={'status':'PASS_SYNTHETIC_CPU_BUILD_PROTOCOL','compiler_calls':1,'workers':8,'rows':rows,'rejected_cases':failures,'scope':'Synthetic compiler protocol only; no CUDA compilation, model, kernel or timing acceptance.','accepted':False,'adapter_sha256':hashlib.sha256((root/'cold_build.py').read_bytes()).hexdigest()}
(root/'BUILD_PROTOCOL_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
