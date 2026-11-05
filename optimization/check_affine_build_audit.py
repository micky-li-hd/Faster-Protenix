"""Synthetic malformed-evidence tests; never claims a real CUDA build."""
import json,tempfile,hashlib,copy
from pathlib import Path
from affine_build_audit import verify

def dump(p,d):p.write_text(json.dumps(d))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
with tempfile.TemporaryDirectory() as t:
    root=Path(t);run=root/'run';folder=run/'affine/cold_layernorm';(folder/'build').mkdir(parents=True)
    (root/'affine').mkdir();source=root/'affine/source.txt';source.write_text('source');manifest={'sources':{'source.txt':sha(source)},'cuda_flags':['strict'],'cxx_flags':['strict']};dump(root/'affine/BUILD_MANIFEST.json',manifest);digest=sha(root/'affine/BUILD_MANIFEST.json')
    binary=folder/'build/mock.so';binary.write_text('mock');h=sha(binary);nonce='1'*64
    receipt={'nonce':nonce,'manifest_sha256':digest,'status':'COLD_COMPILED_NO_CUDA','cuda_initialized':False,'cuda_flags':['strict'],'cxx_flags':['strict'],'cold_build_seconds':3,'binary':str(binary),'binary_sha256':h}
    dump(folder/'receipt.json',receipt);dump(folder/'started.json',{'nonce':nonce,'manifest_sha256':digest});(folder/'compiler.log').write_text('synthetic')
    rows=[{'enabled':True,'compiled_this_rank':rank==3,'binary_sha256':h,'manifest_sha256':digest,'compile_and_wait_seconds':4,'calls':{'eligible_backward':10,'native_backward_fallback':1,'eligible_width64':4,'eligible_width128':6}} for rank in range(8)]
    def save_rows():
        for rank,row in enumerate(rows):dump(run/f'rank{rank}.affine_stats.json',row)
    save_rows();assert verify(root,run,True,nonce)['status']=='ONE_COLD_BUILD_ALL8_RANKS_VERIFIED';checks.append('valid synthetic evidence accepted')
    def reject(name,fn):
        try:fn()
        except AssertionError:checks.append(name)
        else:raise AssertionError(name+' accepted')
    reject('foreign nonce rejected',lambda:verify(root,run,True,'2'*64))
    original=copy.deepcopy(rows)
    for name,key,value in [('cross-rank binary mismatch','binary_sha256','0'*64),('multiple builders','compiled_this_rank',True),('missing eligible calls','calls',{'eligible_backward':0,'native_backward_fallback':1}),('missing width128 calls','calls',{'eligible_backward':4,'native_backward_fallback':1,'eligible_width64':4,'eligible_width128':0}),('bad width sum','calls',{'eligible_backward':11,'native_backward_fallback':1,'eligible_width64':4,'eligible_width128':6})]:
        rows=copy.deepcopy(original);rows[0][key]=value;save_rows();reject(name,lambda:verify(root,run,True,nonce))
    rows=original;save_rows();binary.write_text('corrupt');reject('binary corruption',lambda:verify(root,run,True,nonce));binary.write_text('mock')
    receipt['cold_build_seconds']=5;dump(folder/'receipt.json',receipt);reject('compile time excluded from builder accounting',lambda:verify(root,run,True,nonce))
    disabled=root/'disabled';disabled.mkdir()
    for rank in range(8):dump(disabled/f'rank{rank}.affine_stats.json',{'enabled':False})
    assert verify(root,disabled,False,nonce)['status']=='NO_NATIVE_BUILD_EXPECTED';checks.append('disabled arm requires no build')
report={'status':'SYNTHETIC_NATIVE_BUILD_AUDIT_PASS','checks':checks,'accepted':False,'scope':'Evidence auditing only; no CUDA compilation or training.','audit_sha256':sha(Path(__file__).with_name('affine_build_audit.py'))};Path(__file__).with_name('AFFINE_BUILD_AUDIT_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
