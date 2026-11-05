"""One per-run build in a nonce-inheriting CPU child; training workers only load."""
import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
NONCE='PROTENIX_RESEARCHBOOST_LAUNCH_NONCE'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def checked_manifest(root):
    path=root/'BUILD_MANIFEST.json';manifest=json.loads(path.read_text())
    for name,digest in manifest['sources'].items():assert sha(root/name)==digest,name
    return manifest,sha(path)

def ensure_binary(output):
    started=time.monotonic();root=Path(__file__).resolve().parent
    manifest,digest=checked_manifest(root)
    nonce=os.environ[NONCE];assert len(nonce)==64
    parent_env=dict(os.environ);folder=Path(output).resolve()/'cold_simt';folder.mkdir(exist_ok=True)
    receipt=folder/'receipt.json';did_compile=False
    with (folder/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not receipt.exists():
            # An interrupted or failed build is evidence; never silently rebuild it.
            with (folder/'started.json').open('x') as f:json.dump({'nonce':nonce,'manifest_sha256':digest,'pid':os.getpid()},f)
            child_env={**parent_env,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','MAX_JOBS':'2'}
            with (folder/'compiler.log').open('x') as log:
                subprocess.run([sys.executable,str(root/'simt_compile_child.py'),str(folder)],env=child_env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            did_compile=True
        data=json.loads(receipt.read_text())
        assert data['status']=='COLD_COMPILED_NO_CUDA'
        assert data['nonce']==nonce and data['manifest_sha256']==digest
        binary=Path(data['binary']).resolve();assert binary.parent==folder/'build'
        assert sha(binary)==data['binary_sha256']
        assert data['cuda_flags']==manifest['cuda_flags'] and data['cxx_flags']==manifest['cxx_flags']
        assert data['cuda_initialized'] is False
        assert data['codegen']['passed']
        for mode in ['ptx','sass']:assert sha(folder/('kernel.'+mode))==data['codegen'][mode]['sha256']
    assert dict(os.environ)==parent_env,'CPU compiler mutated training environment'
    return binary,data,{'compiled_this_rank':did_compile,'compile_and_wait_seconds':time.monotonic()-started,'manifest_sha256':digest,'binary_sha256':data['binary_sha256']}
