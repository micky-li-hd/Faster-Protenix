"""Read-only audit for future native cold-build integration; no GPU imports."""
import hashlib,json,math
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify(root,run,enabled,nonce):
    root,run=Path(root),Path(run)
    rows=[json.loads((run/f'rank{rank}.hybrid_layernorm_stats.json').read_text()) for rank in range(8)]
    folder=run/'cold_layernorm'
    if not enabled:
        assert all(row=={'enabled':False} for row in rows)
        assert not folder.exists()
        return {'enabled':False,'status':'NO_NATIVE_BUILD_EXPECTED'}
    manifest_path=root/'BUILD_MANIFEST.json';manifest=json.loads(manifest_path.read_text());digest=sha(manifest_path)
    for n,h in manifest['sources'].items():assert sha(root/n)==h,n
    receipt=json.loads((folder/'receipt.json').read_text());started=json.loads((folder/'started.json').read_text())
    assert started['nonce']==receipt['nonce']==nonce
    assert started['manifest_sha256']==receipt['manifest_sha256']==digest
    assert receipt['status']=='COLD_COMPILED_NO_CUDA' and receipt['cuda_initialized'] is False
    assert receipt['cuda_flags']==manifest['cuda_flags'] and receipt['cxx_flags']==manifest['cxx_flags']
    seconds=receipt['cold_build_seconds'];assert math.isfinite(seconds) and seconds>0
    binary=Path(receipt['binary']).resolve();assert binary.parent==(folder/'build').resolve()
    binary_hash=sha(binary);assert binary_hash==receipt['binary_sha256']
    assert (folder/'compiler.log').is_file()
    for row in rows:
        assert row['enabled'] is True and type(row['compiled_this_rank']) is bool
        assert row['binary_sha256']==binary_hash and row['manifest_sha256']==digest
        assert math.isfinite(row['compile_and_wait_seconds']) and row['compile_and_wait_seconds']>=0
        assert row['calls']['eligible_calls']>0 and row['calls']['bypassed_calls']>=0
    assert sum(row['compiled_this_rank'] for row in rows)==1
    builder=next(row for row in rows if row['compiled_this_rank']);assert builder['compile_and_wait_seconds']>=seconds
    return {'enabled':True,'status':'ONE_COLD_BUILD_ALL8_RANKS_VERIFIED','binary_sha256':binary_hash,'manifest_sha256':digest,'cold_build_seconds':seconds,'eligible_calls_by_rank':[row['calls']['eligible_calls'] for row in rows]}
