import os,sys,json,subprocess,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent;out=root/'dual_cpu_run';out.mkdir();procs=[]
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and len(os.environ['PROTENIX_RESEARCHBOOST_LAUNCH_NONCE'])==64
for rank in range(8):
 with (out/f'rank{rank}.log').open('x') as log:procs.append(subprocess.Popen([sys.executable,str(root/'dual_proof_worker.py'),str(out),str(rank)],stdout=log,stderr=subprocess.STDOUT))
for p in procs:assert p.wait(timeout=600)==0,(p.pid,p.returncode)
rows=[json.loads((out/f'rank{rank}.json').read_text()) for rank in range(8)]
assert all(r['status']=='DUAL_CPU_IMPORT_FALLBACK_PASS' and r['rng_exact'] and r['environment_unchanged'] and r['parameter_identity_preserved'] and not r['cuda_initialized'] for r in rows)
for key,folder in [('forward',out/'cold_layernorm'),('affine',out/'affine/cold_layernorm')]:
 assert sum(r[key]['compiled_this_rank'] for r in rows)==1 and len({r[key]['binary_sha256'] for r in rows})==1
 receipt=json.loads((folder/'receipt.json').read_text());assert receipt['nonce']==os.environ['PROTENIX_RESEARCHBOOST_LAUNCH_NONCE'] and receipt['status']=='COLD_COMPILED_NO_CUDA' and not receipt['cuda_initialized']
 assert hashlib.sha256(Path(receipt['binary']).read_bytes()).hexdigest()==receipt['binary_sha256']
 assert all(r[key]['binary_sha256']==receipt['binary_sha256'] for r in rows)
expected=json.loads((root/'expected_source_fixture.json').read_text())['source']
for name,entry in expected.items():
 data=(root/'reference'/name).read_bytes();assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'],name
report={'source_files_unchanged_after_dual_build':len(expected),'status':'REAL_DUAL_COLD_BUILD_8_WORKER_CPU_PASS','rows':rows,'scope':'Two actual isolated coldcompiles,8imports,RNG/environment/parameteridentity andCPUfallback;noGPUtraining claim.','accepted':False};(root/'DUAL_COLD_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'workers':8,'compiles':2}))
