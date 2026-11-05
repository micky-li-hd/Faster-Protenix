import os,sys,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent;out=Path(os.environ['RSI_OUT']);processes=[];logs=[]
try:
 for rank in range(8):
  log=(out/f'worker{rank}.log').open('x');logs.append(log)
  processes.append(subprocess.Popen([sys.executable,str(root/'worker_check.py'),str(rank)],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT))
 codes=[p.wait(timeout=800) for p in processes];assert codes==[0]*8,codes
 rows=[json.loads((out/f'worker{k}.json').read_text()) for k in range(8)]
 assert sum(x['timing']['compiled_this_rank'] for x in rows)==1
 assert len({x['binary_sha256'] for x in rows})==1 and len({x['nonce'] for x in rows})==1
 assert all(x['codegen_passed'] and x['cuda_initialized'] is False for x in rows)
 report=dict(status='EIGHT_WORKERS_ONE_COLD_CPU_COMPILATION_PASS',accepted=False,workers=rows,scope='Compiler lifecycle and actual generated precision audit only. No GPU arithmetic or native full-run cold timing acceptance.')
 (out/'CONCURRENCY_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
finally:
 for log in logs:log.close()
