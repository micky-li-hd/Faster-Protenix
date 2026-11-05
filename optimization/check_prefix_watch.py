from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parent;subprocess.run([sys.executable,str(root/'analysis/check_queue_readiness.py')],check=True)
r=json.loads((root/'analysis/QUEUE_READINESS_CHECK.json').read_text());assert r['status']=='PASS' and r['count']==19
(root/'PREFIX_WATCH_CHECK.json').write_text(json.dumps(r,indent=2)+'\n')
