from pathlib import Path
import subprocess,sys,json,hashlib
root=Path(__file__).resolve().parent;cp=root/'combined_cpu'
for src,dst in [('combined_layernorm.py','combined_layernorm.py'),('ordered_affine.py','ordered_affine.py'),('isolated_layernorm.py','isolated_layernorm.py'),('hybrid_layernorm.py','hybrid_layernorm.py'),('guarded_layernorm.py','combined_guarded_layernorm.py')]:assert (cp/src).read_bytes()==(root/dst).read_bytes()
for name in ['check_combined_cpu.py','check_checkpoint_cpu.py']:subprocess.run([sys.executable,str(cp/name)],check=True)
a=json.loads((cp/'COMBINED_CPU_CHECK.json').read_text());b=json.loads((cp/'CHECKPOINT_CPU_CHECK.json').read_text());assert a['cases']==48 and b['count']==40 and b['broken_original_reproduced']
(root/'COMBINED_ADAPTER_CHECK.json').write_text(json.dumps({'status':'PASS','adapter_cases':48,'checkpoint_cases':40,'source_sha256':hashlib.sha256((root/'combined_layernorm.py').read_bytes()).hexdigest(),'accepted':False},indent=2)+'\n')
