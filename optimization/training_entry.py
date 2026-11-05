"""Separately receipted FP32 bias-softmax adapter around immutable native entry."""
import os,sys,json,runpy,hashlib
from pathlib import Path
prior=Path('/root/tianbali/protenix_8h20_20261103_round31')
for n,h in json.loads((Path(__file__).resolve().parent/'PRIOR_BINDINGS.json').read_text()).items():assert hashlib.sha256((prior/n).read_bytes()).hexdigest()==h,n
assert os.environ['LAYERNORM_TYPE']=='torch'
assert os.environ['RSI_BIAS_SOFTMAX'] in ('0','1')
# Native configs read the data-root environment on first import.
os.environ['PROTENIX_ROOT_DIR']=json.loads((prior/'recipe.json').read_text())['data_root']
sys.path[:0]=[str(prior/'reference'),str(prior)]
import torch
import protenix.model.modules.primitives as primitives
from guarded_bias_softmax import install
from guarded_ema import install as install_ema
import runner.ema as ema_module
enabled=os.environ['RSI_BIAS_SOFTMAX']=='1'
policy=json.loads((prior/'performance_policies.json').read_text())[os.environ['RSI_POLICY']]
assert not policy.get('_compile_attention',False), 'Do not stack attention adapters'
restore=None;stats={'enabled':False}
if enabled:restore,stats=install(primitives,torch)
from guarded_simt import install as install_simt
assert os.environ['RSI_SIMT_ATTENTION'] in ('0','1')
simt_restore=None;simt_stats={'enabled':False}
if os.environ['RSI_SIMT_ATTENTION']=='1':
    assert enabled
    from adapter import attention as simt_attention
    simt_restore,simt_stats=install_simt(primitives,torch,simt_attention)
ema_restore=None;ema_stats={'enabled':False}
assert os.environ['RSI_FOREACH_EMA'] in ('0','1')
if os.environ['RSI_FOREACH_EMA']=='1':ema_restore,ema_stats=install_ema(ema_module,torch)
import runner.train as training
from guarded_init import install as install_init
assert os.environ['RSI_PARALLEL_INIT'] in ('0','1')
init_restore,init_stats=install_init(training.AF3Trainer,torch,os.environ['RSI_PARALLEL_INIT']=='1',
    json.loads((Path(__file__).resolve().parent/'INIT_BINDING.json').read_text())['init_model_sha256'],
    json.loads((Path(__file__).resolve().parent/'PPF_BINDING.json').read_text())['original_ppf_sha256'])
metadata_restore=None;metadata_stats={'enabled':False}
assert os.environ['RSI_PERMUTATION_METADATA'] in ('0','1')
if os.environ['RSI_PERMUTATION_METADATA']=='1':
    from protenix.utils.permutation import atom_permutation as permutation_module
    import metadata_collect
    expected=json.loads((Path(__file__).resolve().parent/'CPU_CHECK.json').read_text())['function_sha256']
    assert metadata_collect.verify_source(permutation_module.collect_residues_with_symmetric_atoms)==expected
    metadata_restore,metadata_stats=metadata_collect.install(permutation_module)
    metadata_stats['enabled']=True
try:
    runpy.run_path(str(prior/'native_entry.py'),run_name='__main__')
finally:
    if metadata_restore is not None:metadata_restore()
    (Path(os.environ['RSI_OUT'])/f'rank{os.environ["RANK"]}.permutation_metadata.json').write_text(json.dumps(metadata_stats,indent=2)+"\n")
    init_restore()
    (Path(os.environ['RSI_OUT'])/f'rank{os.environ["RANK"]}.parallel_init.json').write_text(json.dumps(init_stats,indent=2)+'\n')
    if ema_restore is not None:ema_restore()
    if simt_restore is not None:simt_restore()
    if restore is not None:restore()
    out=Path(os.environ['RSI_OUT']);rank=int(os.environ['RANK'])
    (out/f'rank{rank}.bias_softmax.json').write_text(json.dumps(stats,indent=2)+'\n')
    (Path(os.environ['RSI_OUT'])/f'rank{os.environ["RANK"]}.foreach_ema.json').write_text(json.dumps(ema_stats,indent=2)+'\n')
if enabled:assert stats['restored'] and stats['eligible_calls']>0

if ema_restore is not None:assert ema_stats['restored'] and ema_stats['eligible_calls']==int(os.environ['RSI_STEPS']) and ema_stats['bypassed_calls']==0

(Path(os.environ['RSI_OUT'])/f'rank{os.environ["RANK"]}.simt.json').write_text(json.dumps(simt_stats,indent=2)+'\n')
if simt_restore is not None:assert simt_stats['restored'] and simt_stats['eligible_calls']>0

if metadata_restore is not None:assert metadata_stats['restored'] and metadata_stats['eligible']>0
