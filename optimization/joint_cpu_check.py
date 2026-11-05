"""Joint retention through real48-block Pairformer and24-block DiffusionTransformer."""
import os,sys,json,hashlib,copy,time,platform
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['LAYERNORM_TYPE']='torch'
r=Path(__file__).resolve().parent
for n,h in json.loads((r/'JOINT_SOURCE_BINDINGS.json').read_text()).items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
sys.path.insert(0,str(r/'reference'))
import torch
import torch.utils.checkpoint
import protenix.model.modules.pairformer as pair_module
import protenix.model.modules.transformer as diff_module
from partial_pair_checkpoint import install as install_pair
from partial_diffusion_checkpoint import install as install_diff
assert torch.__version__.split('+')[0]=='2.7.1';torch.set_num_threads(2);torch.manual_seed(20261101)
pair=pair_module.PairformerStack(n_blocks=48,n_heads=4,c_z=16,c_s=32,dropout=.25,blocks_per_ckpt=1).train()
diff=diff_module.DiffusionTransformer(c_a=32,c_s=32,c_z=16,n_blocks=24,n_heads=4,drop_path_rate=.1,blocks_per_ckpt=1).train()
model=torch.nn.ModuleDict({'pair':pair,'diff':diff})
with torch.no_grad():
 for p in model.parameters():
  if p.ndim>=2 and torch.count_nonzero(p)==0:p.normal_(0,.015)
values=[torch.randn(4,32),torch.randn(4,4,16),torch.randn(2,4,32)];state=copy.deepcopy(model.state_dict());reference=None;rows=[]
original_pair=pair_module.checkpoint_blocks;original_diff=diff_module.checkpoint_blocks
try:
 for pair_prefix,diff_prefix in [(0,0),(8,24),(10,24),(12,24)]:
  pair_module.checkpoint_blocks=original_pair;diff_module.checkpoint_blocks=original_diff
  pair_stats=install_pair(pair_prefix);pair_wrapper=pair_module.checkpoint_blocks
  assert diff_module.checkpoint_blocks is original_diff
  diff_stats=install_diff(diff_prefix);assert pair_module.checkpoint_blocks is pair_wrapper
  model.load_state_dict(state);model.zero_grad(set_to_none=True)
  s,z,a=[v.clone().requires_grad_(True) for v in values];torch.manual_seed(24680)
  ps,pz=pair(s,z,pair_mask=None,triangle_attention='torch',triangle_multiplicative='torch')
  output=diff(a,ps.unsqueeze(0).expand(2,-1,-1),pz.unsqueeze(0))
  loss=output.square().mean()+.1*ps.square().mean()+.1*pz.square().mean();loss.backward()
  tensors={'pair_s':ps.detach(),'pair_z':pz.detach(),'diffusion_output':output.detach(),'input_s_grad':s.grad,'input_z_grad':z.grad,'input_a_grad':a.grad,**{n+'_grad':p.grad for n,p in model.named_parameters() if p.grad is not None}}
  assert all(torch.isfinite(v).all() for v in tensors.values());rng=torch.get_rng_state().clone()
  if reference is None:reference=({n:v.clone() for n,v in tensors.items()},rng)
  assert tensors.keys()==reference[0].keys()
  differences=[n for n,v in tensors.items() if not torch.equal(v,reference[0][n])];rng_exact=torch.equal(rng,reference[1])
  assert pair_stats['matched_gradient_calls']==int(pair_prefix>0)
  assert diff_stats['matched_gradient_calls']==int(diff_prefix>0) and diff_stats['retained_block_calls']==diff_prefix
  row={'pair_prefix':pair_prefix,'diffusion_prefix':diff_prefix,'tensor_count':len(tensors),'differences':differences,'rng_exact':rng_exact,'pair_stats':pair_stats,'diffusion_stats':diff_stats};rows.append(row)
  (r/'JOINT_CPU_PROGRESS.json').write_text(json.dumps(rows,indent=2)+'\n');assert not differences and rng_exact,row
 assert not torch.cuda.is_initialized()
 report={'status':'PASS_LOCAL_CPU_JOINT_RETENTION','torch':torch.__version__,'platform':platform.platform(),'cases':rows,'cuda_initialized':False,'accepted':False,'scope':'Actual48+24block connected small-channel CPU models, nonzero output projections, pair dropout and diffusion stochastic depth. Does not prove native-width CUDA correctness, memory, speed or128step acceptance.'}
 (r/'JOINT_CPU_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'cases':len(rows),'tensors_per_case':len(tensors)}))
finally:pair_module.checkpoint_blocks=original_pair;diff_module.checkpoint_blocks=original_diff
