"""Future native integration: charge two per-run cold builds before patching."""
from pathlib import Path

def install(output,layernorm_class):
 from cold_build import ensure_binary as forward_binary
 from affine.cold_build import ensure_binary as affine_binary
 from isolated_layernorm import load_verified_extension as load_forward
 from ordered_affine import load_verified_extension as load_affine
 from combined_guarded_layernorm import install as patch
 from combined_layernorm import _STATS
 output=Path(output);affine_out=output/'affine';affine_out.mkdir(exist_ok=True)
 forward,forward_receipt,forward_stats=forward_binary(output)
 affine,affine_receipt,affine_stats=affine_binary(affine_out)
 load_forward(forward,forward_receipt['binary_sha256'])
 load_affine(affine,affine_receipt['binary_sha256'])
 calls=patch(layernorm_class)
 return {'enabled':True,**forward_stats,'calls':calls},{'enabled':True,**affine_stats,'calls':_STATS}
