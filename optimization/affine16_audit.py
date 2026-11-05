import json,hashlib
from pathlib import Path

def verify_sources(root):
    root=Path(root);proof=json.loads((root/'operator_evidence/affine16_bindings.json').read_text());digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for n in ['affine_kernel.cu','affine_binding.cpp','BUILD_MANIFEST.json']:assert digest(root/'affine'/n)==proof[n],n
    old=root/'operator_evidence/affine16_guarded_gpu.py';assert digest(old)==proof['guarded_affine16.py']
    expected=old.read_text().replace("previous=module.__dict__.get('forward',missing);fallback=module.forward","previous=module.__dict__.get('forward',missing)\n        assert previous is missing, 'Refuse to replace an existing instance forward'\n        fallback=cls.forward").replace('return fallback(x)','return fallback(self,x)')
    expected=expected.replace('records=[];missing=object()','records=[];missing=object();active=[True]').replace('        def forward(self,x):\n','        def forward(self,x):\n            if not active[0]:return fallback(self,x)\n').replace('        cls.__init__=original_init','        active[0]=False\n        cls.__init__=original_init')
    assert (root/'guarded_affine16.py').read_text()==expected
    review=json.loads((root/'operator_evidence/affine16_DECISION_REVIEW.json').read_text());assert review['status']=='AFFINE16_COMPOSED_MODULE_EXACT_PASS_CLEAN' and review['candidate_tensor_comparisons']==1152 and review['parent_affine_cases']==96
    prefix=(root/'affine/affine_kernel.cu').read_text().split('\nnamespace width16 {\n')[0]
    assert hashlib.sha256(prefix.encode()).hexdigest()==json.loads((root/'operator_evidence/affine16_PREPARATION_REVIEW.json').read_text())['parent_sha256']
    return {'gpu_kernel_sources_exact':True,'adapter_change':'Unbound fallback dispatch on current self for EMA/deepcopy safety; reject preexisting instance overrides.'}

def verify_stats(stats,enabled):
    if not enabled:
        assert stats=={'enabled':False};return
    assert stats['enabled'] is True and stats['restored'] is True
    for k in ['eligible','fallback','installed']:assert type(stats[k]) is int and stats[k]>=0
    assert stats['eligible']>0 and stats['installed']>0
