"""Actual native Attention rewrite: output/all gradients/RNG and restoration."""
import os, sys, json, hashlib
from pathlib import Path
root = Path(__file__).resolve().parent
reference = root/'reference'
for name, item in json.loads((root/'expected_source_fixture.json').read_text())['source'].items():
    assert hashlib.sha256((reference/name).read_bytes()).hexdigest() == item['sha256'], name
sys.path.insert(0, str(reference))
os.environ['LAYERNORM_TYPE'] = 'torch'
import torch
from torch.utils.checkpoint import checkpoint
from protenix.model.triangular.layers import Attention
from guarded_head_gate_backward import install
assert not torch.cuda.is_initialized()
torch.set_num_threads(2)
cases = []
for dtype in [torch.float32, torch.bfloat16]:
    for gating in [False, True]:
        for heads in [4, 8]:
            for ckpt in [False, True]:
                torch.manual_seed(20261027)
                model = Attention(128, 128, 128, 32, heads, gating=gating).to(dtype)
                with torch.no_grad():
                    model.linear_o.weight.normal_(0, .02)
                    if gating: model.linear_g.weight.normal_(0, .02)
                raw = torch.randn(3, heads, 7, 32, dtype=dtype)
                qraw = torch.randn(3, 7, 128, dtype=dtype)
                up = torch.randn(3, 128, 7, dtype=dtype).transpose(1, 2)
                params = [model.linear_o.weight] + ([model.linear_g.weight] if gating else [])
                initial = [p.detach().clone() for p in params]
                state = torch.get_rng_state().clone()

                def run():
                    o = raw.clone().transpose(1, 2).requires_grad_()
                    q = qraw.clone().requires_grad_()
                    torch.set_rng_state(state)
                    y = checkpoint(model._wrap_up, o, q, use_reentrant=False) if ckpt else model._wrap_up(o, q)
                    grads = torch.autograd.grad(y, [o, q, *params], up, allow_unused=True)
                    return [y.detach().clone(), *[g.detach().clone() if g is not None else None for g in grads]], torch.get_rng_state().clone()

                baseline, rng = run()
                original = Attention._wrap_up
                restore, stats = install(Attention)
                try:
                    candidate, rng2 = run()
                    assert len(baseline) == len(candidate)
                    for a, b in zip(baseline, candidate):
                        assert (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
                    assert torch.equal(rng, rng2)
                    assert stats['eligible_calls'] == 0
                    assert (stats['fallback_calls'] > 0) == gating
                finally:
                    restore()
                assert Attention._wrap_up is original
                assert all(torch.equal(p, v) for p, v in zip(params, initial))
                cases.append({'dtype': str(dtype), 'gating': gating, 'heads': heads,
                              'checkpoint': ckpt, 'output_all_gradients_rng_exact': True,
                              'restored': True, 'stats': stats})
assert not torch.cuda.is_initialized()
report = {'status': 'HEAD_GATE_NATIVE_CPU_INTEGRATION_PASS', 'cases': cases,
          'accepted': False, 'scope': 'Actual rewritten method with CPU fallback; fused CUDA path separately tested, native8/128 acceptance still required.'}
(root/'HEAD_GATE_BACKWARD_CPU_CHECK.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'status': report['status'], 'cases': len(cases)}))
