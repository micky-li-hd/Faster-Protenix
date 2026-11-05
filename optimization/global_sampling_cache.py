"""Reuse verified global projection only inside a training no-grad sampler."""
import functools
import hashlib
import inspect
import textwrap
import torch
import projection_cache as cache


def install(model_cls, attention_cls, device_types=('cuda',)):
    original_sample = model_cls.sample_diffusion
    original_standard = attention_cls.standard_multihead_attention
    original_local = attention_cls.local_multihead_attention
    patched, source_hash = cache.patched_method(original_standard)
    stats = {'invocations': 0, 'hits': 0, 'misses': 0, 'bypasses': 0,
             'peak_output_bytes': 0, 'scope': 'training no-grad global attention only',
             'standard_source_sha256': source_hash,
             'local_source_sha256': hashlib.sha256(textwrap.dedent(inspect.getsource(original_local)).encode()).hexdigest()}

    @functools.wraps(original_standard)
    def standard(self, *args, **kwargs):
        if cache._scope.get() is None or torch.is_grad_enabled():
            return original_standard(self, *args, **kwargs)
        return patched(self, *args, **kwargs)

    @functools.wraps(original_sample)
    def sample(self, **kwargs):
        if not self.training or torch.is_grad_enabled() or kwargs.get('pair_z') is None:
            return original_sample(self, **kwargs)
        with cache.cache_scope(device_types) as state:
            try:
                return original_sample(self, **kwargs)
            finally:
                stats['invocations'] += 1
                for field in ('hits', 'misses', 'bypasses'):
                    stats[field] += state[field]
                stats['peak_output_bytes'] = max(stats['peak_output_bytes'], state['peak_output_bytes'])

    attention_cls.standard_multihead_attention = standard
    model_cls.sample_diffusion = sample
    assert attention_cls.local_multihead_attention is original_local

    def restore():
        model_cls.sample_diffusion = original_sample
        attention_cls.standard_multihead_attention = original_standard
        assert attention_cls.local_multihead_attention is original_local

    return restore, stats
