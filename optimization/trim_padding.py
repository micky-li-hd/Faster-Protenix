"""Omit only wholly padded MSAStack chunks; retain original live-chunk shapes."""
import hashlib
import inspect
import textwrap

SOURCE_SHA256 = '876b29a323ebf5a9929be0b8efa6b45027108640784a250e3ce446e39ccae2eb'


def install():
    import protenix.model.modules.pairformer as module
    from pathlib import Path
    assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == SOURCE_SHA256
    original = module.MSAStack.forward
    source = textwrap.dedent(inspect.getsource(original))
    expression = 'self.msa_max_size - m.shape[-3]'
    assert source.count(expression) == 2
    source = source.replace(expression, '((m.shape[-3] + chunk_size - 1) // chunk_size) * chunk_size - m.shape[-3]')
    namespace = dict(original.__globals__)
    exec(compile(source, '<msa_wholly_padded_chunk_elision>', 'exec'), namespace)
    implementation = namespace['forward']
    stats = {'matched_calls': 0, 'omitted_rows_per_stage': 0, 'fallback_calls': 0}

    def forward(self, m, z):
        chunk = self.msa_chunk_size
        maximum = self.msa_max_size
        eligible = (self.training and m.ndim == 3 and isinstance(chunk, int)
                    and chunk > 0 and isinstance(maximum, int) and maximum > 0
                    and maximum % chunk == 0 and 0 < m.shape[-3] <= maximum)
        if not eligible:
            stats['fallback_calls'] += 1
            return original(self, m, z)
        target = ((m.shape[-3] + chunk - 1) // chunk) * chunk
        stats['matched_calls'] += 1
        stats['omitted_rows_per_stage'] += maximum - target
        return implementation(self, m, z)

    module.MSAStack.forward = forward

    def restore():
        assert module.MSAStack.forward is forward
        module.MSAStack.forward = original

    return stats, restore
