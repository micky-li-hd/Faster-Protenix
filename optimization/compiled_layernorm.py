"""Scoped deterministic LayerNorm compilation; Selective large-shape dispatch validated separately; native acceptance remains required."""


def install(layernorm_class, torch, backend_name='inductor'):
    from torch._dynamo.backends.registry import lookup_backend
    original = layernorm_class.forward
    compiler = lookup_backend(backend_name)
    stats = {'backend': backend_name, 'requested_torch_compile': True,
             'dynamic': True, 'fullgraph': True, 'calls': 0, 'eligible_calls': 0, 'bypassed_calls': 0, 'graphs': [],
             'dispatch': {'last_dimension': 128, 'minimum_rows': 147456},
             'scope': 'LayerNorm forward only; original dtype/epsilon/parameter behavior retained.'}

    def backend(graph, example_inputs):
        metadata = {'nodes': len(list(graph.graph.nodes)),
                    'inputs': [{'shape': [str(x) for x in value.shape], 'dtype': str(value.dtype),
                                'device': str(value.device)}
                               if isinstance(value, torch.Tensor) else {'type': type(value).__name__}
                               for value in example_inputs]}
        compiled = compiler(graph, example_inputs)
        stats['graphs'].append(metadata)
        return compiled

    # A cache overflow must fail rather than quietly switch candidate calls to eager.
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 256
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    compiled = torch.compile(original, backend=backend, fullgraph=True, dynamic=True)

    def forward(self, value):
        stats['calls'] += 1
        if not eligible(value.shape):
            stats['bypassed_calls'] += 1
            return original(self, value)
        stats['eligible_calls'] += 1
        return compiled(self, value)

    layernorm_class.forward = forward
    return stats


def eligible(shape):
    import math
    return len(shape) >= 2 and shape[-1] == 128 and math.prod(shape[:-1]) >= 147456
