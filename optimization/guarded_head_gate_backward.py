"""Scoped replacement of native Attention gate and head flattening only."""
import ast
import hashlib
import inspect
import textwrap


def rewrite(source):
    tree = ast.parse(textwrap.dedent(source))
    fn, = tree.body
    branch, flatten, projection, result = fn.body
    assert isinstance(branch, ast.If) and not branch.orelse
    assert ast.unparse(branch.test) == 'self.linear_g is not None'
    assert len(branch.body) == 3
    assert ast.unparse(branch.body[-1]) == 'o = o * g'
    assert ast.unparse(flatten) == 'o = flatten_final_dims(o, 2)'
    assert ast.unparse(projection) == 'o = self.linear_o(o)'
    assert ast.unparse(result) == 'return o'
    branch.body[-1] = ast.parse('o = _rsi_gate_flat(o, g)').body[0]
    branch.orelse = [flatten]
    fn.body = [branch, projection, result]
    return ast.fix_missing_locations(tree)


def install(cls):
    from head_gate_backward import gate_flat, eligible
    original = cls._wrap_up
    source = textwrap.dedent(inspect.getsource(original))
    stats = {'enabled': True, 'backward_fused': True, 'eligible_calls': 0, 'fallback_calls': 0,
             'block': 1024, 'original_sha256': hashlib.sha256(source.encode()).hexdigest()}

    def fused(o, g):
        stats['eligible_calls' if eligible(o, g) else 'fallback_calls'] += 1
        return gate_flat(o, g, 1024)

    namespace = dict(original.__globals__)
    namespace['_rsi_gate_flat'] = fused
    exec(compile(rewrite(source), inspect.getsourcefile(original), 'exec'), namespace)
    patched = namespace['_wrap_up']
    cls._wrap_up = patched

    def restore():
        assert cls._wrap_up is patched
        cls._wrap_up = original

    return restore, stats
