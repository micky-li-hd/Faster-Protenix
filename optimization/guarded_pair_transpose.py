"""Scoped Pairformer copy replacement; native fallback outside tested regime."""
import ast
import hashlib
import inspect
import textwrap
import torch


def eligible(x):
    return (x.is_cuda and x.dtype in (torch.bfloat16, torch.float32)
            and x.ndim in (3,4) and x.is_contiguous()
            and x.shape[-3] >= 256 and x.shape[-2] >= 256
            and x.shape[-1] in (128,256))


def rewrite(source):
    tree=ast.parse(textwrap.dedent(source))
    expected=ast.dump(ast.parse('z.transpose(-2, -3).contiguous()',mode='eval').body)
    class Copies(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            if ast.dump(node)==expected:
                self.count+=1
                return ast.copy_location(ast.Call(func=ast.Name(id='_rsi_pair_copy',ctx=ast.Load()),
                    args=[ast.Name(id='z',ctx=ast.Load())],keywords=[]),node)
            return self.generic_visit(node)
    patch=Copies();tree=patch.visit(tree);assert patch.count==4,patch.count
    ast.fix_missing_locations(tree)
    return tree


def install(cls):
    original=cls.forward
    source=textwrap.dedent(inspect.getsource(original))
    stats={'enabled':True,'eligible_calls':0,'fallback_calls':0,'tile':4,
           'original_forward_sha256':hashlib.sha256(source.encode()).hexdigest()}
    def copy(x):
        if eligible(x):
            from pair_transpose import transpose
            stats['eligible_calls']+=1
            return transpose(x,4)
        stats['fallback_calls']+=1
        return x.transpose(-2,-3).contiguous()
    tree=rewrite(source)
    namespace=dict(original.__globals__);namespace['_rsi_pair_copy']=copy
    exec(compile(tree,inspect.getsourcefile(original),'exec'),namespace)
    patched=namespace['forward'];cls.forward=patched
    def restore():
        assert cls.forward is patched
        cls.forward=original
    return restore,stats
