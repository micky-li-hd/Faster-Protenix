"""Remove only the final empty_cache call from the verified native train_step."""
import ast
import hashlib
import inspect
import textwrap


def install(trainer_class):
    original = trainer_class.train_step
    assert original.__closure__ is None
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    assert len(tree.body) == 1
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef) and function.name == 'train_step'
    assert not function.decorator_list
    expected = ast.parse('torch.cuda.empty_cache()').body[0]
    matching = [node for node in ast.walk(function)
                if ast.dump(node, include_attributes=False) == ast.dump(expected, include_attributes=False)]
    assert len(matching) == 1 and matching[0] is function.body[-1], 'Native allocator call changed'
    stats = {'policy': 'retain_training_allocator_cache', 'skipped_empty_cache': 0,
             'original_train_step_sha256': hashlib.sha256(source.encode()).hexdigest(),
             'scope': 'Only terminal train_step empty_cache; model initialization and evaluation unchanged.'}
    def skipped():
        stats['skipped_empty_cache'] += 1
    replacement = ast.parse('_rsi_allocator_skipped()').body[0]
    function.body[-1] = ast.copy_location(replacement, function.body[-1])
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    assert '_rsi_allocator_skipped' not in namespace
    namespace['_rsi_allocator_skipped'] = skipped
    exec(compile(tree, inspect.getsourcefile(original) + ':allocator_policy', 'exec'), namespace)
    modified = namespace['train_step']
    modified.__module__ = original.__module__
    modified.__qualname__ = original.__qualname__
    modified.__doc__ = original.__doc__
    trainer_class.train_step = modified
    return stats
