"""Experimental FP32 DiffusionTransformer SAC; never installed implicitly."""
import functools


def install(policy_name):
    import torch
    import torch.utils.checkpoint as cp
    import protenix.model.modules.transformer as module
    assert torch.__version__.startswith('2.7.1')
    assert policy_name in ('mm', 'matrix')
    ops = {torch.ops.aten.mm.default, torch.ops.aten.addmm.default}
    if policy_name == 'matrix':
        ops.add(torch.ops.aten.bmm.default)
    original = module.checkpoint_blocks
    stats = dict(policy=policy_name, matched_stack_calls=0, contexts=0,
                 saved_op_calls=0, cached_recompute_hits=0, saved_bytes_estimate=0,
                 max_context_saved_bytes_estimate=0, mutation_check=True,
                 saved_by_op={}, hit_by_op={})

    def context_fn():
        stats['contexts'] += 1
        context_bytes = 0

        def policy(ctx, op, *args, **kwargs):
            nonlocal context_bytes
            if op not in ops:
                return cp.CheckpointPolicy.PREFER_RECOMPUTE
            a, b = args[1:3] if op == torch.ops.aten.addmm.default else args[:2]
            if a.dtype != torch.float32 or b.dtype != torch.float32:
                return cp.CheckpointPolicy.PREFER_RECOMPUTE
            name = str(op)
            if ctx.is_recompute:
                stats['cached_recompute_hits'] += 1
                stats['hit_by_op'][name] = stats['hit_by_op'].get(name, 0) + 1
            else:
                elements = a.shape[-2] * b.shape[-1]
                if op == torch.ops.aten.bmm.default:
                    elements *= a.shape[0]
                size = elements * a.element_size()
                context_bytes += size
                stats['saved_bytes_estimate'] += size
                stats['max_context_saved_bytes_estimate'] = max(
                    stats['max_context_saved_bytes_estimate'], context_bytes)
                stats['saved_op_calls'] += 1
                stats['saved_by_op'][name] = stats['saved_by_op'].get(name, 0) + 1
            return cp.CheckpointPolicy.MUST_SAVE

        # Keep PyTorch's mutation/version checks. Never disable them to make a probe pass.
        return cp.create_selective_checkpoint_contexts(policy, allow_cache_entry_mutation=False)

    @functools.wraps(original)
    def checkpoint_blocks(blocks, args, blocks_per_ckpt):
        target = (torch.is_grad_enabled() and blocks_per_ckpt is not None
                  and len(blocks) == 24
                  and all(isinstance(getattr(b, 'func', None), module.DiffusionTransformerBlock)
                          for b in blocks))
        if not target:
            return original(blocks, args, blocks_per_ckpt)
        assert blocks_per_ckpt == 1
        assert not torch.is_autocast_enabled('cuda'), 'Native diffusion branch must remain FP32'
        stats['matched_stack_calls'] += 1
        values = args if type(args) is tuple else (args,)
        for block in blocks:
            values = cp.checkpoint(block, *values, use_reentrant=False, context_fn=context_fn)
            if type(values) is not tuple:
                values = (values,)
        return values

    module.checkpoint_blocks = checkpoint_blocks

    def restore():
        assert module.checkpoint_blocks is checkpoint_blocks
        module.checkpoint_blocks = original

    return stats, restore
