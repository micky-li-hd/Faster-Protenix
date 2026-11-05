"""Scoped FP32 CUDA EMA update; preserve native fallback and storage swaps."""
import hashlib
import inspect


def eligible(wrapper, torch):
    if wrapper.decay != 0.999 or wrapper.backup:
        return False
    device = None
    count = 0
    for name, param in wrapper.model.named_parameters():
        if wrapper.mutable_param_keywords and not any(
            k in name for k in wrapper.mutable_param_keywords
        ):
            continue
        shadow = wrapper.shadow.get(name)
        if shadow is None:
            return False
        if device is None:
            device = param.device
        if (param.device.type != 'cuda' or param.device != device
                or shadow.device != device or param.dtype != torch.float32
                or shadow.dtype != torch.float32 or param.shape != shadow.shape
                or param.layout != torch.strided or shadow.layout != torch.strided
                or not param.is_contiguous() or not shadow.is_contiguous()
                or shadow.requires_grad):
            return False
        count += 1
    return count > 0


def install(module, torch):
    from foreach_ema import EMAWrapper as Candidate
    from native_ema import EMAWrapper as Reference
    cls = module.EMAWrapper
    original = cls.update
    assert inspect.getsource(original) == inspect.getsource(Reference.update), 'Native EMA update changed'
    stats = dict(enabled=True, calls=0, eligible_calls=0, bypassed_calls=0,
                 restored=False, source_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest())

    def update(self):
        stats['calls'] += 1
        if not eligible(self, torch):
            stats['bypassed_calls'] += 1
            return original(self)
        stats['eligible_calls'] += 1
        return Candidate.update(self)

    cls.update = update

    def restore():
        assert cls.update is update
        cls.update = original
        stats['restored'] = True

    return restore, stats
