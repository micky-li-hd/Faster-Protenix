"""Recompute a native SiLU activation in backward; no altered GEMMs or precision."""
import torch
import torch.nn.functional as F

class RecomputeSiLUMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx,u,v):
        assert u.shape==v.shape and u.dtype==v.dtype and u.device==v.device
        ctx.save_for_backward(u,v)
        return F.silu(u)*v
    @staticmethod
    def backward(ctx,dy):
        u,v=ctx.saved_tensors
        if torch.is_grad_enabled():
            with torch.enable_grad():
                y=F.silu(u)*v
                targets=[x for x,need in zip((u,v),ctx.needs_input_grad) if need]
                grads=iter(torch.autograd.grad(y,targets,dy,create_graph=True))
                return tuple(next(grads) if need else None for need in ctx.needs_input_grad)
        du=torch.ops.aten.silu_backward.default(dy*v,u) if ctx.needs_input_grad[0] else None
        dv=dy*F.silu(u) if ctx.needs_input_grad[1] else None
        return du,dv

def conditioned_forward(module,a,s,variant):
    if variant=='native':return module(a,s)
    assert variant=='recompute_silu'
    a=module.adaln(a,s)
    u=module.linear_nobias_a1(a);v=module.linear_nobias_a2(a)
    b=RecomputeSiLUMul.apply(u,v)
    return torch.sigmoid(module.linear_s(s))*module.linear_nobias_b(b)

def saved_storage(module,a,s,variant):
    excluded={x.untyped_storage().data_ptr() for x in (a,s,*module.parameters())}
    unique={};calls=[]
    def pack(x):
        storage=x.untyped_storage();key=storage.data_ptr()
        if key not in excluded:unique[key]=storage.nbytes()
        calls.append({'shape':list(x.shape),'dtype':str(x.dtype),'storage_bytes':storage.nbytes(),'excluded':key in excluded})
        return x
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):
        y=conditioned_forward(module,a,s,variant)
    return {'unique_saved_nonparameter_input_bytes':sum(unique.values()),'saved_tensor_calls':calls}
