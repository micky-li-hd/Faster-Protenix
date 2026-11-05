import torch
EXTENSION=None

class Affine16(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,b,eps):
        ctx.eps=eps
        with torch.autocast('cuda',enabled=False):
            y,mean,rs=torch.ops.aten.native_layer_norm.default(x,[16],w,b,eps)
        ctx.save_for_backward(x,w,b,mean,rs)
        return y
    @staticmethod
    def backward(ctx,dy):
        x,w,b,mean,rs=ctx.saved_tensors
        mask=(ctx.needs_input_grad[0],w is not None and ctx.needs_input_grad[1],b is not None and ctx.needs_input_grad[2])
        if torch.is_grad_enabled():
            # Recompute native statistics in the graph for correct higher derivatives.
            with torch.enable_grad(),torch.autocast('cuda',enabled=False):
                y=torch.nn.functional.layer_norm(x,[16],w,b,ctx.eps)
                targets=[t for t,want in zip((x,w,b),mask) if want]
                derivatives=iter(torch.autograd.grad(y,targets,dy,create_graph=True))
                return (*[next(derivatives) if want else None for want in mask],None)
        dy=dy.contiguous();dx=dw=db=None
        if mask[0]:dx=torch.ops.aten.native_layer_norm_backward.default(dy,x,[16],mean,rs,w,b,(True,False,False))[0]
        if mask[1] or mask[2]:dw,db=EXTENSION.affine16_backward(x,dy,mean.contiguous(),rs.contiguous(),mask[1],mask[2],1)
        return dx,dw,db,None

def eligible(module,x):
    return (x.is_cuda and x.dtype==torch.float32 and x.is_contiguous() and x.ndim>=2
            and x.shape[-1]==16 and tuple(module.c_in)==(16,) and 131072<=x.numel()//16<=344065
            and module.eps in (1e-5,1e-3)
            and all(t is None or (t.device==x.device and t.dtype==torch.float32 and t.is_contiguous() and t.shape==(16,)) for t in (module.weight,module.bias)))

def install(cls,extension,existing=()):
    import types,weakref
    global EXTENSION
    EXTENSION=extension
    original_init=cls.__init__;stats={'eligible':0,'fallback':0,'installed':0};records=[];missing=object();active=[True]
    def attach(module):
        if tuple(module.c_in)!=(16,):return
        previous=module.__dict__.get('forward',missing)
        assert previous is missing, 'Refuse to replace an existing instance forward'
        fallback=cls.forward
        def forward(self,x):
            if not active[0]:return fallback(self,x)
            if not eligible(self,x):stats['fallback']+=1;return fallback(self,x)
            stats['eligible']+=1
            return Affine16.apply(x,self.weight,self.bias,self.eps)
        module.forward=types.MethodType(forward,module)
        records.append((weakref.ref(module),previous,forward));stats['installed']+=1
    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs);attach(self)
    cls.__init__=init
    for module in existing:attach(module)
    def restore():
        assert cls.__init__ is init
        active[0]=False
        cls.__init__=original_init
        for ref,previous,forward in records:
            module=ref()
            if module is None:continue
            assert getattr(module.forward,'__func__',None) is forward
            if previous is missing:del module.forward
            else:module.forward=previous
    return restore,stats
