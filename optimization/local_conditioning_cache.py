"""Experimental scoped no-grad encoder conditioning reuse; not integrated in training."""
import ast, contextlib, contextvars, hashlib, inspect, textwrap
import torch
_scope = contextvars.ContextVar('local_conditioning', default=None)

def tensor_key(x):
    try:
        return (x.device,x.dtype,tuple(x.shape),tuple(x.stride()),x.data_ptr(),x.storage_offset(),x._version)
    except (RuntimeError,NotImplementedError):
        return None

@contextlib.contextmanager
def cache_scope(device_types=('cuda',), allow_eval_child=False):
    state={'entries':{},'hits':0,'misses':0,'peak_output_bytes':0,'device_types':device_types,'allow_eval_child':allow_eval_child}
    token=_scope.set(state)
    try: yield state
    finally:
        state['entries'].clear()
        _scope.reset(token)

def reuse(module, stage, inputs, compute):
    state=_scope.get()
    if state is None or torch.is_grad_enabled(): return compute()
    keys=tuple(tensor_key(x) for x in inputs)
    params=tuple((id(p),tensor_key(p)) for p in module.parameters())
    if any(k is None for k in keys) or any(k is None for _,k in params):return compute()
    device=inputs[0].device
    key=(keys,params,tuple((name,id(m),m.training,getattr(m,'eps',None)) for name,m in module.named_modules()),module.n_queries,module.n_keys,torch.is_autocast_enabled(device.type),torch.get_autocast_dtype(device.type),torch.backends.cuda.matmul.allow_tf32,torch.cuda.current_stream(device).cuda_stream if device.type=='cuda' else None)
    entry=state['entries'].get((id(module),stage))
    if entry is not None and entry[0]==key and tensor_key(entry[2])==entry[3]:
        state['hits']+=1
        return entry[2]
    result=compute();state['misses']+=1
    state['entries'][(id(module),stage)]=(key,inputs,result,tensor_key(result))
    state['peak_output_bytes']=max(state['peak_output_bytes'],sum(e[2].numel()*e[2].element_size() for e in state['entries'].values()))
    return result

def install(cls):
    original=cls.forward
    source=textwrap.dedent(inspect.getsource(original));tree=ast.parse(source)
    fn=tree.body[0]
    class Single(ast.NodeTransformer):
        count=0
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='c_l' and isinstance(node.value,ast.BinOp):
                self.count+=1
                expr=node.value
                node.value=ast.Call(func=ast.Name(id='_reuse_conditioning',ctx=ast.Load()),args=[ast.Name(id='self',ctx=ast.Load()),ast.Constant('single'),ast.Tuple(elts=[ast.Name(id=n,ctx=ast.Load()) for n in ('c_l','s','atom_to_token_idx')],ctx=ast.Load()),ast.Lambda(args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=expr)],keywords=[])
            return self.generic_visit(node)
    rewrite=Single();tree=rewrite.visit(tree);assert rewrite.count==1
    start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple) and [x.id for x in n.targets[0].elts]==['c_l_q','c_l_k','_'])
    assert isinstance(fn.body[start+1],ast.If) and ast.unparse(fn.body[start+1].test)=='inplace_safe'
    pair_body=fn.body[start:start+2]
    helper=ast.parse('def _compute_pair(p_lm):\n return p_lm').body[0]
    helper.body=pair_body+[ast.Return(value=ast.Name(id='p_lm',ctx=ast.Load()))]
    call=ast.parse("p_lm = _reuse_conditioning(self, 'pair', (p_lm,c_l), lambda: _compute_pair(p_lm))").body[0]
    fn.body[start:start+2]=[helper,call]
    ast.fix_missing_locations(tree);namespace=dict(original.__globals__);namespace['_reuse_conditioning']=reuse
    exec(compile(tree,inspect.getsourcefile(original),'exec'),namespace)
    patched=namespace['forward'];signature=inspect.signature(original)
    def forward(self,*args,**kwargs):
        state=_scope.get()
        if state is None or torch.is_grad_enabled() or (not self.training and not state['allow_eval_child']):return original(self,*args,**kwargs)
        values=signature.bind(self,*args,**kwargs);values.apply_defaults();v=values.arguments
        if v['inplace_safe'] or any(v[k] is None for k in ('r_l','s','p_lm','c_l')) or v['c_l'].device.type not in state['device_types']:
            return original(self,*args,**kwargs)
        return patched(self,*args,**kwargs)
    cls.forward=forward
    def restore():cls.forward=original
    return restore,hashlib.sha256(source.encode()).hexdigest()

def install_sampler(model_cls, device_types=('cuda',)):
    """Limit reuse to a single training mini-rollout with native shared inputs."""
    original=model_cls.sample_diffusion
    stats={'invocations':0,'hits':0,'misses':0,'peak_output_bytes':0,'scope':'training no-grad encoder conditioning only'}
    def sample(self,**kwargs):
        if not self.training or torch.is_grad_enabled() or any(kwargs.get(k) is None for k in ('p_lm','c_l')):
            return original(self,**kwargs)
        with cache_scope(device_types, allow_eval_child=True) as state:
            try:return original(self,**kwargs)
            finally:
                stats['invocations']+=1
                for k in ('hits','misses'):stats[k]+=state[k]
                stats['peak_output_bytes']=max(stats['peak_output_bytes'],state['peak_output_bytes'])
    model_cls.sample_diffusion=sample
    def restore():model_cls.sample_diffusion=original
    return restore,stats
