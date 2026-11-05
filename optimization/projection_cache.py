"""Scoped caching of native pair bias projection; no training integration yet."""
import ast,contextvars,contextlib,hashlib,inspect,textwrap,functools
import torch
_scope=contextvars.ContextVar('rsi_native_pair_projection_scope',default=None)

def tensor_key(x):
    try:return (x.device,x.dtype,tuple(x.shape),tuple(x.stride()),x.data_ptr(),x.storage_offset(),x._version)
    except (RuntimeError,NotImplementedError):return None

@contextlib.contextmanager
def cache_scope(device_types=('cuda',)):
    state={'entries':{},'device_types':tuple(device_types),'hits':0,'misses':0,'bypasses':0,'peak_output_bytes':0}
    token=_scope.set(state)
    try:yield state
    finally:state['entries'].clear();_scope.reset(token)

def project(module,z):
    def original():return module.linear_nobias_z(module.layernorm_z(z))
    state=_scope.get()
    if state is None or torch.is_grad_enabled() or z.device.type not in state['device_types']:
        if state is not None:state['bypasses']+=1
        return original()
    ln=module.layernorm_z;linear=module.linear_nobias_z
    params=tuple((id(p),tensor_key(p)) for m in [ln,linear] for p in m.parameters())
    key=tensor_key(z)
    if key is None or any(k is None for _,k in params):state['bypasses']+=1;return original()
    key=(key,params,id(ln),id(linear),ln.training,linear.training,ln.eps,repr(getattr(ln,'normalized_shape',getattr(ln,'c_in',None))),torch.is_autocast_enabled(z.device.type),torch.get_autocast_dtype(z.device.type),torch.backends.cuda.matmul.allow_tf32,torch.cuda.current_stream(z.device).cuda_stream if z.device.type=='cuda' else None)
    entry=state['entries'].get(id(module))
    if entry is not None and entry[0]==key and tensor_key(entry[2])==entry[3]:state['hits']+=1;return entry[2]
    y=original();state['misses']+=1;yk=tensor_key(y)
    if yk is not None:
        # Only retain projected n_heads output, never the wider layernorm intermediate.
        state['entries'][id(module)]=(key,z,y,yk)
        state['peak_output_bytes']=max(state['peak_output_bytes'],sum(e[2].numel()*e[2].element_size() for e in state['entries'].values()))
    return y

def patched_method(fn):
    """Change exactly one native LN->linear expression; retain all attention code."""
    source=textwrap.dedent(inspect.getsource(fn));tree=ast.parse(source)
    target=ast.dump(ast.parse('self.linear_nobias_z(self.layernorm_z(z))',mode='eval').body,include_attributes=False)
    class Rewrite(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            if ast.dump(node,include_attributes=False)==target:
                self.count+=1
                return ast.copy_location(ast.parse('_rsi_cached_projection(self,z)',mode='eval').body,node)
            return self.generic_visit(node)
    rewrite=Rewrite();tree=rewrite.visit(tree);assert rewrite.count==1
    ast.fix_missing_locations(tree);namespace=dict(fn.__globals__);namespace['_rsi_cached_projection']=project
    exec(compile(tree,inspect.getsourcefile(fn),'exec'),namespace)
    return namespace[fn.__name__],hashlib.sha256(source.encode()).hexdigest()

def install_attention(cls):
    names=['standard_multihead_attention','local_multihead_attention']
    originals={name:getattr(cls,name) for name in names};replacements={};hashes={}
    for name,fn in originals.items():replacements[name],hashes[name]=patched_method(fn)
    for name,fn in replacements.items():setattr(cls,name,fn)
    def restore():
        for name,fn in originals.items():setattr(cls,name,fn)
    return restore,hashes

def sampling_wrapper(fn,stats):
    """A scope exists only for one no-grad invocation with native pair_z cache."""
    @functools.wraps(fn)
    def wrapped(self,**kwargs):
        if torch.is_grad_enabled() or kwargs.get('pair_z') is None:return fn(self,**kwargs)
        with cache_scope() as state:
            try:return fn(self,**kwargs)
            finally:
                stats['invocations']=stats.get('invocations',0)+1
                for k in ['hits','misses','bypasses']:stats[k]=stats.get(k,0)+state[k]
                stats['peak_output_bytes']=max(stats.get('peak_output_bytes',0),state['peak_output_bytes'])
    return wrapped
