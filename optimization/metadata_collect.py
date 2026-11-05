"""CPU preparation of discrete residue metadata; preserve native coordinate views."""
import ast,hashlib,inspect,textwrap,functools
import torch

def verify_source(original):
    source=textwrap.dedent(inspect.getsource(original))
    tree=ast.parse(source)
    # Coordinate values must not influence selection: the native function only
    # appends a slice to coord_list. Refuse source changes that violate this.
    uses=[n for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id=='coord']
    slices=[n for n in ast.walk(tree) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='coord']
    assert len(uses)==len(slices)==1
    assert ast.unparse(slices[0])=='coord[start:end, :]'
    parents={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    assert ast.unparse(parents[slices[0]])=='coord_list.append(coord[start:end, :])'
    return hashlib.sha256(source.encode()).hexdigest()

def collect(original,coord,coord_mask,ref_space_uid,atom_perm_list,run_checker=False):
    # Fresh metadata on every call: no pointer/version cache or stale decisions.
    mask=coord_mask.cpu();uid=ref_space_uid.cpu()
    dummy=torch.empty((coord.shape[0],coord.shape[1]),dtype=coord.dtype,device='cpu')
    positions,_,masks,perms,n_res=original(dummy,mask,uid,atom_perm_list,run_checker)
    return (positions,[coord[start:end,:] for start,end in positions],
            [m.to(coord_mask.device) for m in masks],
            [p.to(coord_mask.device) for p in perms],n_res)

def install(module):
    original=module.collect_residues_with_symmetric_atoms
    stats=dict(calls=0,eligible=0,restored=False,source_sha256=verify_source(original))
    @functools.wraps(original)
    def wrapped(coord,coord_mask,ref_space_uid,atom_perm_list,run_checker=False):
        stats['calls']+=1
        if (coord.device.type!='cuda' or coord.ndim!=2 or coord.shape[-1]!=3
            or coord_mask.ndim!=1 or ref_space_uid.ndim!=1
            or coord_mask.device!=coord.device or ref_space_uid.device!=coord.device
            or len(coord_mask)!=len(coord) or len(ref_space_uid)!=len(coord)
            or not isinstance(atom_perm_list,list)):
            return original(coord,coord_mask,ref_space_uid,atom_perm_list,run_checker)
        stats['eligible']+=1
        return collect(original,coord,coord_mask,ref_space_uid,atom_perm_list,run_checker)
    module.collect_residues_with_symmetric_atoms=wrapped
    def restore():
        assert module.collect_residues_with_symmetric_atoms is wrapped
        module.collect_residues_with_symmetric_atoms=original;stats['restored']=True
    return restore,stats
