"""Coordinate provenance for routing primitives observed in a real ATen trace."""
import torch
from torch import fx

def normalize_execution(c,node,result,op,attrs):
    target=str(node.target)
    aliases={'aten.div_.Tensor':'divide','aten.t.default':'transpose','aten.min.default':'amin','aten.max.default':'amax'}
    if target in aliases:
        if target=='aten.t.default':attrs['dims']=[0,1]
        if target in ('aten.min.default','aten.max.default'):attrs.update(dims=list(range(c.env[node.args[0]].ndim)),keepdim=False)
        return aliases[target]
    names={'aten.topk.default':'topk','aten.index.Tensor':'index','aten.index_add_.default':'index_add',
           'aten.scatter_.value':'scatter','aten.bitwise_not.default':'not','aten.masked_fill.Scalar':'masked_fill',
           'aten.one_hot.default':'one_hot','aten.where.default':'nonzero','aten.nonzero.default':'nonzero',
           'aten.unbind.int':'unbind','aten.zeros.default':'constant','aten.zeros_like.default':'constant','aten.full.default':'constant'}
    if target=='aten.concat.default':
        attrs['dim']=int(node.args[1]) if len(node.args)>1 else 0
        return 'concat'
    if target not in names:return op
    args=fx.map_arg(node.args,lambda n:c.env[n]);kind=names[target]
    outputs=list(result) if isinstance(result,(tuple,list)) else [result]
    if sum(v.numel() for v in outputs)>4096:
        attrs['boundary_reason']='Routing coordinate map exceeds 4096 output elements';return 'opaque'
    refs=lambda arg:c.deps(arg)[0]
    def dep(arg,i,weight=None):
        return {'tensor':refs(arg),'index':int(i),**({'weight':weight} if weight is not None else {})}
    maps=[]
    if kind=='constant':maps=[[[] for _ in v.flatten()] for v in outputs]
    elif kind=='index':
        grid=torch.arange(args[0].numel()).reshape(args[0].shape)
        selected=torch.ops.aten.index.Tensor(grid,args[1]).flatten()
        maps=[[[dep(node.args[0],i,1)] for i in selected]]
    elif kind=='unbind':
        grids=torch.arange(args[0].numel()).reshape(args[0].shape).unbind(int(args[1]) if len(args)>1 else 0)
        maps=[[[dep(node.args[0],i,1)] for i in grid.flatten()] for grid in grids]
    elif kind=='topk':
        x=args[0];dim=int(args[2]) if len(args)>2 else -1;dim%=x.ndim
        grid=torch.arange(x.numel()).reshape(x.shape);selected=grid.gather(dim,result[1]).flatten()
        maps=[[[dep(node.args[0],i,1)] for i in selected]]
        # Index selection depends on every candidate in its comparison row.
        idx=[]
        for co in torch.cartesian_prod(*[torch.arange(s) for s in result[1].shape]).reshape(-1,result[1].ndim):
            sl=list(co.tolist());sl[dim]=slice(None);idx.append([dep(node.args[0],i) for i in grid[tuple(sl)].flatten()])
        maps.append(idx)
    elif kind=='index_add':
        x,dim,idx,src=args[:4];alpha=float(node.kwargs.get('alpha',1));grid=torch.arange(x.numel()).reshape(x.shape)
        destinations=grid.index_select(dim,idx).flatten();rows=[[dep(node.args[0],i,1)] for i in range(x.numel())]
        for i,d in enumerate(destinations):rows[int(d)].append(dep(node.args[3],i,alpha))
        maps=[rows]
    elif kind=='not':maps=[[[dep(node.args[0],i)] for i in range(result.numel())]]
    elif kind=='one_hot':maps=[[[dep(node.args[0],i//result.shape[-1])] for i in range(result.numel())]]
    elif kind=='nonzero':
        coords=args[0].nonzero();grid=torch.arange(args[0].numel()).reshape(args[0].shape)
        selected=[int(grid[tuple(co.tolist())]) for co in coords]
        maps=[[[dep(node.args[0],selected[i if isinstance(result,tuple) else i//coords.shape[1]])] for i in range(v.numel())] for v in outputs]
    elif kind=='masked_fill':
        mask=args[1].expand_as(args[0]).flatten();maps=[[[dep(node.args[0],i,1)] if not mask[i] else [] for i in range(result.numel())]]
    elif kind=='scatter':
        x,dim,idx,value=args[:4];grid=torch.arange(x.numel()).reshape(x.shape);dest=set(grid.gather(dim,idx).flatten().tolist())
        maps=[[[dep(node.args[0],i,1)] if i not in dest else [] for i in range(result.numel())]]
    attrs['coordinateDependencies']=maps
    attrs['routingKind']=kind
    attrs['capturedTarget']=target
    return 'routing_'+kind

def lower_execution(graph):
    """Lower the two upstream Lightning contractions without altering their order."""
    for node in list(graph.graph.nodes):
        if str(node.target)!='aten.einsum.default':continue
        equation=node.args[0].replace(' ','');a,b=node.args[1]
        if equation not in ('...nd,...ne->...de','...ne,...ed->...nd'):continue
        with graph.graph.inserting_before(node):
            if equation=='...nd,...ne->...de':
                a=graph.graph.call_function(torch.ops.aten.transpose.int,(a,-1,-2));a.meta=dict(node.meta)
            out=graph.graph.call_function(torch.ops.aten.matmul.default,(a,b));out.meta=dict(node.meta)
            out.meta['silicondevine_lowered_from']='aten.einsum: '+equation
        node.replace_all_uses_with(out);graph.graph.erase_node(node)
    graph.graph.lint();graph.recompile();return graph

def fold_norms(result,model):
    """One real RMSNorm remains one shared atom, retaining the captured children."""
    tensors={t['id']:t for t in result['tensors']}
    for path,part in model.named_modules():
        if not path or not type(part).__name__.endswith('RMSNorm'):continue
        children=[n for n in result['nodes'] if n['group']==path]
        if len(children)<2:continue
        produced={o for n in children for o in n['outputs']}
        external=list(dict.fromkeys(i for n in children for i in n['inputs'] if i not in produced))
        weight=next((i for i in external if tensors[i]['role']=='parameter'),None)
        inputs=[i for i in external if tensors[i]['role'] not in ('parameter','constant')]
        if len(inputs)!=1 or not weight:continue
        last=children[-1];ids={n['id'] for n in children}
        outside={i for n in result['nodes'] if n['id'] not in ids for i in n['inputs']}|set(result['outputs'])
        if produced&outside!=set(last['outputs']):continue
        node=dict(last,op='rmsnorm',name=path,inputs=inputs,parameters={'weight':weight},
                  attrs={'normalized_shape':[tensors[weight]['shape'][-1]],'eps':getattr(part,'variance_epsilon',1e-6),
                         'children':children,'codeTrace':last['attrs'].get('codeTrace',''),'sourceModule':type(part).__name__})
        result['nodes']=[node if n['id']==last['id'] else n for n in result['nodes'] if n['id'] not in ids or n['id']==last['id']]
