"""Optional readable lowering of fused attention, using real PyTorch operators."""
import math
import torch


def repeat_kv(value, repeats):
    return value.repeat_interleave(repeats, dim=-3)


def mask_attention(scores, mask=None, causal=False):
    if causal:
        keep = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~keep, float('-inf'))
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf')) if mask.dtype == torch.bool else scores + mask
    return scores


def safe_softmax(value, dim=-1):
    # Native SDPA returns zeros for entirely masked rows, unlike ordinary softmax.
    return torch.nan_to_num(torch.softmax(value, dim=dim), nan=0.)

def restore_layout(value, strides):
    # Fused SDPA may return a transposed-stride allocation expected by a later view.
    # Preserve that storage contract instead of changing downstream mathematical ops.
    return torch.empty_strided(value.shape, strides, dtype=value.dtype, device=value.device).copy_(value)


def lower_attention(module):
    for node in list(module.graph.nodes):
        if str(node.target) != 'aten.scaled_dot_product_attention.default':
            continue
        q,k,v=node.args[:3]
        mask=node.args[3] if len(node.args)>3 else node.kwargs.get('attn_mask')
        dropout=node.args[4] if len(node.args)>4 else node.kwargs.get('dropout_p',0)
        causal=node.args[5] if len(node.args)>5 else node.kwargs.get('is_causal',False)
        if dropout:
            node.meta['silicondevine_attention_budget']='Stochastic attention dropout is kept fused.'
            continue
        meta=q.meta.get('val')
        if meta is None or not isinstance(meta.shape[-1],int):
            continue
        key_meta=k.meta.get('val')
        try:
            score_count=math.prod(int(v) for v in meta.shape[:-1])*int(key_meta.shape[-2]) if key_meta is not None else 262145
        except (TypeError,ValueError):
            score_count=262145
        if score_count>262144:
            node.meta['silicondevine_attention_budget']='Fused attention retained: scores exceed 262144 elements.'
            continue
        scale=node.kwargs.get('scale')
        scale=scale if scale is not None else 1/math.sqrt(meta.shape[-1])
        with module.graph.inserting_before(node):
            expanded=[]
            if node.kwargs.get('enable_gqa',False):
                repeats=int(meta.shape[-3])//int(key_meta.shape[-3])
                k=module.graph.call_function(repeat_kv,(k,repeats))
                v=module.graph.call_function(repeat_kv,(v,repeats))
                expanded.extend([k,v])
            kt=module.graph.call_function(torch.ops.aten.transpose.int,(k,-1,-2))
            scores=module.graph.call_function(torch.ops.aten.matmul.default,(q,kt))
            scaled=module.graph.call_function(torch.ops.aten.mul.Tensor,(scores,scale))
            masked=module.graph.call_function(mask_attention,(scaled,mask,bool(causal))) if mask is not None or causal else scaled
            probability=module.graph.call_function(safe_softmax,(masked,-1)) if mask is not None or causal else module.graph.call_function(torch.ops.aten.softmax.int,(scaled,-1))
            output=module.graph.call_function(torch.ops.aten.matmul.default,(probability,v))
            result_meta=node.meta.get('val')
            restored=module.graph.call_function(restore_layout,(output,tuple(int(s) for s in result_meta.stride()))) if result_meta is not None else output
            for inner in (*expanded,kt,scores,scaled,masked,probability,output,restored):
                inner.meta.update({key:value for key,value in node.meta.items() if key!='val'})
                inner.meta['silicondevine_lowered_from']='aten.scaled_dot_product_attention.default'
            for inner,semantic in [(scores,'QK score'),(scaled,'scaled score'),(masked,'masked score'),(probability,'attention probability'),(output,'weighted V')]:
                inner.meta['silicondevine_semantic']=semantic
        node.replace_all_uses_with(restored)
        module.graph.erase_node(node)
    module.graph.lint()
    module.recompile()
    return module


def inline_disabled_autocast(module):
    """Expose upstream RoPE subgraphs without changing enabled autocast semantics."""
    import operator
    for node in list(module.graph.nodes):
        if str(node.target) != 'wrap_with_autocast' or node.args[2] is not False:
            continue
        ref=node.args[4]
        sub=module.get_submodule(ref.target)
        if any(user.target is not operator.getitem for user in node.users):
            continue
        env={};arguments=iter(node.args[5:]);outputs=None
        with module.graph.inserting_before(node):
            for inner in sub.graph.nodes:
                if inner.op=='placeholder':env[inner]=next(arguments)
                elif inner.op=='output':outputs=torch.fx.map_arg(inner.args[0],lambda n:env[n])
                elif inner.op=='get_attr':
                    env[inner]=module.graph.get_attr(str(ref.target)+'.'+str(inner.target))
                else:
                    new=module.graph.node_copy(inner,lambda n:env[n]);env[inner]=new
                    new.meta={**node.meta,**inner.meta}
        for user in list(node.users):
            user.replace_all_uses_with(outputs[user.args[1]])
            module.graph.erase_node(user)
        module.graph.erase_node(node)
    module.graph.eliminate_dead_code();module.graph.lint();module.recompile()
    return module
