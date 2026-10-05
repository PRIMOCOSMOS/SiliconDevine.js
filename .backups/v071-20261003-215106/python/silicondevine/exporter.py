"""Capture real tensors and executed graphs. No approximation of unknown operators.

FX preserves module boundaries. The export backend lowers to ATen and requires
a traceable graph. Only the supplied execution path and input shapes are shown.
"""
import copy
import json
import math
import inspect
from pathlib import Path
import torch
from torch import fx, nn


def _json(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple, torch.Size)):
        return [_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    return str(value)


def _op(target, module=None):
    if module is not None:
        table = {nn.Linear: 'linear', nn.Conv1d: 'conv1d', nn.Conv2d: 'conv2d', nn.Conv3d: 'conv3d',
                 nn.ReLU: 'relu', nn.GELU: 'gelu', nn.Sigmoid: 'sigmoid', nn.Tanh: 'tanh',
                 nn.Softmax: 'softmax', nn.LayerNorm: 'layernorm', nn.BatchNorm2d: 'batchnorm',
                 nn.Flatten: 'flatten', nn.Identity: 'identity', nn.Dropout: 'dropout',
                 nn.MaxPool2d: 'maxpool2d', nn.AvgPool2d: 'avgpool2d',
                 nn.BatchNorm1d:'batchnorm', nn.BatchNorm3d:'batchnorm', nn.GroupNorm:'groupnorm',
                 nn.InstanceNorm1d:'instancenorm', nn.InstanceNorm2d:'instancenorm', nn.InstanceNorm3d:'instancenorm',
                 nn.MaxPool1d:'maxpool1d', nn.MaxPool3d:'maxpool3d', nn.AvgPool1d:'avgpool1d', nn.AvgPool3d:'avgpool3d',
                 nn.AdaptiveAvgPool1d:'adaptive_avg_pool1d', nn.AdaptiveAvgPool2d:'adaptive_avg_pool2d', nn.AdaptiveAvgPool3d:'adaptive_avg_pool3d',
                 nn.AdaptiveMaxPool1d:'adaptive_max_pool1d', nn.AdaptiveMaxPool2d:'adaptive_max_pool2d', nn.AdaptiveMaxPool3d:'adaptive_max_pool3d',
                 nn.SiLU:'silu', nn.LeakyReLU:'leaky_relu', nn.ELU:'elu', nn.Softplus:'softplus',
                 nn.ConvTranspose1d:'conv_transpose1d', nn.ConvTranspose2d:'conv_transpose2d', nn.ConvTranspose3d:'conv_transpose3d',
                 nn.Embedding:'embedding', nn.RMSNorm:'rmsnorm'}
        return table.get(type(module), 'opaque')
    name = str(target).lower()
    # Specific names first; never classify arbitrary names by substring matches.
    parts = name.replace('aten.', '').split('.')
    base = parts[0]
    if hasattr(target, '__name__') and not name.startswith('aten.'):
        base = target.__name__.lower()
    if name.startswith('<built-in function '):
        base = name.removeprefix('<built-in function ').removesuffix('>')
    aliases = {'add': 'add', 'sub':'subtract', 'mul': 'multiply', 'matmul': 'matmul', 'bmm': 'matmul', 'mm': 'matmul',
               'linear': 'linear', 'conv1d': 'conv1d', 'conv2d': 'conv2d', 'conv3d': 'conv3d',
               'relu': 'relu', 'gelu': 'gelu', 'sigmoid': 'sigmoid', 'tanh': 'tanh',
               'softmax': 'softmax', '_softmax': 'softmax', 'layer_norm': 'layernorm',
               'view': 'reshape', '_unsafe_view':'reshape', 'reshape': 'reshape', 'flatten': 'flatten', 'unflatten':'reshape',
               'unsqueeze':'reshape', 'squeeze':'reshape', 'contiguous':'identity', 'restore_layout':'identity',
               'select':'select', 'slice':'slice',
               'permute': 'permute', 'transpose': 'transpose', 'cat': 'concat',
               'clone': 'identity', 'detach': 'identity', 'dropout': 'dropout',
               'rms_norm':'rmsnorm','chunk':'chunk','split':'split','split_with_sizes':'split','getitem':'getitem',
               'stack':'stack','arange':'arange','repeat_kv':'repeat_kv','mask_attention':'attention_mask','safe_softmax':'softmax',
               'scaled_dot_product_attention':'attention','repeat':'repeat','div':'divide','truediv':'divide','pow':'power','mean':'mean','rsqrt':'rsqrt','neg':'negative','sin':'sin','cos':'cos','to':'cast','_to_copy':'cast','alias':'identity','expand':'expand'}
    aliases.update({key:key for key in ['silu','leaky_relu','elu','softplus','batch_norm','group_norm','instance_norm',
        'max_pool1d','max_pool2d','max_pool3d','avg_pool1d','avg_pool2d','avg_pool3d',
        'adaptive_avg_pool1d','adaptive_avg_pool2d','adaptive_avg_pool3d',
        'adaptive_max_pool1d','adaptive_max_pool2d','adaptive_max_pool3d',
        'conv_transpose1d','conv_transpose2d','conv_transpose3d','embedding','pad','constant_pad_nd','reflection_pad2d','replication_pad2d']})
    aliases.update({'batch_norm':'batchnorm','group_norm':'groupnorm','instance_norm':'instancenorm',
                    'max_pool1d':'maxpool1d','max_pool2d':'maxpool2d','max_pool3d':'maxpool3d',
                    'avg_pool1d':'avgpool1d','avg_pool2d':'avgpool2d','avg_pool3d':'avgpool3d'})
    aliases.update({'mix_kernels':'kernel_mix','batch_conv2d':'dynamic_conv2d'})
    aliases.update({'exp':'exp','randn_like':'standard_normal','randn':'standard_normal'})
    return aliases.get(base, 'opaque')


class Capture(fx.Interpreter):
    def __init__(self, module, include_values, value_limit, total_limit, tensor_sink=None):
        super().__init__(module)
        self.include_values, self.limit, self.remaining = include_values, value_limit, total_limit
        self.tensors, self.nodes, self.refs, self.inputs, self.outputs = [], [], {}, [], []
        self.parameters = {}
        self.tensor_sink = tensor_sink

    def tensor(self, value, name, role='activation', source=None):
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided or value.is_quantized or value.is_complex():
                raise ValueError(f'{name}: this exporter currently requires dense real tensors; layout={value.layout}, dtype={value.dtype}')
            tensor = {'id': name, 'shape': list(value.shape), 'dtype': str(value.dtype).replace('torch.', ''),
                      'role': role, 'stride': list(value.stride())}
            if source:
                tensor['source'] = source
            if self.include_values and self.remaining > 0:
                # Slice before device transfer: exports do not copy an entire GPU tensor to CPU.
                count = min(value.numel(), self.limit, self.remaining)
                flat = value.detach().reshape(-1)[:count].cpu().to(torch.float64)
                values = flat.tolist()
                tensor['data'] = {'offset': 0, 'values': [_json(v) for v in values]}
                special = {str(i): '-inf' if v == -math.inf else '+inf' if v == math.inf else 'nan' for i,v in enumerate(values) if not math.isfinite(v)}
                if special:
                    tensor['specialValues']=special
                finite = [v for v in values if math.isfinite(v)]
                if finite:
                    tensor['stats'] = {'min': min(finite), 'max': max(finite), 'absmax': max(abs(v) for v in finite)}
                self.remaining -= count
            self.tensors.append(tensor)
            if self.tensor_sink:
                self.tensor_sink(name, value)
            return [name]
        if isinstance(value, (list, tuple)):
            return [r for i, v in enumerate(value) for r in self.tensor(v, f'{name}:{i}', role, source)]
        if isinstance(value, dict):
            return [r for k, v in value.items() for r in self.tensor(v, f'{name}:{k}', role, source)]
        return []

    def deps(self, value):
        if isinstance(value, fx.Node):
            return self.refs.get(value.name, [])
        if isinstance(value, (tuple, list)):
            return [r for v in value for r in self.deps(v)]
        if isinstance(value, dict):
            return [r for v in value.values() for r in self.deps(v)]
        return []

    def run_node(self, node):
        result = super().run_node(node)
        if node.op == 'output':
            self.outputs = list(dict.fromkeys(self.deps(node.args)))
            return result
        role = 'input' if node.op == 'placeholder' else 'activation'
        if node.op == 'get_attr':
            role = 'parameter' if isinstance(result, nn.Parameter) else 'buffer'
        refs = self.tensor(result, node.name, role, str(node.target))
        self.refs[node.name] = refs
        if node.op == 'placeholder':
            self.inputs.extend(refs)
        if node.op not in ('placeholder', 'get_attr') and refs:
            module = self.fetch_attr(node.target) if node.op == 'call_module' else None
            op = _op(node.target, module)
            attrs, parameters = {}, {}
            # Repeated operands (x+x, x@x) are distinct argument positions.
            inputs = self.deps((node.args, node.kwargs))
            if module is not None:
                for key in ['stride', 'padding', 'padding_mode', 'dilation', 'groups', 'kernel_size', 'dim', 'eps', 'approximate', 'start_dim', 'end_dim', 'normalized_shape', 'ceil_mode', 'count_include_pad', 'divisor_override', 'output_size', 'output_padding', 'num_groups', 'track_running_stats', 'negative_slope', 'alpha', 'beta', 'threshold']:
                    if hasattr(module, key):
                        attrs[key] = _json(getattr(module, key))
                for key, value in list(module.named_parameters(recurse=True)) + list(module.named_buffers(recurse=True)):
                    # Keep tied parameters tied, without losing their module source path.
                    identity = id(value)
                    if identity not in self.parameters:
                        self.parameters[identity] = self.tensor(value, f'parameter:{node.target}.{key}',
                                                               'parameter' if isinstance(value, nn.Parameter) else 'buffer', f'{node.target}.{key}')[0]
                    parameters[key] = self.parameters[identity]
            else:
                attrs['arguments'] = _json([a for a in node.args if not isinstance(a, fx.Node)])
                attrs.update({k: _json(v) for k, v in node.kwargs.items() if not isinstance(v, fx.Node)})
                if op in ('linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d', 'layernorm','rmsnorm'):
                    # ATen tensors are lifted get_attr values in exported.module().
                    for key, index in ([('weight',2)] if op=='rmsnorm' else [('weight', 1 if op != 'layernorm' else 2), ('bias', 2 if op != 'layernorm' else 3)]):
                        if len(node.args) > index:
                            dep = self.deps(node.args[index])
                            if dep:
                                parameters[key] = dep[0]
                                inputs = [i for i in inputs if i not in dep]
                    if op.startswith('conv'):
                        for key, index in [('stride', 3), ('padding', 4), ('dilation', 5), ('groups', 6)]:
                            if len(node.args) > index:
                                attrs[key] = _json(node.args[index])
                        if op.startswith('conv_transpose'):
                            attrs.pop('dilation',None)
                            for key, index in [('output_padding',5),('groups',6),('dilation',7)]:
                                if len(node.args)>index: attrs[key]=_json(node.args[index])
                if op=='dynamic_conv2d':
                    parameters['weight']=inputs[1]
                    for key,index in [('stride',2),('padding',3),('dilation',4),('groups',5)]:
                        if len(node.args)>index:attrs[key]=_json(node.args[index])
                    attrs['generatedWeight']=True
                if op in ('batchnorm','instancenorm','groupnorm'):
                    if op=='groupnorm':
                        positions=[('weight',2),('bias',3)]
                    elif str(node.target).startswith('aten.'):
                        positions=[('weight',1),('bias',2),('running_mean',3),('running_var',4)]
                    else:
                        positions=[('running_mean',1),('running_var',2),('weight',3),('bias',4)]
                    for key,index in positions:
                        dep=self.deps(node.args[index]) if len(node.args)>index else []
                        if dep:
                            parameters[key]=dep[0];inputs=[i for i in inputs if i not in dep]
                    if op=='groupnorm' and len(node.args)>1: attrs['num_groups']=node.args[1]
                    if op=='batchnorm' and len(node.args)>5: attrs['training']=node.args[5]
                    eps_index=7 if op=='batchnorm' else 4 if op=='groupnorm' else 7
                    if len(node.args)>eps_index: attrs['eps']=node.args[eps_index]
                if 'pool' in op:
                    positions = [('output_size',1)] if op.startswith('adaptive') else [('kernel_size',1),('stride',2),('padding',3)]
                    if op.startswith('maxpool'): positions += [('dilation',4),('ceil_mode',5)]
                    if op.startswith('avgpool'): positions += [('ceil_mode',4),('count_include_pad',5),('divisor_override',6)]
                    for key,index in positions:
                        if len(node.args)>index: attrs[key]=_json(node.args[index])
                if op=='concat' and len(node.args)>1: attrs['dim']=node.args[1]
                if op=='embedding' and len(node.args)>1:
                    weight=self.deps(node.args[0]);indices=self.deps(node.args[1])
                    if weight: parameters['weight']=weight[0];inputs=indices
                if op in ('layernorm','rmsnorm'):
                    if len(node.args)>1: attrs['normalized_shape']=_json(node.args[1])
                    eps_index=3 if op=='rmsnorm' else 4
                    if len(node.args)>eps_index: attrs['eps']=_json(node.args[eps_index])
                if op == 'softmax' and len(node.args)>1:
                    attrs['dim'] = _json(node.args[1])
                if getattr(node.target,'__name__','')=='safe_softmax':attrs['safe']=True
                if op in ('chunk','split'):
                    attrs['dim']=node.args[2] if len(node.args)>2 else node.kwargs.get('dim',0)
                if op=='getitem' and isinstance(node.args[1],int):
                    source=self.refs.get(node.args[0].name,[]) if isinstance(node.args[0],fx.Node) else []
                    if isinstance(self.env.get(node.args[0]),(tuple,list)):
                        inputs=[source[node.args[1]]];op='identity'
                if op=='stack':attrs['dim']=node.args[1] if len(node.args)>1 else node.kwargs.get('dim',0)
                if op=='repeat_kv':attrs['repeats']=node.args[1]
                if op=='repeat':attrs['repeats']=_json(node.args[1])
                if op=='attention_mask':
                    attrs['causal']=bool(node.args[2]);attrs['mask_kind']='bool' if len(inputs)>1 and self.env[node.args[1]].dtype==torch.bool else 'additive'
                if op=='arange':attrs['start']=node.args[0] if len(node.args)>1 else 0;attrs['step']=node.args[2] if len(node.args)>2 else 1
                if op == 'gelu':
                    attrs.setdefault('approximate', 'none')
                for key,index in ([('negative_slope',1)] if op=='leaky_relu' else [('alpha',1)] if op=='elu' else [('beta',1),('threshold',2)] if op=='softplus' else []):
                    if len(node.args)>index:attrs[key]=_json(node.args[index])
                if op == 'permute' and len(node.args)>1:
                    attrs['dims'] = _json(node.args[1])
                if op == 'transpose' and len(node.args)>2:
                    attrs['dims'] = _json(node.args[1:3])
                if op in ('select','slice'):
                    for key,index in [('dim',1),('index' if op=='select' else 'start',2),('end',3),('step',4)]:
                        if len(node.args)>index:attrs[key]=_json(node.args[index])
                if op in ('mean',):
                    dims=node.args[1] if len(node.args)>1 else node.kwargs.get('dim')
                    attrs['dims']=list(range(self.env[node.args[0]].ndim)) if dims is None else [dims] if isinstance(dims,int) else _json(dims)
                    attrs['keepdim']=bool(node.args[2]) if len(node.args)>2 else bool(node.kwargs.get('keepdim',False))
                if op=='divide' and attrs.get('rounding_mode') is not None:op='opaque'
                if op in ('add', 'multiply','subtract','divide','power'):
                    inputs=[]
                    for i, value in enumerate(node.args[:2]):
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            attrs.setdefault('scalarOperands',{})[str(i)]=value
                            inputs.extend(self.tensor(torch.tensor(value), f'{node.name}:scalar:{i}', 'constant'))
                        else:inputs.extend(self.deps(value))
            stack = node.meta.get('nn_module_stack', {})
            if op=='standard_normal':
                attrs.update({'distribution':'Normal(0,1)','stochastic':True,'shapeInputs':list(inputs),'valueSource':'captured PyTorch execution; browser never resamples'})
            trace=str(node.meta.get('stack_trace',''))
            if trace:attrs['codeTrace']=trace[-2500:]
            if 'modeling_llama.py' in trace and 'apply_rotary_pos_emb(' in trace:attrs['sourceFunction']='apply_rotary_pos_emb'
            if 'silicondevine_lowered_from' in node.meta:
                attrs['lowered_from']=node.meta['silicondevine_lowered_from']
            if 'silicondevine_attention_budget' in node.meta:
                attrs['boundary_reason']=node.meta['silicondevine_attention_budget']
            if op=='rmsnorm' and attrs.get('eps') is None:
                attrs['eps']=torch.finfo(result.dtype).eps
            if 'silicondevine_semantic' in node.meta:
                attrs['semantic']=node.meta['silicondevine_semantic']
            for tensor in self.tensors:
                if tensor['id'] in refs:
                    rank=len(tensor['shape'])
                    if rank==4 and 'silicondevine_lowered_from' in node.meta:
                        tensor['axes']=['B','H','Q','K'] if 'score' in attrs.get('semantic','') or 'probability' in attrs.get('semantic','') else ['B','H','T','D']
            if op.endswith('3d') and ('conv' in op or 'pool' in op):
                for tensor in self.tensors:
                    if tensor['id'] in inputs+refs:
                        tensor['spatialRank']=3
            group = str(node.target) if module is not None else (list(stack.values())[-1][0] if stack else node.name)
            self.nodes.append({'id': node.name, 'name': attrs.get('semantic',group or node.name), 'op': op, 'inputs': inputs, 'outputs': refs,
                               'parameters': parameters, 'attrs': attrs, 'group': group or node.name,
                               'source': str(type(module).__name__) if module is not None else (node.target.__module__+'.'+node.target.__name__ if getattr(node.target,'__module__','').startswith('silicondevine.') else str(node.target))})
        return result


def export_model(model, args, path=None, *, backend='fx', include_values=False, value_limit=4096,
                 total_value_limit=500000, name=None, kwargs=None, dynamic_shapes=None, tensor_sink=None, expand_attention=True):
    """Export an eval-mode copy using real example inputs; never mutates the original.

    backend='fx' preserves nn.Module calls. backend='export' captures functional ATen.
    Unknown operations remain opaque; values are optional bounded windows. Input
    shapes are concrete for this run, including when the source model is dynamic.
    """
    if backend not in ('fx', 'export'):
        raise ValueError("backend must be 'fx' or 'export'")
    if not 0 <= value_limit <= 100000 or not 0 <= total_value_limit <= 2000000:
        raise ValueError('Invalid value window budget')
    args = args if isinstance(args, tuple) else (args,)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        copied = copy.deepcopy(model).eval()
        inputs, keyword_inputs = copy.deepcopy((args, kwargs or {}))
        constraints={}
        try:
            if backend=='fx':
                if dynamic_shapes is not None: raise ValueError('dynamic_shapes requires backend="export"')
                graph=fx.symbolic_trace(copied)
                bound=inspect.signature(graph.forward).bind(*inputs,**keyword_inputs)
                bound.apply_defaults()
                flat=[bound.arguments[str(n.target)] for n in graph.graph.nodes if n.op=='placeholder']
            else:
                program=torch.export.export(copied,inputs,keyword_inputs,dynamic_shapes=dynamic_shapes)
                graph=program.module()
                from .lowering import inline_disabled_autocast
                graph=inline_disabled_autocast(graph)
                if expand_attention:
                    from .lowering import lower_attention
                    graph=lower_attention(graph)
                constraints={str(k):str(v) for k,v in program.range_constraints.items()}
                from torch.utils._pytree import tree_flatten
                flat=tree_flatten((inputs,keyword_inputs))[0]
            capture = Capture(graph, include_values, value_limit, total_value_limit, tensor_sink)
            capture.run(*flat, enable_io_processing=False)
        except Exception as exc:
            raise RuntimeError(f'{backend} capture failed: {exc}. Try the other backend or export a traceable submodule. No partial graph was saved.') from exc
    result = {'format': 'silicondevine', 'version': 1, 'name': name or type(model).__name__,
              'producer': {'backend': f'pytorch-{backend}', 'version': torch.__version__},
              'constraints': constraints,
              'modules': [{'path':path,'type':type(part).__name__} for path,part in copied.named_modules() if path],
              'tensors': capture.tensors, 'nodes': capture.nodes, 'inputs': capture.inputs, 'outputs': capture.outputs,
              'notes': ['Eval-mode forward graph for the supplied inputs; no backward graph or unexecuted control-flow branches.',
                        'Stored values are contiguous windows. Statistics describe stored values, not the entire tensor.',
                        'Unknown operators preserve topology and captured tensors; internal math is not invented.']}
    from .recognition import annotate_model
    annotate_model(result,copied)
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    return result
