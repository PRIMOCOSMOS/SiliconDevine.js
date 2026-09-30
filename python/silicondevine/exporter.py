"""Capture real tensors and executed graphs. No approximation of unknown operators.

FX preserves module boundaries. The export backend lowers to ATen and requires
a traceable graph. Only the supplied execution path and input shapes are shown.
"""
import copy
import json
import math
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
                 nn.MaxPool2d: 'maxpool2d', nn.AvgPool2d: 'avgpool2d'}
        return table.get(type(module), 'opaque')
    name = str(target).lower()
    # Specific names first; never classify arbitrary names by substring matches.
    parts = name.replace('aten.', '').split('.')
    base = parts[0]
    if name.startswith('<built-in function '):
        base = name.removeprefix('<built-in function ').removesuffix('>')
    aliases = {'add': 'add', 'mul': 'multiply', 'matmul': 'matmul', 'bmm': 'matmul', 'mm': 'matmul',
               'linear': 'linear', 'conv1d': 'conv1d', 'conv2d': 'conv2d', 'conv3d': 'conv3d',
               'relu': 'relu', 'gelu': 'gelu', 'sigmoid': 'sigmoid', 'tanh': 'tanh',
               'softmax': 'softmax', '_softmax': 'softmax', 'layer_norm': 'layernorm',
               'view': 'reshape', 'reshape': 'reshape', 'flatten': 'flatten',
               'permute': 'permute', 'transpose': 'transpose', 'cat': 'concat',
               'clone': 'identity', 'detach': 'identity', 'dropout': 'dropout'}
    return aliases.get(base, 'opaque')


class Capture(fx.Interpreter):
    def __init__(self, module, include_values, value_limit, total_limit):
        super().__init__(module)
        self.include_values, self.limit, self.remaining = include_values, value_limit, total_limit
        self.tensors, self.nodes, self.refs, self.inputs, self.outputs = [], [], {}, [], []
        self.parameters = {}

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
                finite = [v for v in values if math.isfinite(v)]
                if finite:
                    tensor['stats'] = {'min': min(finite), 'max': max(finite), 'absmax': max(abs(v) for v in finite)}
                self.remaining -= count
            self.tensors.append(tensor)
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
                for key in ['stride', 'padding', 'padding_mode', 'dilation', 'groups', 'kernel_size', 'dim', 'eps', 'approximate', 'start_dim', 'end_dim', 'normalized_shape']:
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
                if op in ('linear', 'conv1d', 'conv2d', 'conv3d', 'layernorm'):
                    # ATen tensors are lifted get_attr values in exported.module().
                    for key, index in [('weight', 1 if op != 'layernorm' else 2), ('bias', 2 if op != 'layernorm' else 3)]:
                        if len(node.args) > index:
                            dep = self.deps(node.args[index])
                            if dep:
                                parameters[key] = dep[0]
                                inputs = [i for i in inputs if i not in dep]
                    if op.startswith('conv'):
                        for key, index in [('stride', 3), ('padding', 4), ('dilation', 5), ('groups', 6)]:
                            if len(node.args) > index:
                                attrs[key] = _json(node.args[index])
                if op == 'softmax' and len(node.args)>1:
                    attrs['dim'] = _json(node.args[1])
                if op == 'gelu':
                    attrs.setdefault('approximate', 'none')
                if op == 'permute' and len(node.args)>1:
                    attrs['dims'] = _json(node.args[1])
                if op == 'transpose' and len(node.args)>2:
                    attrs['dims'] = _json(node.args[1:3])
                if op in ('add', 'multiply'):
                    for i, value in enumerate(node.args[:2]):
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            inputs.extend(self.tensor(torch.tensor(value), f'{node.name}:scalar:{i}', 'constant'))
            if op.startswith('conv') and (isinstance(attrs.get('padding'), str) or attrs.get('padding_mode', 'zeros') != 'zeros'):
                attrs['unsupported_reason'] = 'This convolution padding variant needs a dedicated dependency plugin.'
                op = 'opaque'
            if op == 'matmul':
                operands = [t for t in self.tensors if t['id'] in inputs]
                if any(len(t['shape']) < 2 for t in operands):
                    attrs['unsupported_reason'] = 'Vector dot/matvec is not the matrix-matrix dependency rule.'
                    op = 'opaque'
            stack = node.meta.get('nn_module_stack', {})
            group = str(node.target) if module is not None else (list(stack.values())[-1][0] if stack else node.name)
            self.nodes.append({'id': node.name, 'name': group or node.name, 'op': op, 'inputs': inputs, 'outputs': refs,
                               'parameters': parameters, 'attrs': attrs, 'group': group or node.name,
                               'source': str(type(module).__name__) if module is not None else str(node.target)})
        return result


def export_model(model, args, path=None, *, backend='fx', include_values=False, value_limit=4096,
                 total_value_limit=500000, name=None):
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
        inputs = copy.deepcopy(args)
        try:
            graph = fx.symbolic_trace(copied) if backend == 'fx' else torch.export.export(copied, inputs).module()
            capture = Capture(graph, include_values, value_limit, total_value_limit)
            capture.run(*inputs)
        except Exception as exc:
            raise RuntimeError(f'{backend} capture failed: {exc}. Try the other backend or export a traceable submodule. No partial graph was saved.') from exc
    result = {'format': 'silicondevine', 'version': 1, 'name': name or type(model).__name__,
              'producer': {'backend': f'pytorch-{backend}', 'version': torch.__version__},
              'tensors': capture.tensors, 'nodes': capture.nodes, 'inputs': capture.inputs, 'outputs': capture.outputs,
              'notes': ['Eval-mode forward graph for the supplied inputs; no backward graph or unexecuted control-flow branches.',
                        'Stored values are contiguous windows. Statistics describe stored values, not the entire tensor.',
                        'Unknown operators preserve topology and captured tensors; internal math is not invented.']}
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    return result
