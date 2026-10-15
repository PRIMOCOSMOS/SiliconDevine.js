"""Non-numerical model views when example inputs cannot be established.

FX edges are traced code dependencies. A module inventory has ownership only:
it must never invent a sequential data-flow graph from registration order.
"""
import inspect
import torch
from torch import fx


def structural_model(model, reason):
    from .exporter import _op
    graph = {'format': 'silicondevine', 'version': 1, 'name': type(model).__name__,
             'tensors': [], 'nodes': [], 'inputs': [], 'outputs': [],
             'modules': [{'path': p, 'type': type(m).__name__} for p, m in model.named_modules() if p],
             'producer': {'backend': 'symbolic'},
             'notes': ['结构预览 · 输入尺寸尚未确定，未执行数值计算。', reason]}
    def tensor(key, role='activation', shape=None, label=None):
        graph['tensors'].append({'id': key, 'shape': shape or ['?'], 'dtype': 'unknown',
                                'role': role, 'representation': 'aggregate', 'semantic': label or key})
        return key
    def refs(value):
        found = []
        fx.map_arg(value, lambda n: found.append(n.name))
        return list(dict.fromkeys(found))
    try:
        traced = fx.symbolic_trace(model)
        for n in traced.graph.nodes:
            if n.op == 'placeholder':
                graph['inputs'].append(tensor(n.name, 'input', label=str(n.target)))
            elif n.op == 'get_attr':
                obj = traced
                for part in str(n.target).split('.'): obj = getattr(obj, part)
                tensor(n.name, 'parameter' if isinstance(obj, torch.nn.Parameter) else 'constant',
                       list(obj.shape) if isinstance(obj, torch.Tensor) else None, str(n.target))
            elif n.op == 'output':
                graph['outputs'] = refs(n.args)
            else:
                module = traced.get_submodule(n.target) if n.op == 'call_module' else None
                inputs = refs((n.args, n.kwargs)); parameters = {}
                if module is not None:
                    for key, value in module.named_parameters(recurse=False):
                        ident = f'{n.name}.parameter.{key}'
                        parameters[key] = tensor(ident, 'parameter', list(value.shape), f'{n.target}.{key}')
                out = tensor(n.name)
                label = type(module).__name__ if module is not None else getattr(n.target, '__name__', str(n.target))
                graph['nodes'].append({'id': n.name, 'name': label, 'op': 'symbolic',
                    'inputs': inputs, 'outputs': [out], 'parameters': parameters,
                    'group': str(n.target) if module is not None else n.name,
                    'source': str(n.target), 'attrs': {'symbolic': True, 'knownOperator': _op(n.target, module),
                    'displayLabel': label,
                    'description': '源码依赖已确认；输入输出尺寸待定。未生成数值和计算动画。'}})
        if not graph['nodes']: raise ValueError('FX did not expose operations')
        graph['provenance'] = {'mode': 'symbolic', 'topology': 'fx-dependencies', 'numericalExecution': False}
    except Exception as exc:
        graph['nodes'] = []; graph['tensors'] = []; graph['outputs'] = []; graph['inputs'] = []
        for p in inspect.signature(model.forward).parameters.values():
            if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                graph['inputs'].append(tensor('input.' + p.name, 'input', label=p.name))
        # A separate node per registered leaf; no edges claim an execution order.
        leaves = [(p, m) for p, m in model.named_modules() if not list(m.children())]
        for i, (path, module) in enumerate(leaves):
            name = path or type(model).__name__; ident = f'module.{i}'
            params = {key: tensor(f'{ident}.{key}', 'parameter', list(v.shape), name + '.' + key)
                      for key, v in module.named_parameters(recurse=False)}
            graph['nodes'].append({'id': ident, 'name': name, 'op': 'symbolic', 'group': path,
                'inputs': [], 'outputs': [tensor(ident + '.out')], 'parameters': params,
                'attrs': {'symbolic': True, 'knownOperator': _op('', module),
                          'displayLabel': type(module).__name__ + ' · ' + name,
                          'description': '已注册模块；所属关系可展开，forward 调用顺序与数据连接尚未确定。'}})
        graph['provenance'] = {'mode': 'symbolic', 'topology': 'module-inventory', 'numericalExecution': False,
                               'traceError': str(exc)[-1200:]}
        graph['notes'][0] = '模块结构预览 · 数据连接与尺寸待定，图中不绘制推测的数据流。'
    from .source_analysis import annotate_source
    annotate_source(graph, model)
    from .input_contract import describe_inputs, attach_contract
    attach_contract(graph, describe_inputs(model))
    return graph
