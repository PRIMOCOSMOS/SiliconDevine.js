"""Preserve the Python calling contract alongside the captured tensor program.

Names come from signature binding and pytree paths, never from tensor identity.
Two arguments may alias the same tensor while keeping distinct meanings.
"""
import ast
import inspect
import math
import textwrap
import torch


def _literal(value):
    if isinstance(value, float) and not math.isfinite(value):
        return '-inf' if value == -math.inf else '+inf' if value == math.inf else 'nan'
    return value if value is None or type(value) in (bool, int, float, str) else repr(value)[:180]


def _signature(model):
    try: return inspect.signature(model.forward)
    except (TypeError, ValueError):
        return inspect.Signature([inspect.Parameter('args', inspect.Parameter.VAR_POSITIONAL),
                                  inspect.Parameter('kwargs', inspect.Parameter.VAR_KEYWORD)])


def _source(model):
    try:
        lines, start = inspect.getsourcelines(model.forward)
        tree = ast.parse(textwrap.dedent(''.join(lines)))
        fn = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        return fn, {'file': inspect.getsourcefile(model.forward), 'line': start}
    except (OSError, TypeError, SyntaxError, StopIteration):
        return None, None


def describe_inputs(model, args=None, kwargs=None):
    signature = _signature(model)
    supplied = signature.bind_partial(*(args or ()), **(kwargs or {}))
    provided = set(supplied.arguments)
    supplied.apply_defaults()
    fn, source = _source(model)
    specs = getattr(model, '_silicondevine_input_specs', {})

    def structure(value, path, unresolved=False):
        if unresolved:
            return {'kind': 'unknown', 'path': path, 'shape': ['?']}
        if isinstance(value, torch.Tensor):
            return {'kind': 'tensor', 'path': path, 'shape': list(value.shape),
                    'dtype': str(value.dtype).removeprefix('torch.')}
        if isinstance(value, dict):
            return {'kind': 'dict', 'path': path, 'children': [structure(v, f'{path}[{k!r}]') for k, v in value.items()]}
        if isinstance(value, (list, tuple)):
            return {'kind': 'tuple' if isinstance(value, tuple) else 'list', 'path': path,
                    'children': [structure(v, f'{path}[{i}]') for i, v in enumerate(value)]}
        return {'kind': 'none' if value is None else 'scalar', 'path': path,
                'value': _literal(value),
                'type': type(value).__name__}

    parameters = []
    for p in signature.parameters.values():
        required = p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        item = {'name': p.name, 'required': required, 'kind': p.kind.name.lower(),
                'status': 'provided' if p.name in provided else 'unresolved' if required else 'default',
                'structure': structure(supplied.arguments.get(p.name), p.name, required and p.name not in provided)}
        if p.annotation is not p.empty: item['annotation'] = inspect.formatannotation(p.annotation)
        if p.default is not p.empty:
            item['default'] = _literal(p.default)
        if p.name in specs:
            item['symbolicShape'] = specs[p.name].get('symbolicShape')
            item['shapeSource'] = specs[p.name].get('source')
            item['shapeEvidence'] = specs[p.name].get('derivedFrom')
        if fn:
            uses = []
            for n in ast.walk(fn):
                if isinstance(n, ast.Call) and any(isinstance(a, ast.Name) and a.id == p.name for a in ast.walk(n)):
                    expression = ast.unparse(n)
                    if len(expression) <= 220 and expression not in uses: uses.append(expression)
                # A None fallback is evidence of a derived/default input, not a required user tensor.
                if p.default is None and isinstance(n, ast.If) and isinstance(n.test, ast.Compare):
                    test = n.test
                    if (isinstance(test.left, ast.Name) and test.left.id == p.name and len(test.ops) == 1
                            and isinstance(test.ops[0], ast.Is) and isinstance(test.comparators[0], ast.Constant)
                            and test.comparators[0].value is None):
                        for assignment in n.body:
                            if isinstance(assignment, ast.Assign) and any(isinstance(t, ast.Name) and t.id == p.name for t in assignment.targets):
                                item['defaultExpression'] = ast.unparse(assignment.value)[:220]
                                item['derivedFrom'] = sorted({a.id for a in ast.walk(assignment.value)
                                    if isinstance(a, ast.Name) and a.id in signature.parameters and a.id != p.name})
            item['sourceUses'] = uses[:6]
        parameters.append(item)
    return {'signature': str(signature), 'source': source, 'parameters': parameters,
            'dimensions': 'Concrete shapes describe this capture only; ? means unknown.'}


def flat_input_labels(model, args, kwargs):
    """One descriptor per pytree leaf, including specialized scalar leaves."""
    from torch.utils._pytree import tree_flatten_with_path
    signature = _signature(model)
    positional = [p for p in signature.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    vararg = next((p.name for p in signature.parameters.values() if p.kind == p.VAR_POSITIONAL), 'args')
    labels = []
    for i, value in enumerate(args):
        name = positional[i].name if i < len(positional) else vararg
        root = name if i < len(positional) else f'{name}[{i-len(positional)}]'
        labels.extend(_leaves(value, name, root, tree_flatten_with_path))
    extra = next((p.name for p in signature.parameters.values() if p.kind == p.VAR_KEYWORD), 'kwargs')
    for key, value in kwargs.items():
        named = key in signature.parameters and signature.parameters[key].kind != inspect.Parameter.VAR_KEYWORD
        labels.extend(_leaves(value, key if named else extra, key if named else f'{extra}[{key!r}]', tree_flatten_with_path))
    return labels


def _leaves(value, name, root, flatten):
    from torch.utils._pytree import keystr
    return [(name, root + keystr(path)) for path, _ in flatten(value)[0]]


def attach_contract(graph, contract):
    by_name = {p['name']: p for p in contract['parameters']}
    for p in by_name.values(): p['tensors'] = []
    for tensor in graph['tensors']:
        name = tensor.get('inputParameter')
        if name not in by_name and tensor['id'] in graph['inputs']:
            label = tensor.get('semantic', tensor.get('source', ''))
            if label in by_name:
                name = label; tensor.update(inputParameter=name, inputPath=name, semantic=name)
        if name not in by_name: continue
        p = by_name[name]
        # Recognition may attach Q/K/V roles; keep the original caller name too.
        path = tensor.get('inputPath', name)
        role = tensor.get('semantic')
        tensor['semantic'] = path if not role or role == path else path + ' · ' + role
        p['tensors'].append({'id': tensor['id'], 'path': tensor.get('inputPath', name),
            'shape': tensor['shape'], 'dtype': tensor['dtype'],
            'consumers': [{'name': n.get('group') or n['name'], 'op': n['op']}
                          for n in graph['nodes'] if tensor['id'] in n['inputs']][:12]})
    for p in by_name.values():
        expression = p.get('defaultExpression')
        if not expression or p['status'] != 'default': continue
        try:
            call = ast.parse(expression, mode='eval').body
            if not isinstance(call, ast.Call): continue
            target = ast.unparse(call.func).split('.')[-1]
        except SyntaxError: continue
        candidates = [n for n in graph['nodes'] if n.get('source', '').split('.')[:2] == ['aten', target]]
        # Only a unique source call is safe to associate with an optional default.
        if len(candidates) == 1:
            output_ids = candidates[0]['outputs']
            p['generatedTensors'] = output_ids
            for t in graph['tensors']:
                if t['id'] in output_ids: t['semantic'] = p['name'] + ' · 源码默认值'
    graph['inputContract'] = contract
