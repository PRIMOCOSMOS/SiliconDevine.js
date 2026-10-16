"""Conservative constructor contracts, shared by discovery and recovery forms.

Only direct arguments to known torch.nn constructors are inferred. Parameter
names alone (notably ``input``) are never evidence that a value is a tensor.
"""
import ast
import inspect
import textwrap


_INTEGER_ARGUMENTS = {
    'Linear': ('in_features', 'out_features'),
    'Bilinear': ('in1_features', 'in2_features', 'out_features'),
    'Embedding': ('num_embeddings', 'embedding_dim'),
    'MultiheadAttention': ('embed_dim', 'num_heads'),
    'GroupNorm': ('num_groups', 'num_channels'),
    **{name: ('in_channels', 'out_channels') for name in (
        'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d')},
    **{name: ('num_features',) for name in (
        'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d')},
}


def parameter_constraints(callable_):
    """Return source-backed scalar contracts without executing model code."""
    target = callable_.__init__ if inspect.isclass(callable_) else callable_
    try:
        signature = inspect.signature(callable_)
        tree = ast.parse(textwrap.dedent(inspect.getsource(target)))
    except (TypeError, ValueError, OSError, IndentationError, SyntaxError):
        return {}
    namespace = getattr(target, '__globals__', {})
    # A parameter may be converted from a config object before reaching nn.
    # Nested helper scopes can shadow the same name; neither proves its input type.
    def scope_nodes(node):
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                yield from scope_nodes(child)
    definition = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    nodes = list(scope_nodes(definition)) if definition is not None else []
    reassigned = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    contracts = {}
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        parts = []
        expression = node.func
        while isinstance(expression, ast.Attribute):
            parts.insert(0, expression.attr)
            expression = expression.value
        if not isinstance(expression, ast.Name):
            continue
        value = namespace.get(expression.id)
        # Resolve only module dictionaries, avoiding arbitrary descriptors.
        for part in parts:
            value = vars(value).get(part) if inspect.ismodule(value) else None
        name = getattr(value, '__name__', '')
        if not inspect.isclass(value) or not getattr(value, '__module__', '').startswith('torch.nn.modules.'):
            continue
        names = _INTEGER_ARGUMENTS.get(name, ())
        arguments = dict(zip(names, node.args))
        arguments.update({k.arg: k.value for k in node.keywords if k.arg in names or k.arg == 'groups' and name.startswith('Conv')})
        for argument, expr in arguments.items():
            if isinstance(expr, ast.Name) and expr.id in signature.parameters and expr.id not in reassigned:
                contracts[expr.id] = {'kind': 'integer', 'minimum': 1 if argument == 'groups' else 0,
                                      'derivedFrom': f'torch.nn.{name}.{argument}'}
    return contracts


def invalid_parameters(callable_, values):
    """Validate known primitive types; leave generic and user config types intact."""
    signature = inspect.signature(callable_)
    constraints = parameter_constraints(callable_)
    failures = []
    for name, value in values.items():
        parameter = signature.parameters.get(name)
        if parameter is None:
            if not any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
                failures.append((name, '不是此入口的构造参数'))
            continue
        annotation = parameter.annotation
        kind = {int: 'integer', float: 'number', bool: 'boolean', str: 'text'}.get(annotation) if isinstance(annotation, type) else None
        if isinstance(annotation, str):
            kind = {'int': 'integer', 'float': 'number', 'bool': 'boolean', 'str': 'text'}.get(annotation.strip("'\""))
        contract = constraints.get(name, {})
        kind = kind or contract.get('kind')
        valid = {'integer': type(value) is int, 'number': type(value) in (int, float),
                 'boolean': type(value) is bool, 'text': isinstance(value, str)}
        if kind in valid and not valid[kind]:
            label = {'integer': '整数', 'number': '数字', 'boolean': '布尔值', 'text': '文字'}[kind]
            source = '（用于 ' + contract['derivedFrom'] + '）' if contract else ''
            failures.append((name, f'需要{label}{source}，当前为 {type(value).__name__}'))
        elif 'minimum' in contract and type(value) is int and value < contract['minimum']:
            failures.append((name, f'必须不小于 {contract["minimum"]}'))
    return failures


def call_constructor(callable_, values):
    """Bind positional-only parameters while preserving defaults and **kwargs."""
    signature = inspect.signature(callable_)
    positional = [p for p in signature.parameters.values() if p.kind == p.POSITIONAL_ONLY]
    last = max((i for i, p in enumerate(positional) if p.name in values), default=-1)
    args = [values[p.name] if p.name in values else p.default for p in positional[:last + 1]]
    kwargs = {k: v for k, v in values.items() if k not in {p.name for p in positional}}
    signature.bind(*args, **kwargs)
    return callable_(*args, **kwargs)
