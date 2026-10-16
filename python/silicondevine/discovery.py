"""Static multi-file discovery followed by real, bounded PyTorch capture.

Scanning never executes repository code. Import/capture executes the selected model
in the watch worker, with the same trust requirements as running it in Python.
"""
import ast
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import copy

EXCLUDED = {'.git', '.venv', 'venv', 'env', '__pycache__', 'node_modules', 'build', 'dist'}


def source_files(sources):
    files = set()
    for source in sources:
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            for directory, dirs, names in os.walk(path):
                dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not d.startswith('.'))
                files.update(Path(directory, name) for name in names if name.endswith('.py'))
                if len(files) > 5000:
                    raise ValueError('项目超过 5000 个 Python 文件；请选择模型所在子目录。')
        elif path.is_file() and path.suffix == '.py':
            files.add(path)
        else:
            raise ValueError(f'模型路径不存在或不是 Python 文件：{path}')
    if not files:
        raise ValueError('所选目录没有 Python 文件。')
    return sorted(files)


def import_identity(path):
    parts = [] if path.stem == '__init__' else [path.stem]
    parent = path.parent
    while (parent / '__init__.py').is_file():
        parts.insert(0, parent.name)
        parent = parent.parent
    return '.'.join(parts) or path.parent.name, parent


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + '.' + node.attr
    return ''


def scan_project(sources):
    """AST-only inventory; also follows local imports of individually selected files."""
    pending = source_files(sources)
    # Namespace packages need no __init__.py. Locate roots from real local
    # absolute imports (network.blocks -> repository/network/blocks.py).
    namespace_roots = set()
    for path in pending:
        try:
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        except (SyntaxError, UnicodeError):
            continue
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and not n.level and n.module]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        for ancestor in list(path.parents)[:4]:
            for name in imports:
                if '.' not in name:
                    continue
                target = ancestor.joinpath(*name.split('.'))
                if target.with_suffix('.py').is_file() or (target / '__init__.py').is_file():
                    namespace_roots.add(ancestor)
    def identity(path):
        matches = [r for r in namespace_roots if path.is_relative_to(r)]
        if matches:
            root = max(matches, key=lambda p: len(p.parts))
            parts = list(path.relative_to(root).with_suffix('').parts)
            if parts[-1] == '__init__': parts.pop()
            return '.'.join(parts), root
        return import_identity(path)
    records, warnings, errors, visited = {}, [], [], set()
    roots = {identity(p)[1] for p in pending}
    while pending:
        path = pending.pop(0)
        if str(path) in visited:
            continue
        visited.add(str(path))
        if len(records) >= 5000:
            raise ValueError('本地依赖超过 5000 个文件；请缩小项目范围。')
        name, root = identity(path)
        parent = path.parent
        while parent != root and (parent / '__init__.py').is_file():
            init_file = parent / '__init__.py'
            if str(init_file) not in records and init_file != path:
                pending.append(init_file)
            parent = parent.parent
        try:
            tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            warnings.append(f'{path}: {exc}')
            errors.append({'file':str(path),'line':getattr(exc,'lineno',None),'message':str(exc)})
            continue
        aliases, imports = {}, []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for item in node.names:
                    aliases[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
                    imports.append(item.name)
            elif isinstance(node, ast.ImportFrom):
                package = name if path.stem == '__init__' else name.rpartition('.')[0]
                base = node.module or ''
                if node.level:
                    pieces = package.split('.') if package else []
                    base = '.'.join(pieces[:len(pieces)-node.level+1] + ([base] if base else []))
                imports.append(base)
                for item in node.names:
                    aliases[item.asname or item.name] = '.'.join(filter(None, [base, item.name]))
                    imports.append(aliases[item.asname or item.name])
        local_classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
        def resolve(value):
            head, dot, tail = value.partition('.')
            return aliases.get(head, name + '.' + head if head in local_classes else head) + (dot + tail if dot else '')
        classes, functions = [], []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            classes.append({'id': name + ':' + node.name, 'symbol': name + '.' + node.name,
                            'name': node.name, 'bases': [resolve(dotted(b)) for b in node.bases],
                            'references': sorted({resolve(dotted(n.func)) for n in ast.walk(node) if isinstance(n, ast.Call)}),
                            'line': node.lineno, 'file': str(path), 'module': name, 'root': str(root)})
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            assigned = {}
            for statement in node.body:
                if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
                    for target in statement.targets:
                        if isinstance(target, ast.Name):
                            assigned[target.id] = resolve(dotted(statement.value.func))
            returns = [n.value for n in ast.walk(node) if isinstance(n, ast.Return)]
            targets = [resolve(dotted(v.func)) if isinstance(v, ast.Call) else assigned.get(v.id, '') if isinstance(v, ast.Name) else '' for v in returns]
            if targets and all(targets):
                functions.append({'id': name + ':' + node.name, 'symbol': name + '.' + node.name, 'kind': 'factory',
                                  'name': node.name, 'returns': targets, 'references': targets,
                                  'line': node.lineno, 'file': str(path), 'module': name, 'root': str(root)})
        records[str(path)] = {'path': str(path), 'module': name, 'root': str(root),
                             'imports': sorted(set(imports)), 'aliases': aliases, 'classes': classes, 'functions': functions,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for dependency in imports:
            for base in roots | {root}:
                relative = base.joinpath(*dependency.split('.'))
                for candidate in (relative.with_suffix('.py'), relative / '__init__.py'):
                    if candidate.is_file() and str(candidate.resolve()) not in records:
                        pending.append(candidate.resolve())
    classes = [c for r in records.values() for c in r['classes']]
    # Resolve package re-exports, e.g. `from . import Block` where __init__.py
    # imports Block from a sibling file. This also covers aliased base classes.
    exports = {r['module'] + '.' + key: value for r in records.values() for key, value in r['aliases'].items()}
    def canonical(value):
        seen = set()
        while value not in seen:
            seen.add(value)
            match = next((key for key in sorted(exports, key=len, reverse=True) if value == key or value.startswith(key + '.')), None)
            if not match:
                break
            value = exports[match] + value[len(match):]
        return value
    for candidate in classes:
        candidate['bases'] = [canonical(v) for v in candidate['bases']]
        candidate['references'] = [canonical(v) for v in candidate['references']]
    recognized = {'torch.nn.Module', 'torch.nn.modules.module.Module', 'transformers.PreTrainedModel'}
    selected = []
    for _ in range(len(classes) + 1):
        added = [c for c in classes if c['symbol'] not in recognized and any(b in recognized or b.startswith('transformers.') and b.endswith('PreTrainedModel') for b in c['bases'])]
        if not added:
            break
        selected.extend(added)
        recognized.update(c['symbol'] for c in added)
    functions = [f for r in records.values() for f in r['functions']]
    factories = []
    for f in functions:
        targets = [canonical(t) for t in f['returns']]
        if all(t in recognized or t.startswith('torch.nn.') and t.rsplit('.', 1)[-1][:1].isupper() for t in targets):
            f['references'] = targets
            factories.append(f)
    selected.extend(factories)
    used = {ref for c in selected for ref in c['references'] + c.get('bases', []) if ref != c['symbol']}
    candidates = [c for c in selected if c['symbol'] not in used]
    return {'files': list(records.values()), 'candidates': candidates, 'modules': selected, 'warnings': warnings, 'errors':errors}


def repository_defaults(report):
    """Read literal argparse defaults and matching constructor calls, never run training."""
    from .repository_config import infer_repository_config
    return infer_repository_config(report)


def project_config(sources, config=None, read=True):
    paths = [Path(p).resolve() for p in sources]
    base = paths[0] if paths[0].is_dir() else Path(os.path.commonpath([str(p.parent) for p in paths]))
    path = Path(config).resolve() if config else base / 'silicondevine.json'
    if read and config and not path.is_file():
        raise ValueError(f'配置文件不存在：{path}')
    settings = json.loads(path.read_text(encoding='utf-8-sig')) if read and path.is_file() else {}
    if not isinstance(settings, dict):
        raise ValueError('silicondevine.json 顶层必须是 JSON 对象。')
    from .assistance import validate_settings
    validate_settings(settings)
    return settings, path


def _value(spec):
    if isinstance(spec,dict) and '$tensor' in spec:return _tensor(spec['$tensor'])
    if isinstance(spec, dict) and ('$class' in spec or '$enum' in spec):
        module, name = spec.get('$class', spec.get('$enum')).split(':', 1)
        cls = importlib.import_module(module)
        for part in name.split('.'): cls = getattr(cls, part)
        if '$enum' in spec: return cls[spec['name']]
        return cls(**{k: _value(v) for k, v in spec.get('kwargs', {}).items()})
    if isinstance(spec, dict) and '$namespace' in spec:
        from types import SimpleNamespace
        return SimpleNamespace(**{k: _value(v) for k,v in spec['$namespace'].items()})
    if isinstance(spec, dict):
        return {k: _value(v) for k, v in spec.items()}
    if isinstance(spec, list):
        return [_value(v) for v in spec]
    return spec


def _tensor(spec):
    import torch
    if isinstance(spec,dict) and '$tensor' in spec:return _tensor(spec['$tensor'])
    if not isinstance(spec, dict) or 'shape' not in spec:
        if isinstance(spec, list):
            return [_tensor(value) for value in spec]
        if isinstance(spec, dict) and not any(k in spec for k in ('$class','$namespace','$enum')):
            return {key: _tensor(value) for key, value in spec.items()}
        return _value(spec)
    shape = spec['shape']
    if not isinstance(shape, list) or len(shape) > 16 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError('输入 shape 必须是正整数列表。')
    import math
    if math.prod(shape) > 2_000_000:
        raise ValueError('单个示例输入超过 200 万元素；请在配置中缩小示例尺寸。')
    dtype = getattr(torch, spec.get('dtype', 'float32'), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f'未知输入 dtype: {spec.get("dtype")}')
    if 'values' in spec:
        return torch.tensor(spec['values'], dtype=dtype).reshape(shape)
    fill = spec.get('fill', 'random')
    if fill not in ('random', 'zeros', 'ones'):
        raise ValueError('输入 fill 必须是 random、zeros 或 ones。')
    if fill in ('zeros', 'ones'):
        return getattr(torch, fill)(shape, dtype=dtype)
    if dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
        return torch.randint(0, spec.get('high', 2), shape, dtype=dtype)
    return torch.randn(shape, dtype=dtype)


def _inferred_inputs(model, size):
    import torch
    from torch import nn
    signature = inspect.signature(model.forward)
    parameters = [p for p in signature.parameters.values() if p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    if not parameters:
        return (), {}, []
    modalities = getattr(model, 'modalities', None)
    first = getattr(model, 'first_conv', None)
    if len(parameters) == 1 and isinstance(first, nn.ModuleDict) and isinstance(modalities, (list, tuple)) and all(m in first and isinstance(first[m], nn.Conv2d) for m in modalities):
        shape = getattr(model, '_silicondevine_input_shape', [size, size])
        descriptors = {m: {'shape': [1, first[m].in_channels, *shape]} for m in modalities}
        values = {m: _tensor(d) for m, d in descriptors.items()}
        p = parameters[0]
        return (() if p.kind == p.KEYWORD_ONLY else (values,)), ({p.name: values} if p.kind == p.KEYWORD_ONLY else {}), [{'name': p.name, 'input': descriptors, 'derivedFrom': 'ModuleDict + modalities + constructor original_shape'}]
    consumers = {}
    try:
        traced = torch.fx.symbolic_trace(model)
        for node in traced.graph.nodes:
            if node.op != 'placeholder':
                continue
            queue, seen = list(node.users), set()
            while queue:
                current = queue.pop(0)
                if current in seen:
                    continue
                seen.add(current)
                if current.op == 'call_module':
                    module = traced.get_submodule(current.target)
                    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Embedding, nn.MultiheadAttention)):
                        consumers[str(node.target)] = module
                        break
                queue.extend(current.users)
    except Exception:
        pass
    supported = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Embedding, nn.MultiheadAttention))]
    args, kwargs, evidence = [], {}, []
    for p in parameters:
        declared = getattr(model, '_silicondevine_input_specs', {}).get(p.name)
        if declared:
            value = _tensor(declared)
            if p.kind == p.KEYWORD_ONLY: kwargs[p.name] = value
            else: args.append(value)
            evidence.append({'name': p.name, 'input': {'shape': declared['shape']},
                             'symbolicShape': declared['symbolicShape'], 'source': declared['source'],
                             'derivedFrom': declared['derivedFrom']})
            continue
        module = consumers.get(p.name)
        if module is None and len(parameters) == 1 and supported:
            module = supported[0]
        if module is None:
            raise ValueError(f'源码和模块约束尚不足以确定 forward 输入 {p.name} 的结构，继续展示可确认的模型结构。')
        if isinstance(module, nn.Embedding):
            descriptor = {'shape': [1, 4], 'dtype': 'int64', 'high': module.num_embeddings}
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            descriptor = {'shape': [1, module.in_channels] + [size] * len(module.kernel_size)}
        elif isinstance(module, nn.MultiheadAttention):
            descriptor = {'shape': ([1, 4] if module.batch_first else [4, 1]) + [module.embed_dim]}
        else:
            descriptor = {'shape': ([1, 4] if size >= 16 else [1]) + [module.in_features]}
        parameter = next(module.parameters(), None)
        if not isinstance(module, nn.Embedding) and parameter is not None and (parameter.is_floating_point() or parameter.is_complex()):
            descriptor['dtype'] = str(parameter.dtype).removeprefix('torch.')
        value = _tensor(descriptor)
        if p.kind == p.KEYWORD_ONLY:
            kwargs[p.name] = value
        else:
            args.append(value)
        evidence.append({'name': p.name, 'input': descriptor, 'derivedFrom': type(module).__name__})
    return tuple(args), kwargs, evidence


def capture_project(sources, destination, backend='auto', entry=None, config=None, overrides=None):
    from .project_imports import project_imports
    report = scan_project(sources)
    with project_imports(report):
        return _capture_project(report, sources, destination, backend, entry, config, overrides)


def _capture_project(report, sources, destination, backend, entry, config, overrides):
    import torch
    from .exporter import export_model
    from .live import Snapshot
    from .assistance import problem, import_schema, validate_settings
    from .automatic import source_evidence, resolve_entry, constructor_values, example_inputs
    from .constructor_contract import invalid_parameters, call_constructor
    if not report['modules'] and report.get('errors'):
        error=report['errors'][0]
        raise problem('syntax_error','模型源文件无法解析','请修正提示位置的代码并保存，应用会自动重试。',
                      file=error['file'],line=error['line'],detail=error['message'])
    overrides = validate_settings(overrides or {})
    try:
        settings, config_path = project_config(sources, config, read=not overrides.get('ignore_project_config', False))
    except (ValueError, OSError) as exc:
        if config and not overrides.get('ignore_project_config'):
            raise problem('config_invalid', '指定的导入配置无法读取', '可以在界面中选择忽略此配置，再由应用自动解析模型。',
                          detail=str(exc), schema={'candidates':report['modules'],'settings':{},'constructor':[],'inputs':[]}) from exc
        settings={};_,config_path=project_config(sources,config,read=False)
        report['warnings'].append('未采用无法读取的旧导入配置，继续自动解析：'+str(exc))
    inferred_settings = repository_defaults(report)
    declared_entry=overrides.get('entry') or entry or settings.get('entry')
    if inferred_settings and (not declared_entry or inferred_settings['settings']['entry']==declared_entry):
        defaults=inferred_settings['settings']
        settings={**defaults,**settings,'init':{**defaults.get('init',{}),**settings.get('init',{})}}
    settings = {**settings, **overrides}
    if overrides.get('input_mode')=='auto':
        settings.pop('inputs',None);settings.pop('kwargs',None)
    entry = overrides.get('entry') or entry or settings.get('entry')
    choices = report['modules'] if entry else report['candidates']
    matches = [c for c in choices if not entry or c['id'] == entry or c['name'] == entry]
    automatic={'entry':None,'constructor':[],'inputAttempts':[]}
    evidence_source=source_evidence(report)
    if not entry and len(matches)!=1:
        inferred_entry=resolve_entry(report,evidence_source)
        if inferred_entry:
            matches=[inferred_entry]
            automatic['entry']={'entry':inferred_entry['id'],'basis':'unique source construction among root candidates'}
    if len(matches) != 1:
        options = '\n'.join('  ' + c['id'] for c in (matches or choices)) or '未识别到 nn.Module 子类或返回模型的函数'
        raise problem('entry_required', '无法唯一确定模型入口', '已检查模块组成和源码构造调用，仍存在多个独立入口。请只确认要展示的模型。' if choices else '未找到模型类。请选择包含 nn.Module 定义的项目目录。',
                      fields=['entry'],schema={'candidates':report['candidates'] or report['modules'],'settings':settings,'constructor':[],'inputs':[]}, detail=options)
    selected = matches[0]
    if automatic['entry'] is None:
        automatic['entry']={'entry':selected['id'],'basis':'provided setting' if declared_entry else 'repository defaults' if inferred_settings else 'unique root module'}
    # Explicit entry selection can name a child; inspect its construction evidence too.
    evidence_source=source_evidence(report,selected)
    for root in sorted({r['root'] for r in report['files']}):
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        cls = getattr(importlib.import_module(selected['module']), selected['name'])
    except ModuleNotFoundError as exc:
        raise ValueError(f'模型依赖未安装或导入路径不完整：{exc.name}。请选择仓库根目录，并使用该仓库所需的 Python 环境。') from exc
    if selected.get('kind') != 'factory' and (not inspect.isclass(cls) or not issubclass(cls, torch.nn.Module)):
        raise ValueError(f'{selected["id"]} 运行时并非 torch.nn.Module。')
    settings['init'],automatic['constructor']=constructor_values(selected,cls,evidence_source,settings.get('init',{}))
    invalid = invalid_parameters(cls, settings['init'])
    if invalid:
        raise problem('constructor_invalid', '构造参数类型不匹配',
                      '；'.join(name + ' ' + reason for name, reason in invalid) + '。请修正这些字段，或恢复自动解析。',
                      fields=[name for name, _ in invalid], schema=import_schema(selected, cls, settings=settings, report=report))
    init = settings['init']
    missing = [p.name for p in inspect.signature(cls).parameters.values() if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) and p.name not in init]
    if missing:
        raise problem('constructor_required', '仍有构造参数无法确定', '已读取源码构造示例和配置默认值。需要填写：' + ', '.join(missing) + '；其余参数已处理。',
                      fields=missing,schema=import_schema(selected, cls, settings=settings, report=report))
    # Count shapes without allocating weight storage before constructing on CPU.
    try:
        with torch.random.fork_rng(), torch.device('meta'):
            torch.manual_seed(settings.get('seed', 0))
            probe = call_constructor(cls, _value(init))
        if not isinstance(probe, torch.nn.Module):
            raise ValueError('入口函数必须返回 torch.nn.Module 实例。')
        count = sum(p.numel() for p in probe.parameters())
        buffers = sum(p.numel() for p in probe.buffers())
        del probe
    except Exception as exc:
        if isinstance(exc, (ModuleNotFoundError, ImportError, FileNotFoundError)): raise
        import traceback
        meta_unsupported = 'meta' in str(exc).lower() and isinstance(exc, (RuntimeError, NotImplementedError))
        raise problem('constructor_meta_unsupported' if meta_unsupported else 'constructor_failed',
                      '模型构造不支持无内存预检' if meta_unsupported else '模型构造预检未通过',
                      '构造函数需要读取张量数值，无法在 meta 设备上检查大小。请将数值计算移到 forward，或通过 Python 接口 show() 传入已构造的模型。' if meta_unsupported else '请核对构造参数及下方源码位置；也可恢复自动解析后重新选择模型。',
                      detail=''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-10000:],
                      schema=import_schema(selected, cls, settings=settings, report=report)) from exc
    limit = settings.get('max_parameters', 25_000_000)
    if count + buffers > limit:
        raise problem('model_budget', '模型超过当前计算预算', f'模型含 {count:,} 个参数及 {buffers:,} 个缓冲元素，当前预算 {limit:,}。可减小构造尺寸，或根据可用内存调整预算。',
                      schema=import_schema(selected, cls, settings=settings, report=report))
    torch.manual_seed(settings.get('seed', 0))
    init = _value(init)
    model = call_constructor(cls, init).cpu().eval()
    if 'original_shape' in init:
        model._silicondevine_input_shape = list(init['original_shape'])
    explicit = 'inputs' in settings or 'kwargs' in settings
    failures = []; validated = False; evidence = []
    source_examples=[] if explicit else example_inputs(selected,cls,evidence_source,settings['init'])
    for trial in ([{'size':8}] if explicit else [{'source':s} for s in source_examples]+[{'size':s} for s in (8,16,32,64)]):
        source_example=trial.get('source')
        if explicit or source_example:
            descriptors=source_example or settings
            try:
                args = tuple(_tensor(s) for s in descriptors.get('inputs', []))
                kwargs = {k: _tensor(v) for k, v in descriptors.get('kwargs', {}).items()}
            except (ValueError, TypeError, RuntimeError) as exc:
                if not explicit:
                    failures.append(str(exc));continue
                raise problem('inputs_invalid', '示例输入设置有误', '请检查输入的尺寸、类型与内容。', detail=str(exc),
                              schema=import_schema(selected, cls, model, settings, report)) from exc
            evidence = {'inputs': descriptors.get('inputs', []), 'kwargs': descriptors.get('kwargs', {})}
            if source_example:evidence.update(source=source_example['source'],line=source_example['line'],derivedFrom='source forward example; tensor values generated from declarations')
        else:
            size=trial['size']
            from .input_shapes import source_input_specs
            model._silicondevine_input_specs = source_input_specs(report, model, size)
            try:
                args, kwargs, evidence = _inferred_inputs(model, size)
            except (ValueError, TypeError) as exc:
                failures.append(str(exc));break
        try:
            with torch.no_grad(), torch.random.fork_rng():
                validation_args, validation_kwargs = copy.deepcopy((args, kwargs))
                copy.deepcopy(model)(*validation_args, **validation_kwargs)
            validated = True
            automatic['inputAttempts'].append({'basis':'provided example' if explicit else 'source forward example' if source_example else 'source shapes and PyTorch module constraints','validated':True})
            break
        except Exception as exc:
            if isinstance(exc, (ModuleNotFoundError, ImportError, FileNotFoundError)): raise
            failures.append(str(exc))
            automatic['inputAttempts'].append({'basis':'provided example' if explicit else 'source forward example' if source_example else 'source shapes and PyTorch module constraints','validated':False})
    if not validated and explicit:
        raise problem('inputs_invalid', '示例输入未通过模型验证', '请根据 forward 的输入要求调整尺寸或类型；也可以先查看结构。',
                      detail='\n'.join(failures[-2:]), schema=import_schema(selected, cls, model, settings, report))
    chosen_backend = overrides.get('backend') or (settings.get('backend', backend) if backend == 'auto' else backend)
    errors = []; graph = None
    snapshot = Snapshot(Path(destination) / 'tensors')
    if validated:
        for mode in (['export', 'execution', 'fx'] if chosen_backend == 'auto' else [chosen_backend]):
            try:
                graph = export_model(model, args, kwargs=kwargs, backend=mode, include_values=True, tensor_sink=snapshot)
                break
            except Exception as exc:
                errors.append(f'{mode}: {type(exc).__name__}: {str(exc)[-2000:]}')
    if graph is None:
        if explicit:
            raise problem('capture_failed', 'PyTorch 未能捕获该计算路径', '可尝试其他捕获接口，或先清除示例输入以查看结构。', detail='\n'.join(errors),
                          schema=import_schema(selected, cls, model, settings, report))
        from .symbolic import structural_model
        graph = structural_model(model, '\n'.join((failures + errors)[-3:]))
        snapshot.manifest = {}; mode = 'symbolic'
    graph['discovery'] = {'entry': selected['id'], 'files': [{k: f[k] for k in ('path', 'module', 'sha256', 'imports')} for f in report['files']],
                          'importOptions': import_schema(selected, cls, model, settings, report, evidence),
                          'automaticResolution':automatic,
                          'repositoryDefaults': inferred_settings,
                          'inputsInferred': not explicit, 'exampleInputs': evidence, 'config': str(config_path) if config_path.exists() else None,
                          'parameterCount': count, 'weights': 'constructor initialization; no checkpoint automatically loaded',
                          'backend': mode, 'fallbackErrors': errors, 'warnings': report['warnings'],
                          'inputStatus': 'unresolved' if mode == 'symbolic' else 'validated-example',
                          'inferenceErrors': failures,
                          'scope': 'Structure only; no numerical execution or shape claims.' if mode == 'symbolic' else 'Actual graph for the validated example inputs; execution backend records one path.'}
    Path(destination, 'capture.json').write_text(json.dumps({'model': graph, 'manifest': snapshot.manifest}, allow_nan=False), encoding='utf-8')
    return graph


def main():
    import argparse
    parser = argparse.ArgumentParser(description='静态分析多文件 PyTorch 模型，不执行仓库代码')
    parser.add_argument('sources', nargs='+')
    args = parser.parse_args()
    print(json.dumps(scan_project(args.sources), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
