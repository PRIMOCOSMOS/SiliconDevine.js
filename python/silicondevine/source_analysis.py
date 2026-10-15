"""Bounded Python source evidence shared by all capture backends.

Static calls describe possible code, never measured execution or invented edges.
Imported Python helpers are followed without calling them. Numeric dependencies
continue to come exclusively from PyTorch capture.
"""
import ast
import inspect
import re
import textwrap
from pathlib import Path
from collections import Counter


def describe_callable(fn, depth=0, seen=None):
    seen = set() if seen is None else seen
    identity = getattr(fn, '__func__', fn)
    if identity in seen or depth > 3 or len(seen) >= 64: return None
    seen.add(identity)
    try:
        lines, start = inspect.getsourcelines(fn)
        file = inspect.getsourcefile(fn)
        code = textwrap.dedent(''.join(lines))
        tree = ast.parse(code)
    except (OSError, TypeError, SyntaxError): return None
    calls = []; helpers = []
    namespace = getattr(identity, '__globals__', {})
    owner = getattr(fn, '__self__', None)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call): continue
        target = ast.unparse(node.func)
        calls.append({'target': target, 'line': start + node.lineno - 1})
        candidate = namespace.get(node.func.id) if isinstance(node.func, ast.Name) else None
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'self' and owner is not None):
            # Read bound methods from their class; don't evaluate properties.
            raw = inspect.getattr_static(type(owner), node.func.attr, None)
            if inspect.isfunction(raw): candidate = raw.__get__(owner, type(owner))
        if inspect.isfunction(candidate) or inspect.ismethod(candidate):
            if not getattr(candidate, '__module__', '').startswith('torch.'):
                child = describe_callable(candidate, depth + 1, seen)
                if child: helpers.append(child)
    return {'name': getattr(fn, '__qualname__', 'forward'), 'file': file, 'line': start,
            'code': code[:8000], 'calls': calls[:128], 'helpers': helpers,
            'conditions': [ast.unparse(n.test) for n in ast.walk(tree) if isinstance(n, (ast.If, ast.IfExp))][:32],
            'loops': [ast.unparse(n.iter if isinstance(n, ast.For) else n.test)
                      for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))][:32],
            'evidence': 'static Python source; branches and loops are not execution claims'}


def annotate_source(graph, model):
    modules = dict(model.named_modules()); cache = {}; files = {}
    for path, module in modules.items():
        cls = type(module)
        if cls.__module__.startswith('torch.'): continue
        if cls not in cache: cache[cls] = describe_callable(module.forward)
    graph['sourcePrograms'] = [{'module': p, **cache[type(m)]} for p, m in modules.items()
                              if cache.get(type(m))]
    def program(path):
        while path not in modules and '.' in path: path = path.rsplit('.', 1)[0]
        return cache.get(type(modules.get(path, model)))
    for node in graph['nodes']:
        attrs = node.setdefault('attrs', {})
        known = attrs.get('knownOperator', node['op'])
        if node['op'] == 'opaque':
            target = node.get('source', node['name'])
            attrs.setdefault('displayLabel', target.replace('aten.', '') + ' · 待展开')
            attrs.setdefault('description', f'已捕获调用 {target} 的输入与输出。当前尚无逐元素依赖映射，保留原始调用与源码供检查。')
        for path, line in reversed(re.findall(r'File "([^"]+)", line (\d+)', attrs.get('codeTrace', ''))):
            if 'site-packages' in path: continue
            try:
                if path not in files: files[path] = Path(path).read_text(encoding='utf-8-sig').splitlines()
                attrs['sourceExcerpt'] = {'file': path, 'line': int(line), 'code': files[path][int(line)-1].strip()}
                break
            except (OSError, UnicodeError, IndexError): continue
        if node['op'] == 'symbolic':
            p = program(node.get('group', ''))
            if p:
                attrs['sourceExcerpt'] = {k:p[k] for k in ('file','line','code')}
                attrs['sourceCalls'] = p['calls']
            attrs['description'] = ('已识别基础算子：' + known + '。' if known != 'opaque' else '') + attrs.get('description', '')
    by_id = {n['id']: n for n in graph['nodes']}
    for unit in graph.get('composition', {}).get('units', []):
        labels={'linear':'全连接投影','matmul':'矩阵乘法','conv1d':'一维卷积','conv2d':'二维卷积','conv3d':'三维卷积',
                'attention':'注意力','softmax':'概率归一化','layernorm':'层归一化','rmsnorm':'RMS 归一化',
                'batchnorm':'批归一化','silu':'SiLU 激活','gelu':'GELU 激活','relu':'ReLU 激活',
                'standard_normal':'高斯采样','embedding':'词嵌入','sigmoid':'Sigmoid 激活'}
        names=[]
        for key in unit['nodes']:
            node=by_id[key];label='残差连接' if node.get('attrs',{}).get('residualInput') else labels.get(node['op'])
            if label and label not in names:names.append(label)
        summary='、'.join(names[:8])
        visual = unit.setdefault('visual', {})
        visual.setdefault('description', ('包含 '+summary+'。' if summary else '按源码组织的数学函数与特征处理。')+'展开查看向量、矩阵与功能单元的关系。')
