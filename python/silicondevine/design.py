"""Architecture-designer units grounded in math and captured Python invocations."""
import ast
from pathlib import Path


def annotate_design(graph):
    nodes = graph['nodes']; tensors = {t['id']:t for t in graph['tensors']}
    files = {}; groups = {}; units = []
    for node in nodes:
        stack = node.get('attrs', {}).get('designSourceStack', [])
        if not stack: continue
        frame = stack[-1]
        # Module forward remains its own architecture. Outer helpers must not
        # swallow the called learned modules or their parameter projections.
        if frame['function'].split('.')[-1] == 'forward' or frame['call'] < 0: continue
        key = (frame['file'], frame['firstLine'], frame['function'], frame['call'])
        groups.setdefault(key, []).append(node)
    for key, members in groups.items():
        if len(members) < 2 or any(n.get('parameters') for n in members): continue
        file, first, function, _ = key
        try:
            if file not in files: files[file] = ast.parse(Path(file).read_text(encoding='utf-8-sig'))
            fn = next(n for n in ast.walk(files[file]) if isinstance(n, ast.FunctionDef)
                      and (n.lineno == first or n.decorator_list and n.decorator_list[0].lineno == first))
        except (OSError, UnicodeError, SyntaxError, StopIteration): continue
        ops = {n['op'] for n in members}
        name = function.split('.')[-1].replace('_', ' ')
        description = (ast.get_docstring(fn) or '').split('\n')[0][:180]
        kind = 'feature_construction' if ops & {'stack','concat'} else 'source_function'
        title = ('特征构造' if kind == 'feature_construction' else '数学函数') + ' · ' + name
        if ops & {'matmul'}: title = '矩阵运算 · ' + name
        units.append({'id': 'design.' + str(len(units)), 'kind': kind, 'name': title,
                      'nodes': [n['id'] for n in members], 'source': {'file':file,'line':first,'code':ast.unparse(fn)[:4000]},
                      'description': description or '按源码辅助函数聚合的数学单元；保留实际输入、输出与参数。',
                      'evidence': 'actual Python function invocation + captured tensor operations'})
    # Residuals require ancestor and equal-shape evidence established by recognition.
    for n in nodes:
        attrs = n.setdefault('attrs', {})
        if attrs.get('residualInput'):
            attrs['displayLabel'] = '残差连接'
            attrs['description'] = '保留输入表征，并叠加子层计算的更新：y = x + F(x)。'
    graph['designUnits'] = units
