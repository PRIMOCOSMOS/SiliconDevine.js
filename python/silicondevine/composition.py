"""Source-backed presentation hierarchy. Never rewrites the captured computation.

Scopes are partitions of executed nodes, not hand-authored replacement models.
Python loop iterations and nn.Module invocations remain separately inspectable.
"""
import ast
import inspect
import re
from pathlib import Path

# Only standard mathematical APIs receive universal aliases. User class and
# attribute names are evidence labels, never recognition or layout switches.
NAMES = {'rsample': '重参数采样', 'log_prob': '对数概率密度',
         'Normal': '高斯分布 · 均值与方差', 'cat': '张量拼接',
         'zeros_like': '零张量', 'ones_like': '单位张量'}
TYPES = {'Linear': '全连接', 'Conv2d': '二维卷积', 'Conv3d': '三维卷积',
         'LayerNorm': '层归一化', 'Upsample': '上采样'}


def annotate_composition(graph, model):
    if len(graph['nodes']) < 8:
        return
    try:
        file = Path(inspect.getsourcefile(type(model))).resolve()
        tree = ast.parse(file.read_text(encoding='utf-8-sig'))
        cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == type(model).__name__)
        forward = next(n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == 'forward')
    except (OSError, TypeError, StopIteration, SyntaxError):
        return
    by_id = {n['id']: n for n in graph['nodes']}
    producer = {t: n for n in graph['nodes'] for t in n['outputs']}
    # Assertions and distribution validation do not contribute to model outputs.
    live = set()
    def visit(t):
        n = producer.get(t)
        if n is None or n['id'] in live:
            return
        live.add(n['id'])
        for x in n['inputs'] + list(n.get('parameters', {}).values()):
            visit(x)
    for t in graph['outputs']:
        visit(t)
    nodes = [n for n in graph['nodes'] if n['id'] in live]
    lines = {}
    for n in nodes:
        for path, line in re.findall(r'File "([^"]+)", line (\d+)', n.get('attrs', {}).get('codeTrace', '')):
            if Path(path).resolve() == file and forward.lineno <= int(line) <= forward.end_lineno:
                lines[n['id']] = int(line)
                break
    # Partial source maps must not silently omit nodes.
    if not lines or len(lines) < len(nodes) * .9:
        return
    units = []
    modules = {m['path']: m for m in graph.get('modules', [])}

    def unit(name, members, children=None, source=None):
        ident = '@source.' + str(len(units))
        item = {'id': ident, 'name': name, 'nodes': [n['id'] for n in members],
                'children': children or [], 'source': source or str(file)}
        units.append(item)
        return ident

    def independent(groups):
        owners={n['id']:i for i,g in enumerate(groups) for n in g}
        # Transitive ancestry includes operations outside the candidate scope.
        # Shared parameters are allowed; activation dependencies are not.
        for i,g in enumerate(groups):
            pending=[t for n in g for t in n['inputs']]; seen=set()
            while pending:
                t=pending.pop()
                if t in seen: continue
                seen.add(t); n=producer.get(t)
                if not n: continue
                if n['id'] in owners and owners[n['id']] != i: return False
                pending.extend(n['inputs'])
        return True

    def native(members, prefix=''):
        if len(members) <= 1:
            return []
        # A modality key shared by several ModuleDicts forms a genuine branch.
        modality_keys = {}
        if not prefix:
            dictionaries = [p for p, m in modules.items() if m['type'] in ('ModuleDict', 'ModuleList') and 'parametrizations' not in p.split('.')]
            for n in members:
                p = n.get('group', '')
                d = next((d for d in dictionaries if p.startswith(d + '.')), None)
                if d:
                    key = p[len(d) + 1:].split('.')[0]
                    modality_keys.setdefault(key, []).append(n)
            if len(modality_keys) > 1 and sum(map(len, modality_keys.values())) == len(members) and independent(list(modality_keys.values())):
                return [unit(f'{key} · 独立分支', group, native(group, '#branch')) for key, group in modality_keys.items()]
        # Contiguous invocations, not a union of every call to shared weights.
        runs = []
        for n in members:
            p = n.get('group', '')
            candidates = [q for q in modules if q and (p == q or p.startswith(q + '.')) and (prefix.startswith('#') or not prefix or q.startswith(prefix + '.'))]
            q = min(candidates, key=lambda x: len(x.split('.'))) if candidates else ''
            if runs and q and runs[-1][0] == q:
                runs[-1][1].append(n)
            else:
                runs.append((q, [n]))
        result = []
        for q, group in runs:
            if len(group) == 1:
                result.append(unit(group[0]['name'], group))
                continue
            m = modules.get(q, {})
            name = f"{TYPES.get(m.get('type'), m.get('type', '计算'))} · {q}"
            # Containers that contain only one used child need no extra click.
            children = native(group, q) if q else []
            if len(children) == 1 and m.get('type') in ('ModuleDict', 'ModuleList', 'Sequential'):
                result.extend(children)
            else:
                result.append(unit(name, group, children, m.get('code', {})))
        return result

    def title(stmt):
        calls = []
        for n in ast.walk(stmt):
            if isinstance(n, ast.Call):
                f = n.func
                while isinstance(f, ast.Subscript):
                    f = f.value
                key = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ''
                if key in NAMES:
                    calls.append(key)
        # Outer chunk/exp do not hide the module call that actually generated parameters.
        for key in calls:
            if key in NAMES:
                return NAMES[key]
        target = getattr(stmt, 'targets', [None])[0]
        if isinstance(target, ast.Name):
            return target.id
        value = getattr(stmt, 'value', None)
        if isinstance(value, ast.Call):
            return ast.unparse(value.func) + ' · 计算'
        return ast.unparse(stmt).split('\n')[0][:70]

    def lexical(members, body):
        runs = []
        for n in members:
            line = lines.get(n['id'], -1)
            stmt = next((s for s in body if s.lineno <= line <= s.end_lineno), None)
            key = stmt.lineno if stmt else -1
            if runs and runs[-1][0] == key:
                runs[-1][2].append(n)
            else:
                runs.append((key, stmt, [n]))
        result = []
        for _, stmt, group in runs:
            source = {'file': str(file), 'line': stmt.lineno if stmt else None}
            if isinstance(stmt, (ast.For, ast.While)):
                # The recorded source line returning to an earlier body statement
                # marks an actual loop iteration. Nested loops are kept inside it.
                parts, current, previous = [], [], -1
                for n in group:
                    line = lines.get(n['id'], -1)
                    index = next((i for i, s in enumerate(stmt.body) if s.lineno <= line <= s.end_lineno), previous)
                    if index < previous and current:
                        parts.append(current); current = []
                    current.append(n); previous = index
                if current:
                    parts.append(current)
                body_calls = {n.func.attr for n in ast.walk(stmt) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
                name = '循环计算'
                children = [unit(f'{name} · {i+1}/{len(parts)}', part, lexical(part, stmt.body), source) for i, part in enumerate(parts)]
                # A single-line loop has no line-number reset. Do not claim an
                # iteration count that this source map cannot establish.
                count = f'{len(parts)} 次' if len(parts)>1 else '已执行结构'
                ident=unit(f'{name} · {count}', group, children, source)
                units[-1]['loop']=True
                result.append(ident)
            elif isinstance(stmt, ast.If):
                result.extend(lexical(group, stmt.body + stmt.orelse))
            else:
                children = native(group)
                if len(children) == 1 and len(group) == 1:
                    result.extend(children)
                else:
                    result.append(unit(title(stmt) if stmt else '源码调用', group, children, source))
        return result

    roots = lexical(nodes, forward.body)
    from .composition_rules import enrich_composition
    roots = enrich_composition(graph, model, units, roots, tree, cls, producer, independent)
    index = {u['id']: u for u in units}
    reachable = set()
    def mark(ident):
        if ident in reachable:
            return
        reachable.add(ident)
        for child in index[ident]['children']:
            mark(child)
    for ident in roots:
        mark(ident)
    graph['composition'] = {'roots': roots, 'units': [u for u in units if u['id'] in reachable],
                            'hiddenDiagnostics': len(by_id) - len(live),
                            'evidence': 'Executed node order + Python source scopes + nn.Module paths; raw graph unchanged'}
