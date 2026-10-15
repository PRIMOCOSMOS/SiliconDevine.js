"""Model-independent presentation rules, backed by executed dependencies.

Class names and model paths are captions only. Rules never replace computations.
"""
import ast
from collections import defaultdict


def enrich_composition(graph, model, units, roots, tree, cls, producer, independent):
    index = {u['id']: u for u in units}
    nodes = {n['id']: n for n in graph['nodes']}
    tensors = {t['id']: t for t in graph['tensors']}
    containers = [m['path'] for m in graph.get('modules', [])
                  if m['type'] in ('ModuleDict', 'ModuleList')
                  and 'parametrizations' not in m['path'].split('.')]
    dictionaries = {m['path'] for m in graph.get('modules', []) if m['type'] == 'ModuleDict'}
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    # Resolve imported Normal identifiers. A user method called Normal is not proof.
    normal_aliases = {a.asname or a.name for n in tree.body if isinstance(n, ast.ImportFrom)
                      and n.module in ('torch.distributions', 'torch.distributions.normal')
                      for a in n.names if a.name == 'Normal'}

    def normal_call(call):
        f = call.func
        if isinstance(f, ast.Name): return f.id in normal_aliases
        return isinstance(f, ast.Attribute) and ast.unparse(f) in (
            'torch.distributions.Normal', 'torch.distributions.normal.Normal')

    def source_statement(u):
        source = u.get('source', {})
        line = source.get('line') if isinstance(source, dict) else None
        return next((n for n in ast.walk(methods.get('forward', cls))
                     if isinstance(n, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Return))
                     and n.lineno == line), None)

    def gaussian_statement(stmt):
        if stmt is None: return False
        calls = [n for n in ast.walk(stmt) if isinstance(n, ast.Call)]
        for call in calls:
            if normal_call(call): return True
            f = call.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == 'self':
                method = methods.get(f.attr)
                if method and sum(normal_call(n) for n in ast.walk(method) if isinstance(n, ast.Call)) == 1:
                    return True
        return False

    def trace_branch(u, allowed):
        front = [nodes[k] for k in u['nodes']]; seen = set()
        while front:
            found = set(); nxt = []
            for n in front:
                if n['id'] in seen: continue
                seen.add(n['id']); path = n.get('group', '')
                matches = [p for p in allowed if path.startswith(p + '.')]
                if matches:
                    p = min(matches, key=lambda s: len(s.split('.')))
                    found.add(path[len(p) + 1:].split('.')[0])
                nxt.extend(producer[t] for t in n['inputs'] if t in producer)
            if len(found) == 1: return next(iter(found))
            if len(found) > 1: return None
            front = nxt
        return None

    def branch_key(u):
        # A shared layer in a ModuleList is an invocation index, not an input
        # branch identity. Prefer upstream keyed branches, then positional ones.
        return trace_branch(u, [p for p in containers if p in dictionaries]) or trace_branch(u, containers)

    def stage_units(u):
        repeated = defaultdict(list)
        def walk(c):
            if len(c['nodes']) > 1 and any(nodes[k].get('parameters') for k in c['nodes']):
                repeated[tuple(nodes[k]['op'] for k in c['nodes'])].append(c)
            for key in c['children']: walk(index[key])
        for key in u['children']: walk(index[key])
        candidates = [cs for cs in repeated.values() if len(cs) > 1 and
                      len({k for c in cs for k in c['nodes']}) == sum(len(c['nodes']) for c in cs)]
        if candidates:
            # Prefer the largest repeated real block over its smaller constituent
            # Conv/Norm calls. No class-name dictionary is needed.
            return max(candidates, key=lambda cs: sum(len(c['nodes']) for c in cs))
        children = [index[k] for k in u['children']]
        # Containers with one child reveal the next real composition level.
        while len(children) == 1 and children[0]['children']:
            children = [index[k] for k in children[0]['children']]
        return [c for c in children if len(c['nodes']) > 1 and any(
            nodes[k].get('parameters') for k in c['nodes'])]

    def decorate_branch(u, family, fallback, key=None):
        key = key or branch_key(u) or fallback
        stages = []
        for c in stage_units(u):
            members = set(c['nodes'])
            # Last live activation of this invocation; no synthetic placeholder.
            outs = [t for k in c['nodes'] for t in nodes[k]['outputs']
                    if tensors.get(t, {}).get('role') == 'activation']
            if outs:
                stages.append({'tensor': outs[-1], 'label': c['name'], 'scope': c['id']})
        visual = dict(u.get('visual', {}))
        visual.update(branch=key, family=family, evidence='independent executed activation paths',
                      summary=f'{len(stages)} 个计算阶段 · 独立数据通路' if len(stages) > 1 else '独立数据通路 · 展开查看原子组成')
        visual.setdefault('kind', 'branch_tower' if len(stages) > 1 else 'parallel_branch')
        if len(stages) > 1: visual['stages'] = stages
        u['visual'] = visual
        if u['name'].endswith('独立分支') or u['name'].startswith('循环计算'):
            u['name'] = f'{key} · ' + ('多级计算塔' if len(stages) > 1 else '独立计算分支')

    # Gaussian metadata requires both source construction evidence and an actual
    # pair of broadcast tensors. Broadcasting alone does not imply probability.
    for u in units:
        stmt = source_statement(u)
        if gaussian_statement(stmt):
            broadcasts = [nodes[k] for k in u['nodes'] if nodes[k]['op'] == 'broadcast_tensors'
                          and len(nodes[k]['outputs']) == 2]
            if len(broadcasts) == 1:
                mean, std = broadcasts[0]['outputs']
                u['visual'] = {'kind': 'gaussian_parameters', 'distribution': {'mean': mean, 'std': std},
                    'summary': '均值 μ · 方差 σ²',
                    'description': '源码构建的正态分布：均值表示中心，方差表示不确定性。PyTorch Normal 的第二个参数是标准差；这里用其平方显示方差。曲线显示坐标 0，横轴按 μ±3σ 归一显示。展开可查看参数生成的实际计算。',
                    'formula': r'\mathcal N(\mu,\sigma^2),\quad\mathrm{Var}=\sigma^2',
                    'evidence': 'source Normal construction + executed broadcast parameter pair'}

    # Every candidate family must pass a transitive independence check, including
    # ModuleLists and loop invocations with shared weights. Sequential lists fail.
    for u in units:
        children = [index[k] for k in u['children']]
        groups = [[nodes[k] for k in c['nodes']] for c in children]
        if len(children) < 2 or not all(len(g) > 1 for g in groups) or not independent(groups): continue
        # Do not turn an arbitrary bag of disjoint scalar bookkeeping into towers.
        if not all(any(n.get('parameters') for n in g) or any(index[k].get('visual', {}).get('distribution') for k in c['children'])
                   for c, g in zip(children, groups)): continue
        keys = [branch_key(c) for c in children]
        for i, c in enumerate(children):
            key = keys[i] if keys[i] and keys.count(keys[i]) == 1 else str(i + 1)
            # A probability-producing branch preserves its distribution identity.
            ds = [index[k]['visual'] for k in c['children'] if index[k].get('visual', {}).get('distribution')]
            if len(ds) == 1 and 'visual' not in c:
                c['visual'] = dict(ds[0]); c['name'] = f'{key} · 高斯概率头'
            decorate_branch(c, u['id'], str(i + 1), key)
        u['parallel'] = True

    # Promote proven parallel families at every level. Short sequential loops
    # expose their iterations, while large repetition stays expandable and bounded.
    for u in units:
        u['children'] = [j for k in u['children'] for j in
                         (index[k]['children'] if index[k].get('parallel') else [k])]
    def expose(key):
        u = index[key]
        branch_children = u['children'] and all(index[k].get('visual', {}).get('family') for k in u['children'])
        learned_loop = u.get('loop') and any(nodes[k].get('parameters') for k in u['nodes'])
        expand = u.get('parallel') or branch_children or (learned_loop and 1 < len(u['children']) <= 8)
        if not u.get('visual') and len(u['children']) == 1: return expose(u['children'][0])
        return [j for k in u['children'] for j in expose(k)] if expand else [key]
    promoted = [j for key in roots for j in expose(key)]
    # A nested return can span the entire model. Present its native module
    # invocations instead of forcing an extra click into one giant lexical box.
    # Consecutive loose tensor operations remain a source-backed, expandable unit.
    def open_return(key):
        u = index[key]
        stmt = source_statement(u)
        children = [index[k] for k in u['children']]
        learned = [c for c in children if len(c['nodes']) > 1 and any(
            nodes[n].get('parameters') for n in c['nodes'])]
        if not isinstance(stmt, ast.Return) or u.get('visual') or len(learned) < 2:
            return [key]
        result, pending = [], []
        def flush():
            if not pending: return
            if len(pending) == 1:
                result.append(pending[0]['id'])
            else:
                ident = '@source.' + str(len(units))
                item = {'id': ident, 'name': '张量准备与变换',
                        'nodes': [n for c in pending for n in c['nodes']],
                        'children': [c['id'] for c in pending], 'source': u['source']}
                units.append(item); index[ident] = item; result.append(ident)
            pending.clear()
        for c in children:
            if c in learned:
                flush(); result.append(c['id'])
            else: pending.append(c)
        flush()
        return result
    promoted = [j for key in promoted for j in open_return(key)]
    # Attribute-based parallel branches (self.left / self.right) have no container.
    # Equal external activation boundaries establish their common entry point.
    by_entry = {}
    for key in promoted:
        u = index[key]
        if u.get('visual', {}).get('family') or len(u['nodes']) < 2: continue
        members = [nodes[k] for k in u['nodes']]
        if not any(n.get('parameters') for n in members): continue
        made = {t for n in members for t in n['outputs']}
        entry = tuple(sorted({t for n in members for t in n['inputs'] if t not in made
                              and tensors.get(t, {}).get('role') not in ('parameter', 'constant', 'buffer')}))
        if entry: by_entry.setdefault(entry, []).append(u)
    for siblings in by_entry.values():
        if len(siblings) < 2 or not independent([[nodes[k] for k in u['nodes']] for u in siblings]): continue
        family = 'entry:' + siblings[0]['id']
        for i, u in enumerate(siblings): decorate_branch(u, family, str(i + 1))
    return promoted
