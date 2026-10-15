"""Conservative, read-only interpretation of repository argparse configuration."""
import ast
import hashlib


UNKNOWN = object()


def infer_repository_config(report):
    from pathlib import Path
    roots = {Path(r['root']) for r in report['files']}
    candidates = {c['symbol']: c for c in report['candidates']}
    records={Path(r['path']).resolve():r for r in report['files']}
    signatures={}
    def positional_names(candidate):
        if candidate['id'] in signatures:return signatures[candidate['id']]
        try:
            code=ast.parse(Path(candidate['file']).read_text(encoding='utf-8-sig'))
            definition=next(n for n in code.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==candidate['name'])
            fn=next((n for n in definition.body if isinstance(n,ast.FunctionDef) and n.name=='__init__'),None) if isinstance(definition,ast.ClassDef) else definition
            names=[n.arg for n in [*fn.args.posonlyargs,*fn.args.args] if n.arg!='self'] if fn else []
        except (OSError,SyntaxError,StopIteration):names=[]
        signatures[candidate['id']]=names;return names
    proposals = []
    for root in sorted(roots):
        filenames={str(p.relative_to(root)) for p in records if p.is_relative_to(root)}
        filenames.update(p.name for p in root.glob('*.py'))
        for filename in sorted(filenames)[:512]:
            path = root / filename
            if not path.is_file(): continue
            try:
                tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            except (SyntaxError, UnicodeError): continue
            defaults, aliases = {}, dict(records.get(path.resolve(),{}).get('aliases',{}))
            aliases.update({c['name']:c['symbol'] for c in candidates.values() if Path(c['file']).resolve()==path.resolve()})
            parsed_names=set()
            for node in ast.walk(tree):
                if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr in ('parse_args','parse_known_args'):
                    for target in node.targets:
                        if node.value.func.attr=='parse_args' and isinstance(target,ast.Name):parsed_names.add(target.id)
                        elif node.value.func.attr=='parse_known_args' and isinstance(target,(ast.Tuple,ast.List)) and target.elts and isinstance(target.elts[0],ast.Name):parsed_names.add(target.elts[0].id)
                if isinstance(node, ast.ImportFrom) and not node.level and node.module:
                    for item in node.names: aliases[item.asname or item.name] = node.module + '.' + item.name
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != 'add_argument': continue
                try:
                    option = next(ast.literal_eval(a)[2:].replace('-', '_') for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith('--'))
                    values = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg in ('default', 'action', 'dest')}
                    if 'default' in values: value = values['default']
                    elif values.get('action') in ('store_false', 'store_true'): value = values['action'] == 'store_false'
                    else: continue
                    defaults[values.get('dest', option)] = value
                except (ValueError, StopIteration, TypeError): continue
            if not defaults:continue
            def evaluate(node):
                if isinstance(node,ast.Name) and node.id in parsed_names:
                    return {'$namespace':dict(defaults)}
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in parsed_names:
                    return defaults.get(node.attr, UNKNOWN)
                if isinstance(node, ast.Subscript):
                    value = evaluate(node.value)
                    if value is UNKNOWN: return UNKNOWN
                    try:
                        if isinstance(node.slice, ast.Slice):
                            def bound(v): return ast.literal_eval(v) if v else None
                            return value[slice(bound(node.slice.lower), bound(node.slice.upper), bound(node.slice.step))]
                        return value[ast.literal_eval(node.slice)]
                    except (ValueError, TypeError, IndexError, KeyError): return UNKNOWN
                if isinstance(node, ast.Compare) and len(node.ops) == 1:
                    left, right = evaluate(node.left), evaluate(node.comparators[0])
                    if left is UNKNOWN or right is UNKNOWN: return UNKNOWN
                    if isinstance(node.ops[0], ast.Eq): return left == right
                    if isinstance(node.ops[0], ast.NotEq): return left != right
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                    value = evaluate(node.operand)
                    return UNKNOWN if value is UNKNOWN else not value
                try: return ast.literal_eval(node)
                except (ValueError, TypeError): return UNKNOWN
            calls = []
            def visit(node):
                if isinstance(node, ast.If):
                    condition = evaluate(node.test)
                    if condition is not UNKNOWN:
                        for child in node.body if condition else node.orelse: visit(child)
                        return
                if isinstance(node, ast.Call) and isinstance(node.func,(ast.Name,ast.Attribute)):
                    name=ast.unparse(node.func);head,sep,tail=name.partition('.')
                    candidate = candidates.get(aliases.get(head,head)+(sep+tail if sep else ''))
                    if candidate:
                        init = {name:evaluate(value) for name,value in zip(positional_names(candidate),node.args)}
                        init.update({k.arg: evaluate(k.value) for k in node.keywords if k.arg})
                        known = {k: v for k, v in init.items() if v is not UNKNOWN}
                        # Unknown optional runtime services (logger/device) retain defaults.
                        calls.append({'settings': {'entry': candidate['id'], 'init': known},
                                      'source': str(path), 'line': node.lineno,
                                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                      'omittedRuntimeArguments': [k for k, v in init.items() if v is UNKNOWN]})
                for child in ast.iter_child_nodes(node): visit(child)
            visit(tree)
            unique = {repr(c['settings']): c for c in calls}
            if len(unique) == 1: proposals.extend(unique.values())
    unique = {repr(p['settings']): p for p in proposals}
    return next(iter(unique.values())) if len(unique) == 1 else None
