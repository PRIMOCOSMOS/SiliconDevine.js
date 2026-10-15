"""Read literal shape dictionaries without importing or running data pipelines."""
import ast
import inspect
from pathlib import Path


def source_input_specs(report, model, size):
    names = {p.name for p in inspect.signature(model.forward).parameters.values() if p.default is p.empty
             and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    candidates = {name: [] for name in names}
    for file in report['files']:
        try: tree = ast.parse(Path(file['path']).read_text(encoding='utf-8-sig'))
        except (OSError, SyntaxError, UnicodeError): continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict): continue
            for key, value in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant) or key.value not in names: continue
                if not isinstance(value, (ast.Tuple, ast.List)) or not 1 <= len(value.elts) <= 8: continue
                dims = []
                for d in value.elts:
                    if isinstance(d, ast.Constant) and type(d.value) is int and 0 < d.value <= 8192: dims.append(d.value)
                    elif isinstance(d, ast.Name): dims.append(d.id)
                    else: break
                else:
                    candidates[key.value].append((dims, file['path'], value.lineno))
    specs = {}
    for name, choices in candidates.items():
        if not choices or len({tuple(c[0]) for c in choices}) != 1: continue
        dims, file, line = choices[0]
        shape = [d if isinstance(d, int) else 1 if d.lower() in ('batch', 'batch_size', 'b') else min(size, 8) for d in dims]
        specs[name] = {'shape': shape, 'symbolicShape': dims, 'source': {'file': file, 'line': line},
                       'derivedFrom': 'source shape dictionary; symbolic dimensions use validated small example sizes'}
    return specs
