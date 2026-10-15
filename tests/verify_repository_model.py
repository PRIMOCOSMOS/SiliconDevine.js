"""Compare all captured outputs with a repository's unmodified PyTorch forward."""
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
import time
import torch
from torch.utils._pytree import tree_flatten

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'python'))
from silicondevine.discovery import scan_project, repository_defaults, _inferred_inputs
from silicondevine import export_model

parser = argparse.ArgumentParser()
parser.add_argument('directory')
args = parser.parse_args()
torch.set_num_threads(4)
start = time.monotonic()
report = scan_project([args.directory])
settings = repository_defaults(report)['settings']
for path in {r['root'] for r in report['files']}: sys.path.insert(0, path)
module, name = settings['entry'].split(':')
torch.manual_seed(0)
model = getattr(importlib.import_module(module), name)(**settings['init']).eval()
model._silicondevine_input_shape = settings['init']['original_shape']
inputs, kwargs, _ = _inferred_inputs(model, 8)
with torch.no_grad(), torch.random.fork_rng():
    reference, _ = tree_flatten(copy.deepcopy(model)(*copy.deepcopy(inputs), **copy.deepcopy(kwargs)))
shapes = {tuple(v.shape) for v in reference if isinstance(v, torch.Tensor)}
captured = {}
def sink(name, tensor):
    if tuple(tensor.shape) in shapes: captured[name] = tensor.detach().clone()
graph = export_model(model, inputs, kwargs=kwargs, backend='execution', tensor_sink=sink,
                     include_values=True, value_limit=128)
if graph.get('composition'):
    from test_composition import check
    check(graph)
    (root / '.qa/mmhvae-composition.json').write_text(json.dumps(graph), encoding='utf-8')
expected = [v for v in reference if isinstance(v, torch.Tensor)]
assert len(graph['outputs']) == len(expected)
for name, value in zip(graph['outputs'], expected):
    torch.testing.assert_close(captured[name], value, atol=2e-5, rtol=2e-5)
result = {'entry': settings['entry'], 'outputs': len(expected), 'scalarsCompared': sum(v.numel() for v in expected),
          'nodes': len(graph['nodes']), 'parameterCount': sum(p.numel() for p in model.parameters()),
          'elapsedSeconds': round(time.monotonic() - start, 2)}
(root / '.qa/mmhvae-numerical.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
