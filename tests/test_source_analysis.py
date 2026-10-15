import sys
import json
import unittest
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from silicondevine import export_model
from silicondevine.source_analysis import describe_callable
from test_composition import check


def helper(x):
    return torch.sin(x) * x


class GenericPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))
        self.last = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4), nn.SiLU())

    def prepare(self, x):
        return helper(x) + x * .5

    def forward(self, x):
        return self.last(self.first(self.prepare(x)))


class RenamedPipeline(GenericPipeline):
    def __init__(self):
        super().__init__()
        self.alpha = self.first; self.beta = self.last
        del self.first; del self.last

    def forward(self, x):
        return self.beta(self.alpha(self.prepare(x)))


class Logic(nn.Module):
    def forward(self, x):
        return torch.any(x, -1), torch.all(x, -1, keepdim=True), torch.ne(x, 0), torch.eye(3)


class SourceAnalysisTest(unittest.TestCase):
    def test_nested_return_and_rename_preserve_composition(self):
        signatures = []
        for cls in (GenericPipeline, RenamedPipeline):
            g = export_model(cls().eval(), (torch.ones(2,4),), backend='execution', include_values=True)
            check(g)
            index = {u['id']: u for u in g['composition']['units']}
            roots = [index[k] for k in g['composition']['roots']]
            self.assertEqual(len(roots), 3)  # preparation, first module, last module
            self.assertTrue(all('description' in u['visual'] for u in roots))
            signatures.append([len(u['nodes']) for u in roots])
        self.assertEqual(signatures[0], signatures[1])

    def test_source_follows_python_methods_and_helpers(self):
        p = describe_callable(GenericPipeline().forward)
        prepare = next(h for h in p['helpers'] if h['name'].endswith('prepare'))
        h = next(h for h in prepare['helpers'] if h['name'] == 'helper')
        self.assertIn('torch.sin', [c['target'] for c in h['calls']])
        self.assertTrue(Path(h['file']).is_file())

    def test_standard_logic_is_recognized(self):
        g = export_model(Logic(), (torch.tensor([[0.,1.,0.],[1.,1.,1.]]),), backend='execution', include_values=True)
        self.assertEqual({n['op'] for n in g['nodes']}, {'any','all','ne','constant'})
        by_id = {t['id']: t for t in g['tensors']}
        values = [by_id[t]['data']['values'] for t in g['outputs']]
        self.assertEqual(values[:3], [[True, True], [False, True], [False,True,False,True,True,True]])
        Path('.qa/logic.json').write_text(json.dumps(g),encoding='utf8')

if __name__ == '__main__': unittest.main()
