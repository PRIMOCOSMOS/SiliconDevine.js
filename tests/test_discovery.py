"""Real graph assertions for repository imports, not mocked capture results."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from silicondevine.discovery import scan_project, capture_project


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.package = 'project_' + uuid.uuid4().hex
        self.source = self.root / self.package
        self.source.mkdir()
        (self.source / '__init__.py').write_text('')

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, code):
        file = self.source / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(code, encoding='utf-8')
        return file

    def capture(self, entry=None):
        return capture_project([self.source], self.root / 'capture', entry=entry)

    def test_multifile_alias_inheritance_and_real_values(self):
        self.write('base.py', 'from torch.nn import Module as BaseModule\nclass Base(BaseModule): pass\n')
        self.write('block.py', 'from .base import Base\nfrom torch import nn\nclass Block(Base):\n def __init__(self):\n  super().__init__(); self.fc=nn.Linear(4,3)\n def forward(self,x): return self.fc(x).relu()\n')
        self.write('__init__.py', 'from .block import Block\n')
        main = self.write('network.py', 'from .base import Base\nfrom . import Block as Unit\nclass Network(Base):\n def __init__(self):\n  super().__init__(); self.unit=Unit()\n def forward(self,x): return self.unit(x)\n')
        self.write('train.py', 'raise RuntimeError("unrelated training script must never execute")\n')
        report = scan_project([main])
        self.assertEqual([c['name'] for c in report['candidates']], ['Network'])
        self.assertGreaterEqual(len(report['files']), 4)
        graph = self.capture()
        self.assertTrue(graph['discovery']['inputsInferred'])
        self.assertTrue(graph['discovery']['entry'].endswith('network:Network'))
        out = next(t for t in graph['tensors'] if t['id'] == graph['outputs'][0])
        self.assertEqual(out['shape'], [1, 3])
        self.assertEqual(len(out['data']['values']), 3)
        self.assertTrue(all(x >= 0 for x in out['data']['values']))

    def test_ambiguous_entry_is_not_arbitrarily_selected(self):
        self.write('models.py', 'from torch import nn\nclass A(nn.Module): pass\nclass B(nn.Module): pass\n')
        with self.assertRaisesRegex(ValueError, '多个独立入口'):
            self.capture()

    def test_arbitrary_factory_name_without_fixed_interface(self):
        self.write('net.py', 'from torch import nn\nclass Network(nn.Module):\n def __init__(self,width):\n  super().__init__(); self.fc=nn.Linear(width,2)\n def forward(self,x): return self.fc(x)\ndef small_network():\n model=Network(3)\n return model\n')
        graph = self.capture()
        self.assertTrue(graph['discovery']['entry'].endswith('net:small_network'))
        self.assertEqual(graph['discovery']['parameterCount'], 8)

    def test_config_constructor_and_keyword_input(self):
        self.write('network.py', 'from torch import nn\nclass Network(nn.Module):\n def __init__(self,width):\n  super().__init__(); self.fc=nn.Linear(width,2)\n def forward(self,x,*,other): return self.fc(x)+other\n')
        with self.assertRaisesRegex(ValueError, '需要填写.*width'):
            self.capture()
        (self.source / 'silicondevine.json').write_text(json.dumps({'init': {'width': 3}, 'inputs': [{'shape': [2, 3], 'fill': 'ones'}], 'kwargs': {'other': {'shape': [2, 2], 'fill': 'zeros'}}}))
        graph = self.capture()
        self.assertFalse(graph['discovery']['inputsInferred'])
        self.assertEqual(next(t['shape'] for t in graph['tensors'] if t['id'] == graph['outputs'][0]), [2, 2])

    def test_convolution_input_is_validated_with_flatten(self):
        self.write('cnn.py', 'from torch import nn\nclass CNN(nn.Module):\n def __init__(self):\n  super().__init__(); self.conv=nn.Conv2d(3,2,3,padding=1); self.fc=nn.Linear(2*16*16,5)\n def forward(self,x): return self.fc(self.conv(x).flatten(1))\n')
        graph = self.capture()
        self.assertEqual(graph['discovery']['exampleInputs'][0]['input']['shape'], [1, 3, 16, 16])

    def test_embedding_uses_integer_tokens(self):
        self.write('lm.py', 'from torch import nn\nclass LM(nn.Module):\n def __init__(self):\n  super().__init__(); self.emb=nn.Embedding(7,4); self.fc=nn.Linear(4,7)\n def forward(self,tokens): return self.fc(self.emb(tokens))\n')
        graph = self.capture()
        self.assertEqual(graph['discovery']['exampleInputs'][0]['input']['dtype'], 'int64')
        self.assertEqual(next(t['shape'] for t in graph['tensors'] if t['id'] == graph['outputs'][0]), [1, 4, 7])

    def test_parameter_budget_rejects_before_weight_allocation(self):
        self.write('large.py', 'from torch import nn\nclass Large(nn.Module):\n def __init__(self):\n  super().__init__(); self.fc=nn.Linear(10000,10000)\n def forward(self,x): return self.fc(x)\n')
        with self.assertRaisesRegex(ValueError, '当前预算'):
            self.capture()

    def test_missing_dependency_scan_does_not_import(self):
        self.write('broken.py', 'import silicondevine_intentionally_missing_dependency\nfrom torch import nn\nclass Model(nn.Module): pass\n')
        self.assertEqual(len(scan_project([self.source])['candidates']), 1)
        with self.assertRaisesRegex(ValueError, '依赖未安装'):
            self.capture()

    def test_upstream_sab_direct_class_without_adapter(self):
        import torch
        source = Path(__file__).resolve().parents[1] / 'examples/upstream/set_transformer/modules.py'
        config = self.root / 'sab.json'
        config.write_text(json.dumps({'entry': 'modules:SAB', 'init': {'dim_in': 8, 'dim_out': 8, 'num_heads': 2, 'ln': True},
                                     'inputs': [{'shape': [1, 3, 8], 'fill': 'ones'}]}))
        graph = capture_project([source], self.root / 'upstream', config=config)
        from silicondevine.project_imports import project_imports
        with project_imports(scan_project([source])):
            from modules import SAB
            torch.manual_seed(0)
            reference = SAB(8, 8, 2, ln=True).eval()(torch.ones(1, 3, 8)).detach()
        output = next(t for t in graph['tensors'] if t['id'] == graph['outputs'][0])
        torch.testing.assert_close(torch.tensor(output['data']['values']).reshape(output['shape']), reference)
        self.assertTrue(any(n['op'] == 'softmax' for n in graph['nodes']))

    def test_data_dependent_path_uses_real_execution_fallback(self):
        self.write('branch.py', 'from torch import nn\nclass Branch(nn.Module):\n def __init__(self):\n  super().__init__(); self.fc=nn.Linear(3,2)\n def forward(self,x):\n  if x.sum().item()>0: return self.fc(x)\n  return -self.fc(x)\n')
        (self.source / 'silicondevine.json').write_text(json.dumps({'inputs': [{'shape': [1, 3], 'fill': 'ones'}]}))
        graph = self.capture()
        self.assertEqual(graph['discovery']['backend'], 'execution')
        self.assertTrue(graph['discovery']['fallbackErrors'])


if __name__ == '__main__':
    unittest.main()
