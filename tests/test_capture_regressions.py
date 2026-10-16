import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from silicondevine.watch import run_worker
from silicondevine.discovery import scan_project, repository_defaults, capture_project
from silicondevine import export_model
import torch
from torch import nn


class CaptureRegressionTests(unittest.TestCase):
    def test_utf8_and_invalid_bytes_in_worker_error(self):
        code = "import sys; sys.stderr.buffer.write('模型入口错误 → 缺少参数'.encode('utf-8') + bytes([255])); sys.exit(7)"
        with self.assertRaisesRegex(RuntimeError, '模型入口错误 → 缺少参数'):
            run_worker([sys.executable, '-c', code], 10)
        with self.assertRaisesRegex(RuntimeError, '退出码 9'):
            run_worker([sys.executable, '-c', 'import sys;sys.exit(9)'], 10)

    def test_nested_mutable_inputs_and_seeded_distribution(self):
        class Model(nn.Module):
            def forward(self, x, *, offset):
                x['image'] = x['image'] * 2
                return torch.distributions.Normal(x['image'] + offset, torch.ones_like(offset)).rsample()
        original = {'image': torch.ones(1, 4)}
        torch.manual_seed(72)
        expected = Model()({'image': original['image'].clone()}, offset=torch.zeros(1, 4))
        torch.manual_seed(72)
        graph = export_model(Model(), (original,), kwargs={'offset': torch.zeros(1, 4)}, backend='execution', include_values=True)
        out = next(t for t in graph['tensors'] if t['id'] == graph['outputs'][0])
        torch.testing.assert_close(torch.tensor(out['data']['values']).reshape(out['shape']), expected)
        self.assertEqual(len(graph['inputs']), 2)
        self.assertTrue(torch.equal(original['image'], torch.ones(1, 4)))

    def test_namespace_repository_defaults_without_running_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            network = root / 'netspace'
            network.mkdir()
            (network / 'blocks.py').write_text('from torch import nn\nclass Block(nn.Module):\n def __init__(self,width):\n  super().__init__();self.fc=nn.Linear(width,2)\n def forward(self,x):return self.fc(x)\n')
            (network / 'model.py').write_text('from torch import nn\nfrom netspace.blocks import Block\nclass Main(nn.Module):\n def __init__(self,width):\n  super().__init__();self.block=Block(width)\n def forward(self,x):return self.block(x)\nclass Alternative(nn.Module):pass\n')
            (root / 'train.py').write_text('import argparse\nfrom netspace.model import Main,Alternative\nparser=argparse.ArgumentParser()\nparser.add_argument("--width",default=3)\nparser.add_argument("--model",default="main")\nopt=parser.parse_args()\nraise RuntimeError("training must never run")\nif opt.model=="main":\n model=Main(width=opt.width)\nelse:\n model=Alternative()\n')
            report = scan_project([network])
            self.assertEqual({c['id'] for c in report['candidates']}, {'netspace.model:Main', 'netspace.model:Alternative'})
            self.assertEqual(repository_defaults(report)['settings'], {'entry': 'netspace.model:Main', 'init': {'width': 3}})
            graph = capture_project([network], root / 'capture')
            self.assertEqual(graph['discovery']['entry'], 'netspace.model:Main')
            self.assertEqual(graph['discovery']['repositoryDefaults']['source'], str(root / 'train.py'))

    def test_validation_does_not_mutate_dictionary_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'mutating_model.py').write_text('from torch import nn\nclass Model(nn.Module):\n def __init__(self):\n  super().__init__();self.fc=nn.Linear(3,2)\n def forward(self,x):\n  x["image"]=self.fc(x["image"])\n  return x["image"]\n')
            (root / 'silicondevine.json').write_text(json.dumps({'inputs': [{'image': {'shape': [1, 3]}}], 'backend': 'execution'}))
            graph = capture_project([root], root / 'capture')
            self.assertEqual(next(t['shape'] for t in graph['tensors'] if t['id'] == graph['inputs'][0]), [1, 3])
            self.assertEqual(next(t['shape'] for t in graph['tensors'] if t['id'] == graph['outputs'][0]), [1, 2])


if __name__ == '__main__': unittest.main()
