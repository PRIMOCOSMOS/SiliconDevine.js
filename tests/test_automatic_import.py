"""Automatic parsing is the primary path. Assistance is the last resort."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from silicondevine.discovery import capture_project,scan_project
from silicondevine.assistance import ImportProblem


class AutomaticTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.package=self.root/('automatic_'+uuid.uuid4().hex);self.package.mkdir();(self.package/'__init__.py').write_text('')
    def tearDown(self):self.temp.cleanup()
    def write(self,name,code):
        p=self.package/name;p.write_text(code,encoding='utf-8');return p
    def capture(self,**kwargs):return capture_project([self.package],self.root/'capture',backend='fx',**kwargs)

    def test_multifile_source_selects_entry_constructor_and_nested_inputs(self):
        self.write('models.py','''from torch import nn
class Chosen(nn.Module):
 def __init__(self,width,outputs):
  super().__init__();self.fc=nn.Linear(width,outputs)
 def forward(self,samples,*,valid):return self.fc(samples['signal'])*valid.unsqueeze(-1)
class Independent(nn.Module):
 def forward(self,x):return x.cos()
''')
        self.write('experiment.py','''import torch as th
from .models import Chosen as Model
WIDTH=3*2
model=Model(WIDTH,outputs=2).eval()
samples={'signal':th.randn(2,4,WIDTH)}
valid=th.ones(2,4,dtype=th.bool)
output=model(samples,valid=valid)
raise RuntimeError('This experiment must not execute during discovery')
''')
        graph=self.capture()
        d=graph['discovery'];self.assertTrue(d['entry'].endswith('models:Chosen'))
        self.assertEqual(d['parameterCount'],14)
        self.assertEqual(d['inputStatus'],'validated-example')
        self.assertEqual(d['exampleInputs']['inputs'][0]['signal']['shape'],[2,4,6])
        self.assertEqual(d['exampleInputs']['kwargs']['valid']['dtype'],'bool')
        self.assertTrue(d['automaticResolution']['inputAttempts'][0]['validated'])
        self.assertEqual(next(t for t in graph['tensors'] if t['id']==graph['outputs'][0])['shape'],[2,4,2])

    def test_typed_configuration_defaults_need_no_user_input(self):
        self.write('net.py','''from __future__ import annotations
from dataclasses import dataclass
from torch import nn
@dataclass
class Config:
 width:int=5
class Network(nn.Module):
 def __init__(self,config:Config):
  super().__init__();self.fc=nn.Linear(config.width,2)
 def forward(self,x):return self.fc(x)
''')
        graph=self.capture();self.assertEqual(graph['discovery']['parameterCount'],12)
        self.assertEqual(graph['discovery']['automaticResolution']['constructor'][0]['basis'],'annotated configuration class defaults')
        (self.package/'silicondevine.json').write_text('{outdated invalid data')
        graph=self.capture();self.assertEqual(graph['discovery']['inputStatus'],'validated-example')
        self.assertTrue(any('继续自动解析' in w for w in graph['discovery']['warnings']))

    def test_crossfile_configuration_instance_is_recovered(self):
        self.write('config.py','from dataclasses import dataclass\n@dataclass\nclass Config:\n width:int=3\n')
        self.write('network.py','from torch import nn\nclass Network(nn.Module):\n def __init__(self,config):\n  super().__init__();self.fc=nn.Linear(config.width,2)\n def forward(self,x):return self.fc(x)\n')
        self.write('usage.py',"from .config import Config\nfrom .network import Network\nconfig=Config(7)\nmodel=Network(config)\n")
        graph=self.capture();self.assertEqual(graph['discovery']['parameterCount'],16)

    def test_conflicting_examples_request_only_the_unresolved_parameter(self):
        self.write('net.py','''from torch import nn
class Network(nn.Module):
 def __init__(self,width,outputs):
  super().__init__();self.fc=nn.Linear(width,outputs)
 def forward(self,x):return self.fc(x)
def first():
 model=Network(3,2)
def second():
 model=Network(7,2)
''')
        with self.assertRaises(ImportProblem) as error:self.capture()
        issue=error.exception.issue
        self.assertEqual(issue['code'],'constructor_required');self.assertEqual(issue['fields'],['width'])
        self.assertEqual(issue['schema']['settings']['init'],{'outputs':2})

    def test_unresolved_dimensions_keep_structure_without_a_fault(self):
        self.write('net.py','from torch import nn\nclass Network(nn.Module):\n def forward(self,signal):return signal.sin()+1\n')
        graph=self.capture()
        self.assertEqual(graph['discovery']['inputStatus'],'unresolved')
        self.assertEqual(graph['discovery']['backend'],'symbolic')

    def test_ambiguous_independent_entries_still_require_confirmation(self):
        self.write('net.py','from torch import nn\nclass First(nn.Module):\n def forward(self,x):return x.sin()\nclass Second(nn.Module):\n def forward(self,x):return x.cos()\n')
        with self.assertRaises(ImportProblem) as error:self.capture()
        self.assertEqual(error.exception.issue['code'],'entry_required')

    def test_partial_settings_do_not_disable_source_parsing(self):
        self.write('net.py','from torch import nn\nclass Network(nn.Module):\n def __init__(self,width):\n  super().__init__();self.fc=nn.Linear(width,2)\n def forward(self,x):return self.fc(x)\n')
        self.write('launch.py','from .net import Network\nmodel=Network(width=6)\n')
        (self.package/'silicondevine.json').write_text(json.dumps({'seed':9}))
        graph=self.capture(overrides={'input_mode':'auto'})
        self.assertEqual(graph['discovery']['parameterCount'],14)
        self.assertTrue(graph['discovery']['inputStatus']=='validated-example')

    def test_argparse_defaults_in_arbitrarily_named_script(self):
        self.write('net.py','from torch import nn\nclass Network(nn.Module):\n def __init__(self,width):\n  super().__init__();self.fc=nn.Linear(width,2)\n def forward(self,x):return self.fc(x)\nclass Other(nn.Module):\n def forward(self,x):return x.sin()\n')
        self.write('experiment_setup.py',"import argparse\nfrom .net import Network\nparser=argparse.ArgumentParser()\nparser.add_argument('--width',type=int,default=7)\nsettings_for_experiment=parser.parse_args()\nmodel=Network(settings_for_experiment.width)\nraise RuntimeError('Do not execute this script')\n")
        graph=self.capture();self.assertTrue(graph['discovery']['entry'].endswith('net:Network'))
        self.assertEqual(graph['discovery']['parameterCount'],16)

    def test_constructor_tensor_preserves_tensor_identity(self):
        self.write('net.py','import torch\nfrom torch import nn\nclass Network(nn.Module):\n def __init__(self,weight):\n  super().__init__();self.weight=nn.Parameter(weight)\n def forward(self,x):return x @ self.weight\n')
        self.write('usage.py','import torch\nfrom .net import Network\nmodel=Network(torch.ones(3,4))\noutput=model(torch.ones(2,3))\n')
        graph=self.capture();self.assertEqual(graph['discovery']['inputStatus'],'validated-example')
        self.assertEqual(graph['discovery']['parameterCount'],12)


if __name__=='__main__':unittest.main()
