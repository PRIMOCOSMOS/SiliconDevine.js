"""Regression coverage for constructor recovery and repository isolation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from silicondevine.assistance import ImportProblem, parameters
from silicondevine.discovery import capture_project


class ConstructorContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, code, folder='model'):
        directory = self.root / folder
        directory.mkdir(exist_ok=True)
        source = directory / 'net.py'
        source.write_text(code, encoding='utf-8')
        return source

    def capture(self, source, **options):
        return capture_project([source], self.root / 'capture', backend='fx', **options)

    def test_unannotated_channels_reject_tensor_dictionary_with_integer_editor(self):
        source = self.source('''from torch import nn
class Dictionary_Block(nn.Module):
 def __init__(self,input):
  super().__init__();self.conv=nn.Conv2d(input,input,1)
 def forward(self,x):return self.conv(x)
''')
        for value in ({'shape': [8], 'dtype': 'float64'}, True, -2):
            with self.subTest(value=value), self.assertRaises(ImportProblem) as error:
                self.capture(source, overrides={'init': {'input': value}})
            issue = error.exception.issue
            self.assertEqual(issue['code'], 'constructor_invalid')
            self.assertEqual(issue['fields'], ['input'])
            field = issue['schema']['constructor'][0]
            self.assertEqual(field['kind'], 'integer')
            self.assertIn('Conv2d', field['derivedFrom'])

    def test_child_constructor_literals_are_discovered_without_running_parent(self):
        source = self.source('''from torch import nn
class Dictionary_Block(nn.Module):
 def __init__(self,input):
  super().__init__();self.conv=nn.Conv2d(input,input,1)
 def forward(self,x):return self.conv(x)
class Parent(nn.Module):
 def __init__(self):
  super().__init__();input=7;self.child=Dictionary_Block(input)
  raise RuntimeError('Never construct the parent for child discovery')
''')
        graph = self.capture(source, entry='net:Dictionary_Block')
        self.assertEqual(graph['discovery']['parameterCount'], 56)
        self.assertEqual(graph['discovery']['importOptions']['settings']['init'], {'input': 7})

    def test_positional_only_constructor_and_default_gap(self):
        source = self.source('''from torch import nn
class Model(nn.Module):
 def __init__(self,width=3,outputs=2,/,*,bias=True):
  super().__init__();self.fc=nn.Linear(width,outputs,bias=bias)
 def forward(self,x):return self.fc(x)
''')
        graph = self.capture(source, overrides={'init': {'outputs': 4, 'bias': False}})
        self.assertEqual(graph['discovery']['parameterCount'], 12)
        self.assertEqual(graph['discovery']['inputStatus'], 'validated-example')

    def test_same_named_repositories_restore_import_state(self):
        path_before = sys.path[:]
        module_before = sys.modules.get('net')
        for name, width in [('first', 3), ('second', 7)]:
            source = self.source(f'''from torch import nn
class Model(nn.Module):
 def __init__(self):
  super().__init__();self.fc=nn.Linear({width},2)
 def forward(self,x):return self.fc(x)
''', name)
            self.assertEqual(self.capture(source)['discovery']['parameterCount'], 2 * (width + 1))
            self.assertEqual(sys.path, path_before)
            self.assertIs(sys.modules.get('net'), module_before)

    def test_invalid_explicit_configuration_is_recoverable(self):
        source = self.source('from torch import nn\nclass Model(nn.Module):pass\n')
        config = source.parent / 'silicondevine.json'
        config.write_text(json.dumps({'init': []}))
        with self.assertRaises(ImportProblem) as error:
            self.capture(source, config=config)
        self.assertEqual(error.exception.issue['code'], 'config_invalid')

    def test_meta_incompatible_constructor_keeps_budget_guard(self):
        source = self.source('''import torch
from torch import nn
class Model(nn.Module):
 def __init__(self):
  super().__init__();self.value=torch.ones(1).item()
''')
        with self.assertRaises(ImportProblem) as error:
            self.capture(source)
        self.assertEqual(error.exception.issue['code'], 'constructor_meta_unsupported')

    def test_unknown_arguments_report_field_instead_of_generic_failure(self):
        source = self.source('from torch import nn\nclass Model(nn.Module):\n def __init__(self,width:int=3):super().__init__()\n')
        with self.assertRaises(ImportProblem) as error:
            self.capture(source, overrides={'init': {'typo': 3}})
        self.assertEqual(error.exception.issue['fields'], ['typo'])

    def test_generated_inputs_follow_model_precision(self):
        source = self.source('''from torch import nn
class Model(nn.Module):
 def __init__(self):
  super().__init__();self.fc=nn.Linear(3,2).double()
 def forward(self,x):return self.fc(x)
''')
        graph = self.capture(source)
        self.assertEqual(graph['discovery']['inputStatus'], 'validated-example')
        self.assertEqual(graph['discovery']['exampleInputs'][0]['input']['dtype'], 'float64')

    def test_configuration_conversion_does_not_imply_scalar_argument(self):
        source = self.source('''from torch import nn
class Model(nn.Module):
 def __init__(self,width):
  super().__init__();width=width['channels'];self.fc=nn.Linear(width,2)
 def forward(self,x):return self.fc(x)
''')
        graph = self.capture(source, overrides={'init': {'width': {'channels': 3}}})
        self.assertEqual(graph['discovery']['parameterCount'], 8)

    def test_plain_constructor_shape_dictionary_stays_a_mapping(self):
        def build(config):pass
        field = parameters(build, values={'config': {'shape': [3, 4]}})[0]
        self.assertEqual(field['kind'], 'mapping')
        tensor = parameters(build, values={'config': {'$tensor': {'shape': [3, 4]}}})[0]
        self.assertEqual(tensor['kind'], 'tensor')


if __name__ == '__main__':
    unittest.main()
