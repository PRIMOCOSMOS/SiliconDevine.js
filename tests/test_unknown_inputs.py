import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from silicondevine.discovery import capture_project

class UnknownInputs(unittest.TestCase):
    def capture(self, source, extra=None):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); package=root/('unknown_'+uuid.uuid4().hex);package.mkdir()
            (package/'__init__.py').write_text('')
            (package/'model.py').write_text(source,encoding='utf8')
            if extra: (package/'data.py').write_text(extra,encoding='utf8')
            out=root/'out'
            graph=capture_project([str(package)],out)
            result=json.loads((out/'capture.json').read_text(encoding='utf8'))
            return graph,result['manifest']

    def test_unknown_shape_keeps_real_fx_connections(self):
        graph,manifest=self.capture('from torch import nn\nimport torch\nclass Network(nn.Module):\n def forward(self,signal): return torch.sin(signal)*signal\n')
        self.assertEqual(graph['producer']['backend'],'symbolic')
        self.assertEqual(graph['provenance']['topology'],'fx-dependencies')
        self.assertEqual(len(graph['nodes']),2)
        self.assertTrue(all(t['representation']=='aggregate' and 'data' not in t for t in graph['tensors']))
        self.assertEqual(manifest,{})
        Path('.qa/unknown-symbolic.json').write_text(json.dumps(graph),encoding='utf8')

    def test_dynamic_branch_inventory_never_invents_flow(self):
        graph,_=self.capture('from torch import nn\nclass Network(nn.Module):\n def __init__(self):\n  super().__init__();self.a=nn.Linear(3,3);self.b=nn.Linear(3,3)\n def forward(self,signal,other):\n  if signal.sum()>0:return self.a(signal)\n  return self.b(other)\n')
        self.assertEqual(graph['provenance']['topology'],'module-inventory')
        self.assertEqual(len(graph['nodes']),2)
        self.assertTrue(all(not n['inputs'] for n in graph['nodes']))
        Path('.qa/unknown-inventory.json').write_text(json.dumps(graph),encoding='utf8')

    def test_shape_dictionary_is_read_without_execution(self):
        graph,_=self.capture('from torch import nn\nclass Network(nn.Module):\n def forward(self,signal,context): return signal+context\n',
            'raise RuntimeError("Data pipeline must never execute")\ndef collate():\n shapes={"signal":(batch,n,3),"context":(batch,n,3)}\n return shapes\n')
        self.assertEqual(graph['discovery']['inputStatus'],'validated-example')
        self.assertEqual(graph['discovery']['exampleInputs'][0]['input']['shape'],[1,8,3])
        self.assertNotEqual(graph['producer']['backend'],'symbolic')

    def test_conflicting_shapes_remain_unknown(self):
        graph,_=self.capture('from torch import nn\nclass Network(nn.Module):\n def forward(self,signal):return signal.sin()\n',
             'a={"signal":(batch,n,3)}\nb={"signal":(batch,n,4)}\n')
        self.assertEqual(graph['discovery']['inputStatus'],'unresolved')

if __name__=='__main__':unittest.main()
