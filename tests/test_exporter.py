import sys
import unittest
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from silicondevine import export_model

class ExporterTests(unittest.TestCase):
    def test_values_and_non_mutation(self):
        model=nn.Sequential(nn.Linear(4,3),nn.ReLU(),nn.Linear(3,2))
        model.train()
        x=torch.randn(2,4)
        expected=model(x).detach().flatten().tolist()
        state=torch.random.get_rng_state().clone()
        for backend in ['fx','export']:
            result=export_model(model,(x,),backend=backend,include_values=True)
            actual=next(t for t in result['tensors'] if t['id']==result['outputs'][0])['data']['values']
            self.assertEqual(actual,expected)
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
    def test_metadata_default_and_budgets(self):
        model=nn.Sequential(nn.Linear(4,10))
        a=export_model(model,torch.ones(2,4))
        self.assertFalse(any('data' in t for t in a['tensors']))
        b=export_model(model,torch.ones(2,4),include_values=True,value_limit=5,total_value_limit=11)
        self.assertEqual(sum(len(t.get('data',{}).get('values',[])) for t in b['tensors']),11)
        self.assertTrue(all(len(t.get('data',{}).get('values',[]))<=5 for t in b['tensors']))
    def test_tuple_and_unknown_operation(self):
        class Custom(nn.Module):
            def forward(self,x):
                return {'a':torch.log(x), 'b':x+1}
        out=export_model(Custom(),torch.ones(2,3),include_values=True)
        self.assertEqual(len(out['outputs']),2)
        self.assertTrue(any(n['op']=='opaque' for n in out['nodes']))
    def test_control_flow_fails_explicitly(self):
        class Dynamic(nn.Module):
            def forward(self,x):
                return x*2 if x.sum()>0 else x*3
        with self.assertRaisesRegex(RuntimeError,'capture failed'):
            export_model(Dynamic(),torch.ones(2))
    def test_tied_parameter_identity(self):
        class Shared(nn.Module):
            def __init__(self):
                super().__init__();self.layer=nn.Linear(2,2)
            def forward(self,x):
                return self.layer(self.layer(x))
        out=export_model(Shared(),torch.ones(1,2))
        self.assertEqual(out['nodes'][0]['parameters'],out['nodes'][1]['parameters'])

    def test_repeated_operands_and_unsupported_padding(self):
        class Repeated(nn.Module):
            def forward(self,x): return x+x
        result=export_model(Repeated(),torch.ones(2))
        self.assertEqual(result['nodes'][0]['inputs'],['x','x'])
        padded=nn.Sequential(nn.Conv2d(1,1,3,padding=1,padding_mode='reflect'))
        result=export_model(padded,torch.randn(1,1,4,4))
        self.assertEqual(result['nodes'][0]['op'],'conv2d')

if __name__=='__main__': unittest.main()
