import json
import sys
import unittest
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from silicondevine import export_model


def lifted_features(v):
    x,y=v.unbind(-1)
    return torch.stack((x*x,x*y,y*y),-1)


class DesignerModel(nn.Module):
    def __init__(self):
        super().__init__();self.update=nn.Linear(2,2)

    def forward(self,x,y,valid):
        selected=x.masked_fill(~valid.unsqueeze(-1),0)
        feature=lifted_features(selected)
        residual=x+self.update(x)
        return {'features':feature,'dot':(x*y).sum(-1),'modulation':x*y,'residual':residual}


class OtherNames(DesignerModel):
    def __init__(self):
        super().__init__();self.delta=self.update;del self.update

    def forward(self,a,b,present):
        chosen=a.masked_fill(~present.unsqueeze(-1),0)
        encoded=lifted_features(chosen)
        updated=a+self.delta(a)
        return {'features':encoded,'dot':(a*b).sum(-1),'modulation':a*b,'residual':updated}


class ReusedHelper(nn.Module):
    def forward(self,x,y):
        return lifted_features(x),lifted_features(y)


class ProjectedVectors(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Linear(4,4)
    def forward(self,x):
        return self.proj(x).reshape(1,3,2,2).transpose(1,2)


class DesignTest(unittest.TestCase):
    def test_source_boundaries_and_numbers_survive_renaming(self):
        signatures=[]
        inputs=(torch.tensor([[[1.,2.],[3.,4.]]]),torch.tensor([[[.5,2.],[2.,1.]]]),torch.tensor([[True,False]]))
        for cls in (DesignerModel,OtherNames):
            torch.manual_seed(3);model=cls().eval()
            expected=list(model(*inputs).values())
            graph=export_model(model,inputs,backend='execution',include_values=True)
            tensors={t['id']:t for t in graph['tensors']}
            for out,value in zip(graph['outputs'],expected,strict=True):
                torch.testing.assert_close(torch.tensor(tensors[out]['data']['values']),value.flatten())
            self.assertTrue(any(u['kind']=='feature_construction' for u in graph['designUnits']))
            signatures.append([len(u['nodes']) for u in graph['designUnits']])
            self.assertTrue(any(n['attrs'].get('displayLabel')=='残差连接' for n in graph['nodes']))
            Path(f'.qa/design-{cls.__name__}.json').write_text(json.dumps(graph),encoding='utf8')
        self.assertEqual(signatures[0],signatures[1])

    def test_repeated_helper_invocations_keep_independent_boundaries(self):
        graph=export_model(ReusedHelper(),(torch.ones(2,2),torch.ones(2,2)*2),backend='execution',include_values=True)
        units=graph['designUnits'];self.assertEqual(len(units),2)
        self.assertTrue(set(units[0]['nodes']).isdisjoint(units[1]['nodes']))

    def test_projected_vector_organization_keeps_numerical_result(self):
        model=ProjectedVectors().eval();x=torch.arange(12,dtype=torch.float32).reshape(1,3,4)*.1
        g=export_model(model,(x,),backend='execution',include_values=True)
        value=next(t for t in g['tensors'] if t['id']==g['outputs'][0])['data']['values']
        torch.testing.assert_close(torch.tensor(value),model(x).flatten())
        Path('.qa/design-projected.json').write_text(json.dumps(g),encoding='utf8')

if __name__=='__main__':unittest.main()
