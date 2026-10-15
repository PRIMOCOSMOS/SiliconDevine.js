import json
import unittest
from pathlib import Path
import torch
from torch import nn
from silicondevine import export_model
from silicondevine.input_contract import describe_inputs
from silicondevine.symbolic import structural_model


class InputModel(nn.Module):
    def forward(self, samples, other, gain=2., *, valid=None):
        if valid is None:
            valid = torch.ones_like(other, dtype=torch.bool)
        return (samples['images'][0] + other).masked_fill(~valid, 0.) * gain


class MaskedAttention(nn.Module):
    def forward(self, query, key, value, valid):
        scores = (query @ key.transpose(-1, -2)) / 2.
        return scores.masked_fill(~valid, float('-inf')).softmax(-1) @ value


class ConstrainedConvolution(nn.Module):
    def __init__(self,fill=0):
        super().__init__();self.filter=nn.Conv2d(1,2,3,padding=1);self.fill=fill
    def forward(self, image, valid):
        return self.filter(image.masked_fill(~valid,self.fill))


class ScalarInput(nn.Module):
    def forward(self, features, gain=2.):
        return features*gain


class VariadicInput(nn.Module):
    def forward(self,*views,**conditions):
        return views[0]+conditions['context']


class VectorCondition(nn.Module):
    def __init__(self):
        super().__init__();self.gamma=nn.Parameter(torch.tensor([1.,2.,3.]))
    def forward(self,features):
        return self.gamma*(features*torch.rsqrt(features.pow(2).mean(-1,keepdim=True)+1e-5))


class ContractTests(unittest.TestCase):
    def test_nested_alias_keyword_and_optional_contract(self):
        x=torch.arange(6,dtype=torch.float32).reshape(2,3)
        for mode in ('execution','export'):
            model=InputModel();g=export_model(model,({'images':[x]},x),backend=mode,include_values=True)
            self.assertEqual([t['semantic'] for t in g['tensors'] if t['role']=='input'],["samples['images'][0]",'other'])
            params={p['name']:p for p in g['inputContract']['parameters']}
            self.assertEqual(params['valid']['status'],'default')
            self.assertEqual(params['valid']['derivedFrom'],['other'])
            self.assertIn('ones_like',params['valid']['defaultExpression'])
            self.assertEqual(params['gain']['default'],2.)
            self.assertEqual(params['valid']['kind'],'keyword_only')
            for p in ('samples','other'):self.assertEqual(len(params[p]['tensors']),1)
            out=next(t for t in g['tensors'] if t['id']==g['outputs'][0])
            torch.testing.assert_close(torch.tensor(out['data']['values']),model({'images':[x]},x).flatten())
        valid=torch.tensor([[True,False,True],[True,False,True]])
        g=export_model(InputModel(),({'images':[x]},x),kwargs={'gain':3.,'valid':valid},backend='execution',include_values=True)
        self.assertEqual([t['semantic'] for t in g['tensors'] if t['role']=='input'],["samples['images'][0]",'other','gain','valid'])
        Path('.qa/contract-nested.json').write_text(json.dumps(g),encoding='utf8')

    def test_fx_input_names_and_numeric_scalar_are_complete(self):
        g=export_model(ScalarInput(),(torch.arange(6.),),kwargs={'gain':3.},backend='fx',include_values=True)
        self.assertEqual([t['semantic'] for t in g['tensors'] if t['role']=='input'],['features','gain'])
        scalar=g['inputContract']['parameters'][1]
        self.assertEqual(scalar['structure']['value'],3.)
        self.assertEqual(scalar['tensors'][0]['shape'],[])
        Path('.qa/contract-scalar.json').write_text(json.dumps(g),encoding='utf8')

    def test_unknown_signature_keeps_required_and_optional_inputs(self):
        g=structural_model(InputModel(),'no example shape supplied')
        params={p['name']:p for p in g['inputContract']['parameters']}
        self.assertEqual(params['samples']['status'],'unresolved')
        self.assertEqual(params['samples']['structure']['shape'],['?'])
        self.assertEqual(params['valid']['status'],'default')
        self.assertEqual(params['gain']['default'],2.)
        self.assertFalse(g['provenance']['numericalExecution'])
        Path('.qa/contract-unknown.json').write_text(json.dumps(g),encoding='utf8')

    def test_variadic_arguments_keep_container_ownership(self):
        g=export_model(VariadicInput(),(torch.ones(2,3),),kwargs={'context':torch.ones(2,3)*2},backend='execution',include_values=True)
        self.assertEqual([t['semantic'] for t in g['tensors'] if t['role']=='input'],['views[0]',"conditions['context']"])
        self.assertEqual([len(p['tensors']) for p in g['inputContract']['parameters']],[1,1])
        x=torch.ones(2,3)
        alias=export_model(VariadicInput(),(x,),kwargs={'context':x},backend='execution',include_values=True)
        self.assertFalse(any(n['attrs'].get('residualInput') for n in alias['nodes']),'Aliased independent arguments do not prove a residual sublayer')

    def test_normalization_definition_has_no_class_name_dependency(self):
        x=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        model=VectorCondition();g=export_model(model,(x,),backend='execution',include_values=True)
        out=next(t for t in g['tensors'] if t['id']==g['outputs'][0])
        torch.testing.assert_close(torch.tensor(out['data']['values']),model(x).flatten())
        Path('.qa/contract-normalization.json').write_text(json.dumps(g),encoding='utf8')

    def test_mathematical_constraint_and_convolution_are_captured(self):
        q=torch.randn(1,3,4);k=torch.randn(1,3,4);v=torch.randn(1,3,4)
        mask=torch.ones(3,3,dtype=torch.bool).tril()
        g=export_model(MaskedAttention(),(q,k,v,mask),backend='execution',include_values=True)
        self.assertTrue(any(n['attrs'].get('attentionRole')=='probability' for n in g['nodes']))
        t=next(t for t in g['tensors'] if t['id']==g['outputs'][0])
        torch.testing.assert_close(torch.tensor(t['data']['values']),MaskedAttention()(q,k,v,mask).flatten())
        Path('.qa/contract-attention.json').write_text(json.dumps(g),encoding='utf8')
        model=ConstrainedConvolution().eval();x=torch.arange(16,dtype=torch.float32).reshape(1,1,4,4)
        valid=(x%3)!=0
        g=export_model(model,(x,valid),backend='execution',include_values=True)
        Path('.qa/contract-conv.json').write_text(json.dumps(g),encoding='utf8')
        out=next(t for t in g['tensors'] if t['id']==g['outputs'][0])
        torch.testing.assert_close(torch.tensor(out['data']['values']),model(x,valid).flatten())
        model=ConstrainedConvolution(fill=2.).eval()
        g=export_model(model,(x,valid),backend='execution',include_values=True)
        Path('.qa/contract-conv-fill.json').write_text(json.dumps(g),encoding='utf8')
        large=torch.arange(4225,dtype=torch.float32).reshape(1,1,65,65)*.001
        graph=export_model(ConstrainedConvolution(fill=2.),(large,large%2!=0),backend='execution',include_values=True,value_limit=10000)
        self.assertFalse(any(n['op']=='opaque' for n in graph['nodes']))
        self.assertTrue(any(n['attrs'].get('analyticCoordinates') for n in graph['nodes']))
        Path('.qa/contract-large-mask.json').write_text(json.dumps(graph),encoding='utf8')


if __name__=='__main__':unittest.main()
