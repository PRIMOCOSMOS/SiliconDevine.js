import json
import sys
import unittest
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'python'),str(root/'examples')]
from silicondevine import export_model
from llm_models import TinyGPT,LlamaStyle,Config,RotaryEmbedding


class SDPA(nn.Module):
    def __init__(self,causal=False,gqa=False):
        super().__init__();self.causal=causal;self.gqa=gqa
    def forward(self,q,k,v,mask=None):
        return F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=self.causal,enable_gqa=self.gqa)


class LLMTests(unittest.TestCase):
    def test_native_and_causal_models(self):
        torch.manual_seed(42)
        graphs=[]
        for cls in (TinyGPT,LlamaStyle):
            for batch,length in [(1,6),(2,3)]:
                model=cls().eval();tokens=torch.randint(0,32,(batch,length))
                graph=export_model(model,(tokens,),backend='export',include_values=True,value_limit=100000,total_value_limit=2000000)
                self.assertFalse(any(n['op']=='opaque' for n in graph['nodes']))
                output=next(t for t in graph['tensors'] if t['id']==graph['outputs'][0])
                torch.testing.assert_close(torch.tensor(output['data']['values']).reshape(output['shape']),model(tokens),atol=3e-6,rtol=3e-6)
                changed=tokens.clone();changed[:,-1]=(changed[:,-1]+3)%32
                torch.testing.assert_close(model(tokens)[:,:-1],model(changed)[:,:-1],atol=1e-6,rtol=1e-6)
                for node in graph['nodes']:
                    if node['op']=='softmax' and node['attrs'].get('safe'):
                        probability=next(t for t in graph['tensors'] if t['id']==node['outputs'][0]);p=torch.tensor(probability['data']['values']).reshape(probability['shape'])
                        self.assertEqual(float(p.triu(1).abs().sum()),0.)
                if cls is TinyGPT:
                    embeds=[n for n in graph['nodes'] if n['op']=='embedding']
                    head=next(n for n in graph['nodes'] if n['group']=='lm_head')
                    self.assertEqual(embeds[0]['parameters']['weight'],head['parameters']['weight'])
                graph['name']=f'{cls.__name__}-B{batch}-T{length}'
                graphs.append(graph)
        for causal,gqa,kind in [(True,False,None),(True,True,None),(False,True,'bool'),(False,False,'float')]:
            q=torch.randn(1,4,3,4);k=torch.randn(1,2 if gqa else 4,5,4);v=torch.randn(1,2 if gqa else 4,5,4)
            mask=torch.ones(3,5,dtype=torch.bool) if kind=='bool' else torch.randn(3,5) if kind=='float' else None
            if kind=='bool':mask[0,:]=False;mask[1,3:]=False
            model=SDPA(causal,gqa)
            graph=export_model(model,(q,k,v,mask),backend='export',include_values=True,name=f'SDPA-{causal}-{gqa}-{kind}')
            out=next(t for t in graph['tensors'] if t['id']==graph['outputs'][0])
            torch.testing.assert_close(torch.tensor(out['data']['values']).reshape(out['shape']),model(q,k,v,mask),atol=1e-6,rtol=1e-6)
            graphs.append(graph)
        (root/'.qa').mkdir(exist_ok=True)
        (root/'.qa/llm-cases.json').write_text(json.dumps(graphs,allow_nan=False),encoding='utf-8')

    def test_rope_preserves_pair_norm(self):
        rope=RotaryEmbedding(Config());x=torch.randn(2,4,6,4);y=rope(x)
        torch.testing.assert_close(x.square().sum(-1),y.square().sum(-1),atol=2e-6,rtol=1e-6)
        # Independent complex reference matches adjacent-pair real implementation.
        z=torch.view_as_complex(x.reshape(2,4,6,2,2))
        expected=torch.view_as_real(z*torch.complex(rope.cosine[:6],rope.sine[:6])).flatten(-2)
        torch.testing.assert_close(y,expected)

    def test_attention_budget(self):
        model=SDPA(causal=True);x=torch.randn(1,1,513,4)
        graph=export_model(model,(x,x,x,None),backend='export')
        fused=next(n for n in graph['nodes'] if n['op']=='attention')
        self.assertIn('262144',fused['attrs']['boundary_reason'])


if __name__=='__main__':unittest.main()
