import json
import sys
import unittest
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'python'),str(root/'examples')]
from silicondevine import export_model
from set_attention import SAB,ISAB


def reference_mab(m, query, context):
    # Author-style head split along the batch axis; independent of SDPA lowering.
    q,k,v=m.q(query),m.k(context),m.v(context)
    d=m.width//m.heads
    qh,kh,vh=(torch.cat(t.split(d,2),0) for t in (q,k,v))
    p=torch.softmax(qh.bmm(kh.transpose(1,2))/(m.width**.5),2)
    h=m.norm_attention(torch.cat((qh+p.bmm(vh)).split(q.shape[0],0),2))
    return m.norm_output(h+torch.relu(m.ffn(h)))


class SetAttentionTests(unittest.TestCase):
    def test_reference_and_permutation(self):
        torch.manual_seed(73);graphs=[]
        for cls in (SAB,ISAB):
            model=cls().eval();x=torch.randn(2,6,16)
            reference=reference_mab(model.attention,x,x) if cls is SAB else reference_mab(model.to_set,x,reference_mab(model.to_inducing,model.inducing.repeat(x.shape[0],1,1),x))
            torch.testing.assert_close(model(x),reference,atol=2e-6,rtol=2e-6)
            graph=export_model(model,(x,),backend='export',include_values=True,value_limit=100000,total_value_limit=2000000)
            self.assertFalse(any(n['op']=='opaque' for n in graph['nodes']))
            out=next(t for t in graph['tensors'] if t['id']==graph['outputs'][0])
            torch.testing.assert_close(torch.tensor(out['data']['values']).reshape(out['shape']),model(x),atol=2e-6,rtol=2e-6)
            permutation=torch.tensor([4,0,5,1,3,2])
            torch.testing.assert_close(model(x[:,permutation]),model(x)[:,permutation],atol=2e-6,rtol=2e-6)
            if cls is ISAB:
                probabilities=[next(t for t in graph['tensors'] if t['id']==n['outputs'][0])['shape'][-2:] for n in graph['nodes'] if n['op']=='softmax']
                self.assertEqual(probabilities,[[3,6],[6,3]])
                self.assertTrue(any(t['role']=='parameter' and t['shape']==[1,3,16] for t in graph['tensors']))
            graphs.append(graph)
        (root/'.qa').mkdir(exist_ok=True)
        (root/'.qa/set-cases.json').write_text(json.dumps(graphs,allow_nan=False),encoding='utf-8')


if __name__=='__main__':unittest.main()
