import sys,json,unittest,hashlib
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root/'python'),str(root/'examples')]
from silicondevine import export_model
from official_models import build_sab,build_isab,build_llama

class OfficialTests(unittest.TestCase):
    def test_upstream_capture(self):
        source=root/'examples/upstream/set_transformer';manifest=json.loads((source/'SOURCE.json').read_text())
        for name,sha in manifest['files'].items():self.assertEqual(hashlib.sha256((source/name).read_bytes()).hexdigest(),sha)
        graphs=[]
        for name,f in [('official_sab',build_sab),('official_isab',build_isab),('official_llama',build_llama)]:
            c=f();m=c['model'].eval();args=c['args'];kw=c['options'].get('kwargs',{})
            graph=export_model(m,args,name=name,include_values=True,value_limit=100000,total_value_limit=2000000,**c['options'])
            self.assertFalse(any(n['op']=='opaque' for n in graph['nodes']))
            with torch.no_grad():result=m(*args,**kw)
            if isinstance(result,tuple):result=result[0]
            out=next(t for t in graph['tensors'] if t['id']==graph['outputs'][0]);torch.testing.assert_close(torch.tensor(out['data']['values']).reshape(out['shape']),result,atol=3e-6,rtol=3e-6)
            units=[u for u in graph['functionalUnits'] if u['kind']=='attention'];self.assertEqual(len(units),1 if name=='official_sab' else 2)
            tensors={t['id']:t for t in graph['tensors']}
            for u in units:
                p=tensors[u['probability']];prob=torch.tensor(p['data']['values']).reshape(p['shape']);torch.testing.assert_close(prob.sum(-1),torch.ones_like(prob.sum(-1)),atol=2e-6,rtol=2e-6)
            self.assertTrue(any('Q' in n['attrs'].get('projectionRoles',[]) for n in graph['nodes']))
            self.assertTrue(graph['provenance']);self.assertTrue(any(m.get('code',{}).get('sha256') for m in graph['modules']))
            graphs.append(graph)
        (root/'.qa').mkdir(exist_ok=True);(root/'.qa/official-cases.json').write_text(json.dumps(graphs,allow_nan=False),encoding='utf-8')

    def test_fx_keyword_mean(self):
        class Reduction(torch.nn.Module):
            def forward(self,x):return x.mean(dim=-1,keepdim=True)/2
        for backend in ('fx','export'):
            g=export_model(Reduction(),(torch.randn(2,3),),backend=backend)
            n=next(n for n in g['nodes'] if n['op']=='mean');self.assertEqual(n['attrs']['dims'],[-1]);self.assertTrue(n['attrs']['keepdim']);self.assertTrue(any(n['op']=='divide' for n in g['nodes']))

    def test_names_are_not_evidence(self):
        class AttentionPretender(torch.nn.Module):
            def forward(self,x):return torch.relu(x)
        g=export_model(AttentionPretender(),(torch.randn(2,3),),backend='export');self.assertFalse(g['functionalUnits'])

if __name__=='__main__':unittest.main()
