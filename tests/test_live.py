import sys, json, unittest, tempfile, urllib.request, urllib.error
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from silicondevine import show, export_model

class LiveTests(unittest.TestCase):
    def test_snapshots_and_windows(self):
        model=nn.Sequential(nn.Linear(64,4))
        server=show(model,(torch.arange(128.).reshape(2,64),),port=0,block=False,value_limit=4)
        try:
            base=server.url.split('/?')[0]
            def get(path): return json.load(urllib.request.urlopen(base+path))
            status=get('/api/status');self.assertEqual(status['revision'],1)
            graph=get('/api/model')['model'];name=graph['inputs'][0]
            self.assertEqual(get(f'/api/tensor?id={name}&indices=17,65,127&revision=1')['values'],[17.,65.,127.])
            server.update(model,(torch.zeros(2,64),),value_limit=4)
            with self.assertRaises(urllib.error.HTTPError) as stale: get(f'/api/tensor?id={name}&indices=17&revision=1')
            self.assertEqual(stale.exception.code,409)
            self.assertEqual(get(f'/api/tensor?id={name}&indices=17&revision=2')['values'],[0.])
            req=urllib.request.Request(base+'/api/model',headers={'Origin':'https://example.com'})
            with self.assertRaises(urllib.error.HTTPError) as foreign:urllib.request.urlopen(req)
            self.assertEqual(foreign.exception.code,403)
            with self.assertRaises(Exception):server.update(model,(torch.ones(2,7),))
            self.assertEqual(get('/api/status')['revision'],2)
            self.assertTrue(get('/api/status')['error'])
        finally:server.close()
    def test_kwargs_dynamic_and_nested(self):
        class Model(nn.Module):
            def forward(self,x,*,other):return x+other
        model=Model();x=torch.ones(3,4);other=torch.randn(3,4)
        for backend in ['fx','export']:
            result=export_model(model,(x,),kwargs={'other':other},backend=backend,include_values=True)
            actual=next(t for t in result['tensors'] if t['id']==result['outputs'][0])['data']['values']
            self.assertEqual(actual,(x+other).flatten().tolist())
        batch=torch.export.Dim('batch',min=2,max=8)
        result=export_model(model,(x,),kwargs={'other':other},backend='export',dynamic_shapes={'x':{0:batch},'other':{0:batch}})
        self.assertTrue(result['constraints'])
        class Nested(nn.Module):
            def forward(self,pair):return pair[0]+pair[1]
        result=export_model(Nested(),((x,other),),backend='export',include_values=True)
        self.assertEqual(len(result['inputs']),2)
    def test_native_transformer_decomposition(self):
        torch.manual_seed(19)
        model=nn.TransformerEncoderLayer(8,2,16,batch_first=True,dropout=0).eval()
        x=torch.randn(1,4,8)
        result=export_model(model,(x,),backend='export',include_values=True)
        self.assertTrue(any(n['attrs'].get('lowered_from') for n in result['nodes']))
        self.assertFalse(any(n['op']=='opaque' for n in result['nodes']))
        values=next(t['data']['values'] for t in result['tensors'] if t['id']==result['outputs'][0])
        self.assertTrue(torch.allclose(torch.tensor(values),model(x).detach().flatten(),atol=1e-6,rtol=1e-6))

if __name__=='__main__':unittest.main()
