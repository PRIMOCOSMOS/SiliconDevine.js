import hashlib,json,sys,unittest
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root/'python'),str(root/'examples')]
from silicondevine import export_model
from vae_models import build_official_vae,build_conv_vae,build_conditional_vae

class VAETests(unittest.TestCase):
    def test_upstream_and_stochastic_values(self):
        folder=root/'examples/upstream/pytorch_vae'
        self.assertEqual(hashlib.sha256((folder/'main.py').read_bytes()).hexdigest(),json.loads((folder/'SOURCE.json').read_text())['sha256'])
        for factory in (build_official_vae,build_conv_vae,build_conditional_vae):
            spec=factory();torch.manual_seed(173);state=torch.random.get_rng_state().clone()
            graph=export_model(spec['model'],spec['args'],include_values=True,**spec['options'])
            self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
            with torch.no_grad():expected=spec['model'](*spec['args'])
            tensors={t['id']:t for t in graph['tensors']}
            for actual,tid in zip(expected,graph['outputs']):
                t=tensors[tid];torch.testing.assert_close(torch.tensor(t['data']['values']).reshape(t['shape']),actual,atol=3e-6,rtol=3e-6)
            units=[u for u in graph['functionalUnits'] if u['kind']=='gaussian_reparameterization'];self.assertEqual(len(units),1)
            self.assertFalse(any(n['op']=='opaque' for n in graph['nodes']))

    def test_recognition_without_value_export(self):
        spec=build_conv_vae()
        for backend in ('fx','export'):
            graph=export_model(spec['model'],spec['args'],backend=backend)
            self.assertEqual(sum(u['kind']=='gaussian_reparameterization' for u in graph['functionalUnits']),1)

    def test_not_a_canonical_logvariance(self):
        class DifferentScale(torch.nn.Module):
            def forward(self,mu,logvar):
                return mu+torch.randn_like(mu)*torch.exp(logvar)
        graph=export_model(DifferentScale(),(torch.zeros(2,4),torch.ones(2,4)),backend='export',include_values=True)
        self.assertFalse(any(u['kind']=='gaussian_reparameterization' for u in graph['functionalUnits']))

if __name__=='__main__':unittest.main()
