import sys,unittest
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from silicondevine import audit_model,export_model

class AuditTests(unittest.TestCase):
    def test_unknown_and_recognized(self):
        class Model(nn.Module):
            def forward(self,x):return torch.log(x)+torch.abs(x)
        report=audit_model(Model(),torch.ones(2,3))
        self.assertEqual(report['operators']['abs'],1)
        self.assertEqual(len(report['unresolved']),1)
        self.assertIn('log',report['unresolved'][0]['source'])
    def test_tensor_unfold_does_not_impersonate_image_unfold(self):
        class Model(nn.Module):
            def forward(self,x):return x.unfold(1,2,1)
        for backend in ('fx','export'):
            g=export_model(Model(),torch.ones(2,4),backend=backend)
            self.assertEqual(g['nodes'][0]['op'],'opaque')
    def test_explicit_training_dropout_is_not_identity(self):
        class Model(nn.Module):
            def forward(self,x):return torch.nn.functional.dropout(x,p=.4,training=True)
        for backend in ('fx','export'):
            g=export_model(Model(),torch.ones(2,4),backend=backend,include_values=True)
            self.assertEqual(g['nodes'][0]['op'],'opaque')
            self.assertIn('Dropout',g['nodes'][0]['attrs']['boundary_reason'])
    def test_unsupported_resampling_remains_boundary(self):
        for mode,aa in [('bicubic',False),('bilinear',True)]:
            class Model(nn.Module):
                def forward(self,x):return torch.nn.functional.interpolate(x,size=(3,3),mode=mode,antialias=aa,align_corners=False)
            g=export_model(Model(),torch.ones(1,1,5,5),backend='fx')
            self.assertEqual(g['nodes'][0]['op'],'opaque')
            self.assertIn('插值',g['nodes'][0]['attrs']['boundary_reason'])

if __name__=='__main__':unittest.main()
