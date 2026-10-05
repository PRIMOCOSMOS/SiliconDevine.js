import copy,sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from source_models import build_source
from silicondevine import export_model

class SourceExecution(unittest.TestCase):
    def test_original_forward_outputs_and_parameter_identity(self):
        for family,lightning in [('deepseek_v3',False),('glm45',False),('minimax_m1',False),('minimax_m1',True)]:
            with self.subTest(family=family,lightning=lightning):
                model,args=build_source(family,lightning)
                before={k:v.clone() for k,v in model.state_dict().items()}
                with torch.no_grad():expected=copy.deepcopy(model)(*args)
                graph=export_model(model,args,backend='execution',include_values=True)
                out=next(t for t in graph['tensors'] if t['id']==graph['outputs'][0])
                actual=torch.tensor(out['data']['values']).reshape(out['shape'])
                torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
                self.assertFalse(any(n['op']=='opaque' for n in graph['nodes']))
                for key,value in model.state_dict().items():torch.testing.assert_close(value,before[key])
                self.assertTrue(any(n['op']=='rmsnorm' and n['attrs'].get('children') for n in graph['nodes']))
                self.assertTrue(any(n['attrs'].get('codeTrace') for n in graph['nodes']))

if __name__=='__main__':unittest.main()
