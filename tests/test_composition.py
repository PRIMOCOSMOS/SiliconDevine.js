"""Partitions must cover the execution exactly, including reused modules and loops."""
import sys
import unittest
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from silicondevine import export_model

class FeatureCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.a=nn.Conv2d(2,2,3,padding=1)
        self.b=nn.ReLU()
    def forward(self,x):
        return self.b(self.a(x))+x

class Branches(nn.Module):
    def __init__(self,count=3):
        super().__init__()
        self.paths=nn.ModuleDict({f'camera{i}':nn.Sequential(*[FeatureCell() for _ in range(3)]) for i in range(count)})
    def forward(self,x):
        outputs=[]
        for layer in self.paths.values():
            y=layer(x)
            outputs.append(y)
        return torch.stack(outputs).sum(0)

class RenamedBranches(Branches):
    def __init__(self,count=3):
        super().__init__(count)
        self.instruments=self.paths
        del self.paths
    def forward(self,x):
        outputs=[]
        for layer in self.instruments.values():
            y=layer(x)
            outputs.append(y)
        return torch.stack(outputs).sum(0)

class Twin(nn.Module):
    def __init__(self):
        super().__init__()
        self.left=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.Sigmoid())
        self.right=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.Sigmoid())
    def forward(self,x):
        a=self.left(x)
        b=self.right(x)
        return a+b

class Shared(Twin):
    def __init__(self):
        super().__init__()
        self.right=self.left
    def forward(self,x):
        a=self.left(x)
        b=self.right(x)
        return a+b

class Gaussian(nn.Module):
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(4,8),nn.SiLU(),nn.Linear(8,8))
    def forward(self,x):
        mu,logvar=self.head(x).chunk(2,-1)
        distribution=torch.distributions.Normal(mu,torch.exp(logvar*.5))
        return distribution.rsample()

class PlainBroadcast(Gaussian):
    def forward(self,x):
        a,b=self.head(x).chunk(2,-1)
        a,b=torch.broadcast_tensors(a,b)
        return a*b+a

class Reused(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList([nn.Sequential(nn.Linear(4,4),nn.ReLU()) for _ in range(3)])
    def forward(self,x):
        for i in range(5):
            for layer in self.layers:
                x=layer(x)
            x=x+1
        return x

def check(graph):
    tree=graph['composition']; index={u['id']:u for u in tree['units']}
    all_ids={n['id'] for n in graph['nodes']}
    seen=[]
    def walk(key):
        u=index[key]
        assert set(u['nodes']) <= all_ids
        if u['children']:
            children=[n for k in u['children'] for n in index[k]['nodes']]
            assert len(children)==len(set(children))
            assert set(children)==set(u['nodes'])
            for k in u['children']:walk(k)
    for k in tree['roots']:
        seen.extend(index[k]['nodes']);walk(k)
    assert len(seen)==len(set(seen))
    assert len(seen)+tree['hiddenDiagnostics']==len(all_ids)

class CompositionTest(unittest.TestCase):
    def capture(self,model,shape=(2,4)):
        graph=export_model(model.eval(),(torch.ones(*shape),),backend='execution',include_values=True)
        check(graph)
        return graph

    def test_generic_parallel_and_renaming(self):
        import json
        for count in (2,3,5):
            signatures=[]
            for cls in (Branches,RenamedBranches):
                g=self.capture(cls(count),(1,2,6,6))
                roots={u['id']:u for u in g['composition']['units']}
                towers=[roots[k] for k in g['composition']['roots'] if roots[k].get('visual',{}).get('stages')]
                self.assertEqual(len(towers),count)
                signatures.append([len(u['visual']['stages']) for u in towers])
                self.assertEqual(signatures[-1],[3]*count)
                Path('.qa').mkdir(exist_ok=True)
                Path(f'.qa/generic-{cls.__name__}-{count}.json').write_text(json.dumps(g),encoding='utf8')
            self.assertEqual(signatures[0],signatures[1])

    def test_attribute_branches_shared_parameters(self):
        for cls in (Twin,Shared):
            g=self.capture(cls())
            roots={u['id']:u for u in g['composition']['units']}
            branches=[roots[k] for k in g['composition']['roots'] if roots[k].get('visual',{}).get('family')]
            self.assertEqual(len(branches),2)
            self.assertTrue(set(branches[0]['nodes']).isdisjoint(branches[1]['nodes']))

    def test_probability_requires_source_evidence(self):
        g=self.capture(Gaussian())
        self.assertTrue(any(u.get('visual',{}).get('distribution') for u in g['composition']['units']))
        plain=self.capture(PlainBroadcast())
        self.assertFalse(any(u.get('visual',{}).get('distribution') for u in plain['composition']['units']))

    def test_reused_loops(self):
        graph=export_model(Reused().eval(),(torch.ones(2,4),),backend='execution',include_values=True)
        check(graph)
        roots={u['id']:u for u in graph['composition']['units']}
        self.assertEqual(len(graph['composition']['roots']),5)
        self.assertFalse(any(u.get('visual',{}).get('family') for u in roots.values()))

if __name__=='__main__':unittest.main()
