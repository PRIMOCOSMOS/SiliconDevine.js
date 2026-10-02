import sys,json
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from silicondevine import export_model
torch.manual_seed(23)
cases=[]
def add(name,layer,x,backend='fx'):
    cases.append(export_model(nn.Sequential(layer),x,backend=backend,include_values=True,value_limit=100000,name=name))
for d in [1,2,3]:
    x=torch.randn(2,4,*([5]*d))
    for padding in [1,'same','valid']:
        add(f'conv{d}-{padding}',getattr(nn,f'Conv{d}d')(4,4,3,padding=padding,groups=2),x)
    for mode in ['reflect','replicate','circular']:
        add(f'conv{d}-{mode}',getattr(nn,f'Conv{d}d')(4,4,3,padding=1,padding_mode=mode,groups=2),x)
    add(f'transpose{d}',getattr(nn,f'ConvTranspose{d}d')(4,4,3,stride=2,padding=1,output_padding=1,groups=2),x)
    add(f'max{d}',getattr(nn,f'MaxPool{d}d')(3,stride=2,padding=1,ceil_mode=True),x)
    add(f'avg{d}',getattr(nn,f'AvgPool{d}d')(3,stride=2,padding=1,ceil_mode=True,count_include_pad=False),x)
    add(f'avgpad{d}',getattr(nn,f'AvgPool{d}d')(3,stride=2,padding=1,ceil_mode=True,count_include_pad=True),x)
    add(f'adaptiveavg{d}',getattr(nn,f'AdaptiveAvgPool{d}d')(2),x)
    add(f'adaptivemax{d}',getattr(nn,f'AdaptiveMaxPool{d}d')(2),x)
    add(f'bn{d}',getattr(nn,f'BatchNorm{d}d')(4),x)
    add(f'bnnostats{d}',getattr(nn,f'BatchNorm{d}d')(4,track_running_stats=False),x)
    add(f'instance{d}',getattr(nn,f'InstanceNorm{d}d')(4,affine=True),x)
    add(f'group{d}',nn.GroupNorm(2,4),x)
for d in [1,2,3]:
    x=torch.randn(2,*([4]*d))
    add(f'unbatched-conv{d}',getattr(nn,f'Conv{d}d')(2,3,3,padding=1),x)
    add(f'unbatched-pool{d}',getattr(nn,f'AvgPool{d}d')(2),x)
add('embedding',nn.Embedding(12,4),torch.tensor([[1,3,8],[0,7,11]]))
for name,layer in [('batchnorm',nn.BatchNorm2d(4)),('groupnorm',nn.GroupNorm(2,4)),('avg',nn.AvgPool2d(3,2,1)),('max',nn.MaxPool2d(3,2,1)),('adaptive',nn.AdaptiveAvgPool2d(2)),('transpose',nn.ConvTranspose2d(4,4,3,2,1,1,groups=2))]:
    add('export-'+name,layer,torch.randn(2,4,5,5),'export')
class Cat(nn.Module):
    def forward(self,a,b):return torch.cat((a,b),dim=1)
cases.append(export_model(Cat(),(torch.randn(2,3),torch.randn(2,2)),include_values=True,name='cat'))
class Matmul(nn.Module):
    def forward(self,a,b):return a@b
for shapes in [([4],[4]),([4],[4,3]),([2,4],[4]),([3,2,4],[4,5]),([2,1,3,4],[1,3,4,2])]:
    cases.append(export_model(Matmul(),tuple(torch.randn(*s) for s in shapes),include_values=True,name='matmul'+str(shapes)))
Path('.qa').mkdir(exist_ok=True)
Path('.qa/extended.json').write_text(json.dumps(cases,allow_nan=False),encoding='utf-8')
print(f'Generated {len(cases)} PyTorch reference graphs')
