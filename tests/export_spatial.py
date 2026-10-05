import sys,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from silicondevine import export_model

def cases():
    torch.manual_seed(271)
    for rank in (1,2,3):
        x=torch.randn(2,2,*([3]*rank))
        for mode in ('nearest','nearest-exact',('linear','bilinear','trilinear')[rank-1]):
            for align in ([False,True] if 'linear' in mode else [None]):
                for recompute in (False,True):
                    yield f'upsample-{rank}-{mode}-{align}-{recompute}',nn.Sequential(nn.Upsample(scale_factor=1.7,mode=mode,align_corners=align,recompute_scale_factor=recompute)),(x,)
        yield f'size-{rank}',nn.Sequential(nn.Upsample(size=(1,)*rank,mode=('linear','bilinear','trilinear')[rank-1],align_corners=True)),(x,)
    class Grid(nn.Module):
        def __init__(self,mode,padding,align):super().__init__();self.mode,self.padding,self.align=mode,padding,align
        def forward(self,x,grid):return F.grid_sample(x,grid,self.mode,self.padding,self.align)
    grid=torch.tensor([[[[-2.,-1.],[0.,.5],[2.2,1.8]],[[.25,-.4],[-.7,1.2],[1.,1.]]]])
    for mode in ('bilinear','nearest'):
        for padding in ('zeros','border','reflection'):
            for align in (False,True):yield f'grid-{mode}-{padding}-{align}',Grid(mode,padding,align),(torch.randn(1,2,3,4),grid)
    class Reduce(nn.Module):
        def forward(self,x):return x.sum((0,-1)),torch.amax(x,dim=1,keepdim=True),torch.amin(x),x.sum(dim=())
    yield 'reduce',Reduce(),(torch.randn(2,3,4),)
    class Indices(nn.Module):
        def forward(self,x,idx,rows):return torch.gather(x,1,idx),x.index_select(-1,rows)
    yield 'indices',Indices(),(torch.randn(2,3),torch.tensor([[2,0],[1,2]]),torch.tensor([2,0]))
    class Keywords(nn.Module):
        def forward(self,x,idx):return torch.gather(index=idx,dim=-1,input=x)
    yield 'keyword-indices',Keywords(),(torch.randn(2,3),torch.tensor([[2,0],[1,2]]))
    yield 'anisotropic',nn.Sequential(nn.Upsample(scale_factor=(.7,1.8),mode='bilinear',align_corners=False)),(torch.randn(1,2,6,4),)

if __name__=='__main__':
    graphs=[]
    for name,model,args in cases():
        for backend in ('fx','export'):
            g=export_model(model,args,name=name+'-'+backend,backend=backend,include_values=True,value_limit=10000)
            result=model(*args);result=result if isinstance(result,tuple) else (result,)
            for oid,expected in zip(g['outputs'],result):
                out=next(t for t in g['tensors'] if t['id']==oid)
                torch.testing.assert_close(torch.tensor(out['data']['values']),expected.detach().flatten())
            graphs.append(g)
    Path('.qa/spatial-fixtures.json').write_text(json.dumps(graphs),encoding='utf-8');print('Exported',len(graphs),'spatial and indexing fixtures.')
