import sys,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from silicondevine import export_model

def cases():
    torch.manual_seed(191)
    x=torch.linspace(-8,8,24).reshape(2,3,4)
    for module in [nn.ReLU6(),nn.Hardswish(),nn.Hardsigmoid(),nn.Hardtanh(-.3,1.7),nn.Softsign(),nn.SELU(),nn.CELU(.7),nn.Mish(),nn.LogSigmoid(),nn.LogSoftmax(1),nn.PReLU(3)]:
        if isinstance(module,nn.PReLU):module.weight.data.copy_(torch.tensor([.1,.3,.7]))
        yield type(module).__name__,nn.Sequential(module),x
    image=torch.arange(40,dtype=torch.float32).reshape(1,2,4,5)/10
    for module in [nn.ZeroPad2d((1,2,0,1)),nn.ConstantPad2d((-1,2,1,-1),2.5),nn.ReflectionPad2d((1,2,1,0)),nn.ReplicationPad2d((1,2,1,0)),nn.CircularPad2d((1,2,1,0))]:
        yield type(module).__name__,nn.Sequential(module),image
    yield 'Unfold',nn.Sequential(nn.Unfold((2,3),padding=(1,0),stride=(2,1),dilation=(1,2))),image
    patches=F.unfold(image,(2,3),padding=(1,0),stride=(1,2))
    yield 'Fold',nn.Sequential(nn.Fold((4,5),(2,3),padding=(1,0),stride=(1,2))),patches
    yield 'FoldUnbatched',nn.Sequential(nn.Fold((4,5),(2,3),padding=(1,0),stride=(1,2))),patches[0]
    class Functional(nn.Module):
        def forward(self,x):
            y=F.pad(x,(1,0,0,1),mode='reflect')
            p=F.unfold(y,(2,2),stride=1)
            return F.fold(p,(5,6),(2,2),stride=1)
    yield 'Functional',Functional(),image
    for dims in (1,3):
        value=torch.arange(2*4**dims,dtype=torch.float32).reshape(1,2,*([4]*dims))
        for kind in ('Reflection','Replication','Constant','Circular','Zero'):
            cls=getattr(nn,f'{kind}Pad{dims}d');module=cls(1,3.) if kind=='Constant' else cls(1)
            yield f'{kind}{dims}',nn.Sequential(module),value
    yield 'PixelShuffle',nn.Sequential(nn.PixelShuffle(2)),torch.randn(2,8,3,4)
    yield 'PixelUnshuffle',nn.Sequential(nn.PixelUnshuffle(2)),torch.randn(2,3,6,8)
    for cls in (nn.Dropout1d,nn.Dropout2d,nn.Dropout3d,nn.AlphaDropout,nn.FeatureAlphaDropout):
        dims=3 if cls==nn.Dropout1d else 5 if cls==nn.Dropout3d else 4
        yield cls.__name__,nn.Sequential(cls()).eval(),torch.randn(2,3,*([4]*(dims-2)))
    yield 'Unflatten',nn.Sequential(nn.Unflatten(1,(2,3))),torch.randn(2,6)

def generate():
    graphs=[]
    for name,model,x in cases():
        for backend in ('fx','export'):
            g=export_model(model,(x,),backend=backend,name=name+'-'+backend,include_values=True,value_limit=10000)
            expected=model(x).detach().flatten().tolist();out=next(t for t in g['tensors'] if t['id']==g['outputs'][0])
            torch.testing.assert_close(torch.tensor(out['data']['values']),torch.tensor(expected))
            graphs.append(g)
    return graphs

if __name__=='__main__':
    result=generate();Path('.qa/api-fixtures.json').write_text(json.dumps(result),encoding='utf-8')
    print('Exported',len(result),'FX / ATen operator fixtures with real PyTorch output checks.')
