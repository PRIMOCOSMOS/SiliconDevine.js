"""Real spatial resampling and patch reconstruction; run build() from the launcher."""
import torch
from torch import nn
from torch.nn import functional as F

class SpatialPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample=nn.Upsample(size=(6,8),mode='bilinear',align_corners=False)
        y,x=torch.meshgrid(torch.linspace(-.8,.8,4),torch.linspace(-.8,.8,5),indexing='ij')
        self.register_buffer('grid',torch.stack((x+.18*torch.sin(y*3),y),-1).unsqueeze(0))
        self.patches=nn.Unfold((2,2))
        self.reconstruct=nn.Fold((4,5),(2,2))
    def forward(self,x):
        feature=self.upsample(x)
        warped=F.grid_sample(feature,self.grid,mode='bilinear',padding_mode='border',align_corners=False)
        return self.reconstruct(self.patches(warped))

def build():
    return SpatialPipeline().eval(),(torch.linspace(-1,1,24).reshape(1,2,3,4),)

if __name__=='__main__':
    from pathlib import Path
    from silicondevine import export_model
    model,args=build()
    export_model(model,args,Path(__file__).resolve().parents[1]/'demo/public/models/spatial.sd.json',name='空间采样与分块重组',include_values=True)
