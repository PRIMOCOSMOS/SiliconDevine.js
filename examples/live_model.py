"""Edit and save while start-live.cmd is running. Native PyTorch syntax."""
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(16,12),nn.GELU(),nn.Linear(12,8),nn.ReLU())
        self.latent=nn.Linear(8,4)
    def forward(self,x): return self.latent(self.encoder(x))

def build_model():
    torch.manual_seed(42)
    return Encoder(),(torch.randn(2,16),)

if __name__=='__main__':
    from silicondevine import show
    model,inputs=build_model()
    show(model,inputs)
