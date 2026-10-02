"""Official PyTorch VAE plus explicitly local convolutional/conditional examples."""
import ast
import functools
import json
from pathlib import Path
import sys
import types
import torch
from torch import nn
from torch.nn import functional as F


@functools.lru_cache(maxsize=1)
def official_class():
    file=Path(__file__).parent/'upstream/pytorch_vae/main.py'
    tree=ast.parse(file.read_text(encoding='utf-8'),filename=str(file))
    definition=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VAE')
    module=types.ModuleType('silicondevine_upstream_vae')
    module.__file__=str(file)
    module.__dict__.update(torch=torch,nn=nn,F=F)
    sys.modules[module.__name__]=module
    # Execute the exact upstream class, never its dataset downloads or training CLI.
    exec(compile(ast.Module(body=[definition],type_ignores=[]),str(file),'exec'),module.__dict__)
    return module.VAE


def build_official_vae():
    torch.manual_seed(29)
    model=official_class()().eval()
    model._silicondevine_source=json.loads((Path(__file__).parent/'upstream/pytorch_vae/SOURCE.json').read_text())
    return {'model':model,'args':(torch.linspace(0,1,784).reshape(1,1,28,28),),'options':{'backend':'export'}}


class ConvVAE(nn.Module):
    """Local small Conv2d/ConvTranspose2d demonstration; not an upstream architecture."""
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(1,4,3,2,1),nn.ReLU(),nn.Flatten())
        self.mean=nn.Linear(64,4)
        self.logvar=nn.Linear(64,4)
        self.project=nn.Linear(4,64)
        self.decoder=nn.Sequential(nn.ConvTranspose2d(4,1,4,2,1),nn.Sigmoid())
    def forward(self,x):
        h=self.encoder(x);mu=self.mean(h);logvar=self.logvar(h)
        sigma=torch.exp(.5*logvar);epsilon=torch.randn_like(sigma)
        z=mu+sigma*epsilon
        return self.decoder(self.project(z).reshape(-1,4,4,4)),mu,logvar


class ConditionalVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(11,12),nn.ReLU())
        self.mean=nn.Linear(12,4);self.logvar=nn.Linear(12,4)
        self.decoder=nn.Sequential(nn.Linear(7,12),nn.ReLU(),nn.Linear(12,8),nn.Sigmoid())
    def forward(self,x,condition):
        h=self.encoder(torch.cat([x,condition],dim=-1))
        mu=self.mean(h);logvar=self.logvar(h)
        sigma=torch.exp(.5*logvar);epsilon=torch.randn_like(sigma)
        z=mu+sigma*epsilon
        return self.decoder(torch.cat([z,condition],dim=-1)),mu,logvar


def build_conv_vae():
    torch.manual_seed(29)
    model=ConvVAE().eval();model._silicondevine_source={'implementation':'Local ConvVAE example','weights':'random initialization; synthetic inputs'}
    return {'model':model,'args':(torch.linspace(0,1,128).reshape(2,1,8,8),),'options':{'backend':'export'}}


def build_conditional_vae():
    torch.manual_seed(29)
    model=ConditionalVAE().eval();model._silicondevine_source={'implementation':'Local ConditionalVAE example','weights':'random initialization; synthetic inputs'}
    return {'model':model,'args':(torch.linspace(0,1,16).reshape(2,8),torch.tensor([[1.,0.,0.],[0.,1.,0.]])),'options':{'backend':'export'}}

build_model=build_official_vae
