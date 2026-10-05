"""Traceable conditional operators. Shapes follow ordinary PyTorch conventions."""
import torch
from torch import nn
from torch.nn import functional as F

@torch.fx.wrap
def mix_kernels(coefficients, bank):
    return torch.einsum('be,eoihw->boihw',coefficients,bank)

@torch.fx.wrap
def batch_conv2d(x, kernels, stride=1, padding=0, dilation=1, groups=1):
    if x.ndim!=4 or kernels.ndim!=5 or x.shape[0]!=kernels.shape[0]:
        raise ValueError('Expected X[B,C,H,W], kernels[B,O,C/groups,kH,kW]')
    return torch.cat([F.conv2d(x[b:b+1],kernels[b],None,stride,padding,dilation,groups) for b in range(x.shape[0])],0)

class DynamicConv2d(nn.Module):
    """Input-conditioned mixture of learnable kernels, one effective kernel per batch item."""
    def __init__(self,in_channels,out_channels,kernel_size=3,experts=3,*,stride=1,padding=1,dilation=1,groups=1):
        super().__init__()
        if in_channels%groups or out_channels%groups or experts<1:raise ValueError('Invalid groups or experts')
        self.bank=nn.Parameter(torch.randn(experts,out_channels,in_channels//groups,kernel_size,kernel_size)*.1)
        self.pool=nn.AdaptiveAvgPool2d(1);self.flatten=nn.Flatten();self.router=nn.Linear(in_channels,experts);self.softmax=nn.Softmax(-1)
        self.stride,self.padding,self.dilation,self.groups=stride,padding,dilation,groups
    def forward(self,x):
        coefficients=self.softmax(self.router(self.flatten(self.pool(x))))
        kernel=mix_kernels(coefficients,self.bank)
        return batch_conv2d(x,kernel,self.stride,self.padding,self.dilation,self.groups)

class ConditionalAttention(nn.Module):
    """Cross attention: Query from x, Key/Value from condition; optional additive mask."""
    def __init__(self,dim,condition_dim,heads=2):
        super().__init__()
        if dim%heads:raise ValueError('dim must be divisible by heads')
        self.heads=heads;self.head_dim=dim//heads
        self.q=nn.Linear(dim,dim);self.k=nn.Linear(condition_dim,dim);self.v=nn.Linear(condition_dim,dim);self.out=nn.Linear(dim,dim)
    def forward(self,x,condition,mask=None):
        b,t,_=x.shape;s=condition.shape[1]
        q=self.q(x).reshape(b,t,self.heads,self.head_dim).transpose(1,2)
        k=self.k(condition).reshape(b,s,self.heads,self.head_dim).transpose(1,2)
        v=self.v(condition).reshape(b,s,self.heads,self.head_dim).transpose(1,2)
        scores=q@k.transpose(-1,-2)*(self.head_dim**-.5)
        if mask is not None:scores=scores+mask
        p=scores.softmax(-1)
        return self.out((p@v).transpose(1,2).reshape(b,t,-1))
