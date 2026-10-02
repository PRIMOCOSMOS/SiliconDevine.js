"""Native PyTorch SAB / ISAB fixtures.

Matches the author's MAB ordering and scale=1/sqrt(dim_V):
https://github.com/juho-lee/set_transformer/blob/master/modules.py
Independent implementation; no pretrained weights.
"""
import torch
from torch import nn
from torch.nn import functional as F


class MAB(nn.Module):
    def __init__(self,width=16,heads=4):
        super().__init__()
        self.width,self.heads=width,heads
        self.q=nn.Linear(width,width)
        self.k=nn.Linear(width,width)
        self.v=nn.Linear(width,width)
        self.norm_attention=nn.LayerNorm(width)
        self.ffn=nn.Linear(width,width)
        self.norm_output=nn.LayerNorm(width)

    def forward(self,query,context):
        q,k,v=self.q(query),self.k(context),self.v(context)
        b,l,_=q.shape;s=k.shape[1];d=self.width//self.heads
        qh=q.reshape(b,l,self.heads,d).transpose(1,2)
        kh=k.reshape(b,s,self.heads,d).transpose(1,2)
        vh=v.reshape(b,s,self.heads,d).transpose(1,2)
        weights=F.scaled_dot_product_attention(qh,kh,vh,scale=self.width**-.5)
        h=self.norm_attention(q+weights.transpose(1,2).reshape(b,l,self.width))
        return self.norm_output(h+F.relu(self.ffn(h)))


class SAB(nn.Module):
    def __init__(self,width=16,heads=4):
        super().__init__();self.attention=MAB(width,heads)
    def forward(self,x):return self.attention(x,x)


class ISAB(nn.Module):
    def __init__(self,width=16,heads=4,inducing=3):
        super().__init__()
        self.inducing=nn.Parameter(torch.empty(1,inducing,width))
        nn.init.xavier_uniform_(self.inducing)
        self.to_inducing=MAB(width,heads)
        self.to_set=MAB(width,heads)
    def forward(self,x):
        h=self.to_inducing(self.inducing.repeat(x.shape[0],1,1),x)
        return self.to_set(x,h)


def build_sab():
    torch.manual_seed(73)
    return {'model':SAB(),'args':(torch.randn(1,6,16),),'options':{'backend':'export'}}


def build_isab():
    torch.manual_seed(73)
    return {'model':ISAB(),'args':(torch.randn(1,6,16),),'options':{'backend':'export'}}


build_model=build_isab
