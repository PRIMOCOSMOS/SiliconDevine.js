"""Open in the launcher; both examples accept ordinary PyTorch input tensors."""
import torch
from silicondevine import DynamicConv2d, ConditionalAttention, export_model

def build_dynamic():
    torch.manual_seed(41)
    return DynamicConv2d(2,3,3,experts=3).eval(), (torch.randn(2,2,5,5),)

def build_conditional():
    torch.manual_seed(41)
    return ConditionalAttention(4,6,heads=2).eval(), (torch.randn(1,3,4),torch.randn(1,4,6),torch.zeros(3,4))

if __name__=='__main__':
    from pathlib import Path
    folder=Path(__file__).resolve().parents[1]/'demo/public/models'
    for name,factory in [('dynamic_conv',build_dynamic),('conditional_attention',build_conditional)]:
        model,args=factory();export_model(model,args,folder/(name+'.sd.json'),include_values=True,value_limit=65536,backend='fx' if name=='dynamic_conv' else 'export')
