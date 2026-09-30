"""Run from the project root: python examples/export_models.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import torch
from torch import nn
from silicondevine import export_model

torch.manual_seed(42)
root = Path(__file__).resolve().parents[1] / 'demo/public/models'

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(8, 8)
        self.k = nn.Linear(8, 8)
        self.v = nn.Linear(8, 8)
        self.output = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)
    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = (q @ k.transpose(-1, -2)) * (8 ** -.5)
        p = scores.softmax(dim=-1)
        return self.norm(x + self.output(p @ v))

class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 2, 3, padding=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(2, 2, 3, padding=1)
    def forward(self,x):
        return self.act(self.conv2(self.act(self.conv1(x))) + x)

cases = [
    ('mlp', nn.Sequential(nn.Linear(8, 12), nn.GELU(), nn.Linear(12, 6), nn.ReLU(), nn.Linear(6, 4)), torch.randn(2,8), 'fx'),
    ('conv2d', Residual(), torch.randn(1,2,8,8), 'fx'),
    ('conv3d', nn.Sequential(nn.Conv3d(1,2,3,padding=1),nn.ReLU(),nn.Conv3d(2,2,3,padding=1)), torch.randn(1,1,4,4,4), 'fx'),
    ('attention', Attention(), torch.randn(1,4,8), 'export'),
]
for name, model, x, backend in cases:
    graph = export_model(model, (x,), root / f'{name}.sd.json', backend=backend, include_values=True, name=name)
    print(f'{name}: {len(graph["nodes"])} operators, {len(graph["tensors"])} real tensors')
