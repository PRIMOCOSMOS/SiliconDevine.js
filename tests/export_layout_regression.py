"""Real Conv/BatchNorm/residual capture used by the spatial regression tests."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
import torch
from torch import nn
from silicondevine import export_model

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(64)
    def forward(self, x):
        return torch.relu(x + self.norm2(self.conv2(torch.relu(self.norm1(self.conv1(x))))))

torch.manual_seed(19)
model = nn.Sequential(ResidualBlock()).eval()
for backend in ['fx', 'export']:
    graph = export_model(model, (torch.randn(1,64,8,8),), ROOT/f'.qa/layout-residual-{backend}.json', backend=backend, include_values=True)
    print(backend, len(graph['nodes']), [(n['op'], n.get('parameters')) for n in graph['nodes']])

if len(sys.argv) > 1:
    import runpy
    namespace = runpy.run_path(sys.argv[1], run_name='layout_test_model')
    model, inputs = namespace['build_model']()
    graph = export_model(model, inputs, ROOT/'.qa/layout-user.json', backend='export', include_values=True)
    print('User model:', len(graph['nodes']), 'operators')
