from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                    nn.BatchNorm2d(channels), nn.GELU(),
                                    nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.layers(x)
