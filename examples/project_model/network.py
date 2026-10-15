from torch import nn
from .blocks import ResidualBlock


class VisionModel(nn.Module):
    def __init__(self, channels=4, classes=3):
        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.block = ResidualBlock(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, classes)

    def forward(self, image):
        x = self.block(self.stem(image))
        return self.classifier(self.pool(x).flatten(1))
