import torch
import torch.nn as nn
from torchvision import models


class ResNet3DClassifier(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        b, c, h, w, d = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(b * d, c, h, w)
        features = self.backbone(x)
        features = features.view(b, d, -1).mean(dim=1)
        return features
