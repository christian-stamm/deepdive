import torch
from torch import nn

from .pl_classifier import ClassifierNet


class MultiScaleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        c = out_channels // 2

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, c, 1, padding=0),
                    nn.LeakyReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels, c, 3, padding=1),
                    nn.LeakyReLU(inplace=True),
                ),
            ]
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.fuse = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels * 2,
                kernel_size=1,
            ),
            nn.BatchNorm2d(out_channels * 2),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.fuse(x)
        x = self.pool(x)
        return x


class DenseBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class MNISTClassifier(ClassifierNet):

    def __init__(
        self,
        layer_depth=3,
        kernel_depth=16,
        learning_rate=5e-4,
        weight_decay=0.0,
        lr_scheduler_step_size=5,
        lr_scheduler_gamma=0.1,
    ):

        super().__init__(
            learning_rate,
            weight_decay,
            lr_scheduler_step_size,
            lr_scheduler_gamma,
        )

        self.features = nn.Sequential(
            DenseBlock(1 * 28 * 28, 128),
            DenseBlock(128, 64),
            DenseBlock(64, 32),
        )

        self.classifier = nn.Linear(32, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
