from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AutoPilot(nn.Module):
    """Simple CNN classifier producing raw logits.

    Important:
    - The final layer intentionally does NOT use Softmax.
    - `nn.CrossEntropyLoss` expects raw logits and integer class labels.
    """

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        num_classes: int,
        model_depth: int,
        kernel_depth: int,
        kernel_size: int,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()

        blocks: list[nn.Module] = [
            ConvBlock(
                in_channels=in_channels,
                out_channels=kernel_depth,
                kernel_size=kernel_size,
                negative_slope=negative_slope,
            )
        ]

        blocks.extend(
            ConvBlock(
                in_channels=kernel_depth,
                out_channels=kernel_depth,
                kernel_size=kernel_size,
                negative_slope=negative_slope,
            )
            for _ in range(model_depth)
        )

        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(kernel_depth * image_size * image_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
