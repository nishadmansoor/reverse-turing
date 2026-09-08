"""
Model Architectures
===================
Shared architecture definitions. The CNN class lives here so the training
script and the serving app can never drift apart — a mismatch would make
`load_state_dict` fail (or worse, silently load into the wrong shape).
"""

import torch.nn as nn

from src.features import IMAGE_SIZE

# Three 2x max-pools reduce 224 -> 28
_CONV_OUT = IMAGE_SIZE // 8


class CNN(nn.Module):
    """Small conv net over text-as-heatmap images."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * _CONV_OUT * _CONV_OUT, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))
