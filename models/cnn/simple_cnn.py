import torch
import torch.nn as nn

from core.base import BaseModel


class SimpleCNN(BaseModel, nn.Module):
    """
    A simple CNN for binary classification (NC vs AD).
    Input:  (1, 64, 64)
    Output: 2 logits
    """

    def __init__(self):
        """
        Initialize the SimpleCNN model.

        The model consists of a convolutional layer with ReLU activation,
        followed by an adaptive average pooling layer, and finally a fully connected layer.

        The input shape is (1, 64, 64) and the output shape is (2,).
        """
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
