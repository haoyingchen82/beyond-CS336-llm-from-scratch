"""Activation layer definitions."""

from torch import nn
import torch.nn.functional as F


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * F.sigmoid(x)
