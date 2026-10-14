"""Bias-free linear projection."""

import torch
from torch import nn
import einops

class Linear(nn.Module):
    """Apply a linear projection with weight shape [out_dim, in_dim]."""

    def __init__(self, in_dim, out_dim, device=None, dtype=None):
        """Allocate an uninitialized projection weight.

        Args:
            in_dim: Input feature width.
            out_dim: Output feature width.
            device: Optional parameter device.
            dtype: Optional parameter dtype.
        """
        super(Linear, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight = nn.Parameter(
            torch.empty(
                out_dim, in_dim, dtype=dtype, device=device
            ), requires_grad=True
        )

    def forward(self, x) -> torch.Tensor:
        """Project the last feature axis.

        Args:
            x: Input tensor with shape [..., in_dim].

        Returns:
            Tensor with shape [..., out_dim].
        """
        return einops.einsum(x, self.weight, "... in_dim, out_dim in_dim -> ... out_dim ")

