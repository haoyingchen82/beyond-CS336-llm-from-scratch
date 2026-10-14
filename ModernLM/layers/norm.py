"""Root mean square normalization."""

import torch
from torch import nn

class RMSNorm(nn.Module):
    """Apply RMS normalization with learnable scale gamma [d_model]."""

    def __init__(self, d_model, eps, device=None, dtype=None):
        """Allocate an uninitialized per-feature scale.

        Args:
            d_model: Input feature width.
            eps: Denominator epsilon.
            device: Optional parameter device.
            dtype: Optional parameter dtype.
        """
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps

        self.gamma = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype), requires_grad=True)

    def forward(self, x):
        """Normalize the last axis in FP32 and restore the input dtype.

        Args:
            x: Input tensor with shape [..., d_model].

        Returns:
            Normalized tensor with the same shape and dtype as x.

        Raises:
            ValueError: If the last axis does not have width d_model.
        """
        if x.ndim == 0 or x.shape[-1] != self.d_model:
            raise ValueError(f"RMSNorm input must have last dimension {self.d_model}")

        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        rms = torch.sqrt(
            torch.sum(torch.square(x_fp32), dim=-1, keepdim=True) / self.d_model + self.eps
        )
        normalized = x_fp32 / rms
        return (normalized * self.gamma.to(torch.float32)).to(input_dtype)
