"""Dense feed-forward branches."""

import torch
from einops import einops
from torch import nn

from ModernLM.layers.activations import SiLU
from ModernLM.layers.linear import Linear
from ModernLM.outputs import FFNOutput


class SwiGLU(nn.Module):
    """Apply gated SiLU with gate, up, and down projections."""

    def __init__(
            self,
            d_model: int,
            d_interm: int,
            device=None,
            dtype=None
    ):
        """Allocate uninitialized projections; initialize them externally."""
        super(SwiGLU, self).__init__()

        self.d_model = d_model
        self.d_interm = d_interm

        self.gate_proj = Linear(
            in_dim=d_model,
            out_dim=d_interm,
            device=device,
            dtype=dtype
        )

        self.up_proj = Linear(
            in_dim=d_model,
            out_dim=d_interm,
            device=device,
            dtype=dtype
        )

        self.down_proj = Linear(
            in_dim=d_interm,
            out_dim=d_model,
            device=device,
            dtype=dtype
        )

        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> FFNOutput:
        """Return branch features [..., d_model] before residual addition."""
        g = self.silu(self.gate_proj(x))
        x = self.up_proj(x)
        y = self.down_proj(einops.einsum(g, x, "... d, ... d -> ... d"))

        return FFNOutput(y)
