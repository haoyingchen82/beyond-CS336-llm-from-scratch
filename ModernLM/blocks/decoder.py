"""Serial pre-RMSNorm decoder blocks."""

import torch
from torch import nn

from ModernLM.attention.self_attention import CausalSelfAttention
from ModernLM.config import ModelConfig
from ModernLM.feedforward.dense import SwiGLU
from ModernLM.layers.norm import RMSNorm
from ModernLM.outputs import BlockOutput


class DecoderBlock(nn.Module):
    """Combine causal attention and an FFN with separate residual branches."""

    def __init__(
            self,
            cfg: ModelConfig,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None
    ):
        """Construct two norms, causal self-attention, and a SwiGLU FFN.

        Args:
            cfg: Model dimensions and numerical settings.
            device: Optional parameter and buffer device.
            dtype: Optional parameter dtype.
        """
        super().__init__()

        self.d_model = cfg.hidden_size
        self.eps = cfg.norm_eps
        self.d_interm = cfg.intermediate_size

        self.attn_norm = RMSNorm(
            self.d_model, self.eps, device=device, dtype=dtype
        )

        self.ffn_norm = RMSNorm(
            self.d_model, self.eps, device=device, dtype=dtype
        )

        self.mha = CausalSelfAttention(cfg, device=device, dtype=dtype)

        self.ffn = SwiGLU(
            self.d_model, self.d_interm, device=device, dtype=dtype
        )

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor,
            mask: torch.Tensor | None = None,
    ) -> BlockOutput:
        """Apply attention, then the FFN, with pre-norm residual additions.

        Args:
            x: Input tensor with shape [..., seq_len, hidden_size].
            token_positions: Integer IDs shaped [seq_len] or [..., seq_len].
            mask: Optional boolean visibility mask intersected with causality.

        Returns:
            Residual-updated hidden states and unchanged FFN metadata.
        """
        attn_output = self.mha(self.attn_norm(x), token_positions, mask)
        hidden_states = x + attn_output
        ffn_output = self.ffn(self.ffn_norm(hidden_states))
        hidden_states = hidden_states + ffn_output.hidden_states

        return BlockOutput(
            hidden_states=hidden_states,
            auxiliary_losses=ffn_output.auxiliary_losses,
            metrics=ffn_output.metrics,
        )
