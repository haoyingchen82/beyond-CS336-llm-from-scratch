"""Rotary positional embeddings with adjacent feature pairs."""

import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    """Rotate adjacent feature pairs using explicit token positions."""

    def __init__(self, theta: float, d_head: int, max_seq_len: int, device=None):
        """Cache nonpersistent FP32 sine and cosine tables.

        Args:
            theta: Positive RoPE frequency base.
            d_head: Positive, even feature width.
            max_seq_len: Number of cached token positions.
            device: Optional buffer device.
        """
        super(RotaryPositionalEmbedding, self).__init__()

        self.theta = theta
        self.d_head = d_head
        self.max_seq_len = max_seq_len

        coordinate_indices = torch.arange(0, d_head, 2, dtype=torch.float32, device=device)
        freq = theta ** (-coordinate_indices / d_head)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=freq.device)
        angles = torch.outer(positions, freq)

        self.freq = nn.Buffer(freq, persistent=False)
        self.cosine_pos = nn.Buffer(torch.cos(angles), persistent=False)
        self.sine_pos = nn.Buffer(torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Rotate input features in FP32 and restore the input dtype.

        Args:
            x: Input tensor with shape [..., seq_len, d_head].
            token_positions: Integer IDs in [0, max_seq_len), broadcastable
                to x.shape[:-1]; e.g., [seq_len] or [batch, 1, seq_len].

        Returns:
            Rotated tensor with the same shape and dtype as x.
        """

        input_dtype = x.dtype
        x_fp32 = x.float()
        cos = self.cosine_pos[token_positions].float()
        sin = self.sine_pos[token_positions].float()

        x_even = x_fp32[..., 0::2]
        x_odd = x_fp32[..., 1::2]

        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos


        return torch.stack((y_even, y_odd), dim=-1).flatten(-2).to(input_dtype)
