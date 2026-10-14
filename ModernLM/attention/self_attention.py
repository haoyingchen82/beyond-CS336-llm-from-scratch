import torch
from einops import einops
from torch import nn, Tensor

from ModernLM.attention.backends.reference import scaled_dot_product_attention
from ModernLM.config import ModelConfig
from ModernLM.layers.linear import Linear
from ModernLM.position.rope import RotaryPositionalEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(
            self,
            cfg: ModelConfig,
            device=None,
            dtype=None
    ):
        super(CausalSelfAttention, self).__init__()

        d_model = cfg.hidden_size       # d_head(qk)
        self.n_q_head = cfg.num_attention_heads
        d_qk_head = cfg.head_dim
        self.n_kv_head = cfg.num_kv_heads
        d_v_head = cfg.value_head_dim

        theta = cfg.rope_theta
        max_seq_len = cfg.max_seq_len


        self.q_proj = Linear(
            in_dim=d_model,
            out_dim=d_qk_head * self.n_q_head,
            dtype=dtype,
            device=device
        )

        self.k_proj = Linear(
            in_dim=d_model,
            out_dim=d_qk_head * self.n_kv_head,
            dtype=dtype,
            device=device
        )

        self.v_proj = Linear(
            in_dim=d_model,
            out_dim=d_v_head * self.n_kv_head,
            dtype=dtype,
            device=device
        )

        self.o_proj = Linear(
            in_dim=self.n_q_head * d_v_head,
            out_dim=d_model,
            dtype=dtype,
            device=device
        )

        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_head=d_qk_head,
            max_seq_len=max_seq_len,
            device=device
        )



    def forward(self,
                x: torch.Tensor,
                token_positions: torch.Tensor,
                mask: torch.Tensor | None = None,
    ):
        """Apply causal self-attention with rotary query/key features.

        Args:
            x: Input tensor with shape [..., seq_len, hidden_size].
            token_positions: Integer IDs shaped [seq_len] or [..., seq_len]
                with batch axes matching x; the head axis is added internally.
            mask: Optional boolean visibility mask on x.device, broadcastable
                to scores; intersected with the causal mask.

        Returns:
            Tensor with the same shape as x.
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = einops.rearrange(q, '... seq (n d) ->  ... n seq d', n=self.n_q_head)
        k = einops.rearrange(k, '... seq (n d) -> ... n seq d', n=self.n_kv_head)
        v = einops.rearrange(v, '... seq (n d) -> ... n seq d', n=self.n_kv_head)

        token_positions = token_positions.to(device=x.device, dtype=torch.long).unsqueeze(-2)

        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        """
        expand GQA here in the future
        """

        row = q.size(-2)
        col = k.size(-2)

        causal_mask = (
            torch.arange(row, device=q.device).unsqueeze(1)
            >= torch.arange(col, device=k.device)
        )
        mask = causal_mask if mask is None else causal_mask & mask

        av = scaled_dot_product_attention(q, k, v, mask=mask)

        av = einops.rearrange(av, '... n seq d -> ... seq (n d)', n=self.n_q_head)

        output = self.o_proj(av)

        return output
