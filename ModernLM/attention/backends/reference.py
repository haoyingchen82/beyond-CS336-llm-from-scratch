"""Head-agnostic reference attention operations."""

import torch
from einops import einops


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention over the last two axes.

    Args:
        q: Query tensor with shape [..., seq_len_q, d_qk].
        k: Key tensor with shape [..., seq_len_k, d_qk].
        v: Value tensor with shape [..., seq_len_k, d_v].
        mask: Optional boolean mask broadcastable to [..., seq_len_q,
            seq_len_k]; True allows attention. None allows every key.

    Returns:
        Tensor with shape [..., seq_len_q, d_v] and q.dtype; fully masked
        rows are zero.

    Notes:
        Inputs share a floating-point dtype, device, and broadcastable leading
        axes. Head layout and mask construction belong to the caller.
        FP16/BF16 compute in FP32; FP32/FP64 retain their precision.
    """
    compute_dtype = (
        torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    )

    with torch.autocast(device_type=q.device.type, enabled=False):
        q_compute = q.to(compute_dtype)
        k_compute = k.to(compute_dtype)
        v_compute = v.to(compute_dtype)

        scores = einops.einsum(q_compute, k_compute, "... seq_len_q d_qk, ... seq_len_k d_qk -> ... seq_len_q seq_len_k")
        scores = scores * (q.shape[-1] ** -0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
            # Avoid undefined softmax on fully masked rows.
            scores = torch.where(mask.any(dim=-1, keepdim=True), scores, 0.0)

        scores = scores - scores.amax(dim=-1, keepdim=True)
        weights = scores.exp()
        weights = weights / weights.sum(dim=-1, keepdim=True)
        if mask is not None:
            weights = weights.masked_fill(~mask, 0.0)

        output = torch.matmul(weights, v_compute)

    return output.to(q.dtype)






