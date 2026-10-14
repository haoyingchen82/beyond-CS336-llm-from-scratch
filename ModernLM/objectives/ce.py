"""Stable per-token cross-entropy."""

import torch


def token_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute negative log-probabilities for target token IDs.

    Args:
        logits: Unnormalized scores with shape [..., vocab_size].
        targets: Aligned next-token vocabulary IDs in [0, vocab_size), with
            shape [...] and dtype torch.long on the same device as logits.

    Returns:
        Per-token losses with shape [...], retaining the gradient path.
        FP16/BF16 scores are computed in FP32; FP32/FP64 retain their dtype.
    """
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()

    u = logits - logits.amax(dim=-1, keepdim=True)
    target_scores = u.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return torch.log(torch.sum(torch.exp(u), dim=-1)) - target_scores


