"""Token embedding lookup."""

import torch
from torch import nn


class Embedding(nn.Module):
    """Map token IDs to vectors with weight shape [vocab_size, emb_dim]."""

    def __init__(self, vocab_size, emb_dim, device=None, dtype=None):
        """Allocate an uninitialized embedding table.

        Args:
            vocab_size: Number of token IDs.
            emb_dim: Embedding vector width.
            device: Optional parameter device.
            dtype: Optional parameter dtype.
        """
        super(Embedding, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = emb_dim

        self.weight = nn.Parameter(
            torch.empty(
                vocab_size, emb_dim, device=device, dtype=dtype
            ), requires_grad=True
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up embedding vectors for token indices.

        Args:
            indices: Int32 or int64 tensor of non-negative token IDs.

        Returns:
            Tensor with shape [*indices.shape, embedding_dim].
        """

        assert indices.dtype in (torch.int32, torch.int64), "indices must be int32 or int64"
        assert (indices >= 0).all(), "indices must be non-negative"

        return self.weight[indices]


