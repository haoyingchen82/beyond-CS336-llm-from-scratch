"""Data containers for aligned language-model batches."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TextBatch:
    """Carry aligned IDs and targets shaped [batch_size, seq_len].

    loss_mask is an optional boolean mask selecting supervised targets;
    it does not control the model's attention visibility.
    """

    input_ids: torch.Tensor
    targets: torch.Tensor
    loss_mask: torch.Tensor | None = None
