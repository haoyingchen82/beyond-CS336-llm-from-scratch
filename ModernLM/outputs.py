"""Structured outputs for model components and training tasks."""

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class FFNOutput:
    """Carry branch features before residual addition and optional metadata.

    Attributes:
        hidden_states: Branch output shaped [..., seq_len, hidden_size].
        auxiliary_losses: Differentiable auxiliary losses.
        metrics: Detached statistics.
    """

    hidden_states: torch.Tensor
    auxiliary_losses: dict[str, torch.Tensor] = field(default_factory=dict)
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class BlockOutput:
    """Carry features after both residual additions and branch metadata.

    Attributes:
        hidden_states: Block output shaped [..., seq_len, hidden_size].
        auxiliary_losses: Differentiable auxiliary losses from the FFN.
        metrics: Detached statistics from the FFN.
    """

    hidden_states: torch.Tensor
    auxiliary_losses: dict[str, torch.Tensor] = field(default_factory=dict)
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class BackboneOutput:
    """Carry final-normalized features and metadata from all decoder layers.

    Attributes:
        last_hidden_states: Features shaped [..., seq_len, hidden_size].
        state: None until inference state support is implemented.
        auxiliary_losses: Differentiable auxiliary losses keyed by layer.
        metrics: Detached statistics keyed by layer.
    """

    last_hidden_states: torch.Tensor
    state: None = None
    auxiliary_losses: dict[str, torch.Tensor] = field(default_factory=dict)
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class LMOutput:
    """Carry vocabulary logits and unchanged backbone metadata.

    Attributes:
        logits: Unnormalized scores shaped [..., seq_len, vocab_size].
        state: None until inference state support is implemented.
        auxiliary_losses: Differentiable auxiliary losses keyed by layer.
        metrics: Detached statistics keyed by layer.
    """

    logits: torch.Tensor
    state: None = None
    auxiliary_losses: dict[str, torch.Tensor] = field(default_factory=dict)
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskOutput:
    """Return one micro-batch's scalar contribution and logging statistics.

    Attributes:
        loss: Differentiable contribution to the update window, or None
            when this batch has no objective to optimize.
        statistics: Detached scalar sums and counts for logging,
            independent of the differentiable objective's reduction.
    """

    loss: torch.Tensor | None
    statistics: dict[str, tuple[torch.Tensor, int | float]] = field(
        default_factory=dict
    )
