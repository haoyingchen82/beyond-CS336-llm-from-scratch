"""Model and training configuration declarations."""

from dataclasses import dataclass, field
from typing import Any, Literal

import torch


@dataclass(frozen=True)
class ModelConfig:
    """Store immutable model dimensions and numerical settings.

    Attributes:
        vocab_size: Number of token IDs.
        hidden_size: Residual stream width.
        num_layers: Number of decoder blocks.
        num_attention_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head query/key feature width.
        value_head_dim: Per-head value feature width.
        intermediate_size: Feed-forward hidden width.
        max_seq_len: Maximum sequence length.
        norm_eps: RMSNorm denominator epsilon.
        rope_theta: RoPE frequency base.
    """

    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    value_head_dim: int
    intermediate_size: int
    max_seq_len: int
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        """Check the layer count and head-width parity."""
        assert self.num_layers > 0, "num_layers must be a positive integer"
        assert self.head_dim % 2 == 0, "head_dim must be the matching even number"


@dataclass(frozen=True)
class RuntimeConfig:
    """Select placement, clipping, and activation checkpointing.

    grad_max_norm=None disables norm clipping. Value clipping uses
    grad_clip_value. dtype describes parameter storage, not autocast.
    """

    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32
    grad_max_norm: float | None = None
    grad_clip_type: Literal["norm", "value"] | None = "norm"
    grad_clip_value: float | None = None
    grad_norm_type: float = 2.0
    gradient_checkpointing: bool = False
    init_mode: str = "standard"


@dataclass(frozen=True)
class TrainConfig:
    """Store named optimizer/scheduler options and micro-batches per update.

    Construction dictionaries separate names from PyTorch keyword arguments.
    An empty optimizer config uses AdamW; scheduler=None keeps a fixed LR.
    Scheduler intervals are measured in optimizer updates, not epochs.

    Examples:
        optimizer={"name": "sgd", "params": {}}
        optimizer={"name": "adamw", "params": {"lr": 3e-4, "weight_decay": 0.1}}
        scheduler={"name": "StepLR", "params": {"step_size": 100, "gamma": 0.5}}

    params contains constructor options. The runner supplies model parameters
    to the optimizer and supplies the optimizer to the scheduler.
    """

    optimizer: dict[str, Any] = field(default_factory=dict)
    scheduler: dict[str, Any] | None = None
    gradient_accumulation_steps: int = 1
    batch_size: int = 2
