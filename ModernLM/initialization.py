"""External initialization policies for model parameters."""

import math

import torch
from torch import nn

from ModernLM.layers.embedding import Embedding
from ModernLM.layers.linear import Linear
from ModernLM.layers.norm import RMSNorm


def initialize_parameters(
    model: nn.Module,
    *,
    mode: str = "standard",
    generator: torch.Generator | None = None,
) -> None:
    """Initialize supported parameter layers in place, once per parameter.

    Args:
        model: Model or standalone layer to initialize.
        mode: Initialization policy; currently only "standard" is supported.
        generator: Optional RNG matching the parameter device. None uses the
            default PyTorch RNG.
    """
    initialized_parameters: set[int] = set()

    for module in model.modules():
        if isinstance(module, (Embedding, Linear)):
            parameter = module.weight
        elif isinstance(module, RMSNorm):
            parameter = module.gamma
        else:
            continue

        if id(parameter) in initialized_parameters:
            continue

        if isinstance(module, Embedding):
            _init_embedding(module, mode=mode, generator=generator)
        elif isinstance(module, Linear):
            _init_linear(module, mode=mode, generator=generator)
        else:
            _init_rmsnorm(module, mode=mode)

        initialized_parameters.add(id(parameter))


def _init_linear(
    module: Linear,
    *,
    mode: str = "standard",
    generator: torch.Generator | None = None,
) -> None:
    """Initialize a projection with the selected policy."""
    if mode == "standard":
        std = math.sqrt(2 / (module.in_dim + module.out_dim))
        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std,
            generator=generator,
        )
    else:
        raise ValueError(f"Unknown initialization mode: {mode}")


def _init_embedding(
    module: Embedding,
    *,
    mode: str = "standard",
    generator: torch.Generator | None = None,
) -> None:
    """Initialize an embedding table with the selected policy."""
    if mode == "standard":
        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
            generator=generator,
        )
    else:
        raise ValueError(f"Unknown initialization mode: {mode}")


def _init_rmsnorm(module: RMSNorm, *, mode: str = "standard") -> None:
    """Initialize normalization scales with the selected policy."""
    if mode == "standard":
        nn.init.ones_(module.gamma)
    else:
        raise ValueError(f"Unknown initialization mode: {mode}")
