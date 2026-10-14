"""Save and restore runner state at optimizer-update boundaries."""

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from TrainingSystems.runner.base import BaseRunner


def save_training_checkpoint(
    runner: "BaseRunner",
    path: str | Path,
    *,
    train_generator: torch.Generator | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save prepared components, counters, RNG states, and experiment metadata.

    Call after an update has cleared gradients. Metadata can contain the
    tokenizer identity, corpus fingerprint, and experiment configuration.
    """
    runner.setup()
    if any(parameter.grad is not None for parameter in runner.model.parameters()):
        raise ValueError("Save checkpoints only after clearing accumulated gradients.")

    model_cfg = getattr(runner, "model_cfg", None)
    checkpoint = {
        "format_version": 1,
        "configs": {
            "model": None if model_cfg is None else asdict(model_cfg),
            "train": asdict(runner.train_cfg),
            "runtime": asdict(runner.runtime_cfg),
        },
        "model": runner.model.state_dict(),
        "optimizer": runner.optimizer.state_dict(),
        "scheduler": None if runner.scheduler is None else runner.scheduler.state_dict(),
        "global_step": runner.global_step,
        "micro_step": runner.micro_step,
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        "train_generator": None if train_generator is None else train_generator.get_state(),
        "metadata": dict(metadata or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_training_checkpoint(
    runner: "BaseRunner",
    path: str | Path,
    *,
    train_generator: torch.Generator | None = None,
    expected_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Restore a local checkpoint into a runner with matching construction.

    Expected metadata checks the supplied identity fields before applying
    weights. Return saved configs and metadata, without retaining tensors.
    Restore RNG states after setup, which may initialize fresh parameters.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_cfg = getattr(runner, "model_cfg", None)
    if model_cfg is not None and asdict(model_cfg) != checkpoint["configs"]["model"]:
        raise ValueError("Model configuration does not match the checkpoint.")
    if asdict(runner.train_cfg) != checkpoint["configs"]["train"]:
        raise ValueError("Training configuration does not match the checkpoint.")
    for name, expected in (expected_metadata or {}).items():
        if checkpoint["metadata"].get(name) != expected:
            raise ValueError(f"Checkpoint metadata does not match: {name}")

    runner.setup()
    scheduler_state = checkpoint["scheduler"]
    if (runner.scheduler is None) != (scheduler_state is None):
        raise ValueError("Scheduler configuration does not match the checkpoint.")

    runner.model.load_state_dict(checkpoint["model"])
    runner.optimizer.load_state_dict(checkpoint["optimizer"])
    if runner.scheduler is not None:
        runner.scheduler.load_state_dict(scheduler_state)
    runner.global_step = checkpoint["global_step"]
    runner.micro_step = checkpoint["micro_step"]
    runner.optimizer.zero_grad(set_to_none=True)

    torch.set_rng_state(checkpoint["cpu_rng"])
    if checkpoint["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    if train_generator is not None and checkpoint["train_generator"] is not None:
        train_generator.set_state(checkpoint["train_generator"])
    return {"configs": checkpoint["configs"], "metadata": checkpoint["metadata"]}
