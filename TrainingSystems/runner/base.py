"""Task-independent training windows, optimizer updates, and evaluation."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from itertools import islice
from typing import Any

import torch
from torch import nn

from ModernLM.config import RuntimeConfig, TrainConfig
from ModernLM.outputs import TaskOutput


class BaseRunner(ABC):
    """Separate component preparation, backward passes, and updates.

    Tasks return scalar contributions already reduced for the current
    window. normalizer_fn supplies task-defined counts without a forward.
    Override _clip_gradients to provide a different clipping strategy.
    """

    def __init__(
        self,
        train_cfg: TrainConfig,
        runtime_cfg: RuntimeConfig,
        task_fn: Callable[..., TaskOutput],
        *,
        normalizer_fn: Callable[[Any], dict[str, int | float]] | None = None,
    ) -> None:
        self.train_cfg = train_cfg
        self.runtime_cfg = runtime_cfg
        self.task_fn = task_fn
        self.normalizer_fn = normalizer_fn

        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.global_step = 0
        self.micro_step = 0
        self.is_setup = False

    def setup(self) -> None:
        """Prepare components once, preserving them on subsequent calls."""
        if self.is_setup:
            return

        self.build_model()
        self.prepare_model()
        if self.runtime_cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.build_optimizer()
        self.build_scheduler()
        self.optimizer.zero_grad(set_to_none=True)
        self.is_setup = True

    @abstractmethod
    def build_model(self) -> None:
        """Create the selected model in self.model."""
        raise NotImplementedError

    @abstractmethod
    def prepare_model(self) -> None:
        """Place, initialize or load, tie, and freeze self.model."""
        raise NotImplementedError

    def build_optimizer(self) -> None:
        """Construct self.optimizer from named PyTorch options.

        Bind only the final trainable parameters after model preparation.
        Override this method when custom parameter groups are required.
        """
        config = self.train_cfg.optimizer or {"name": "adamw", "params": {}}
        name = config["name"]
        options = dict(config["params"])
        optimizer_classes = {
            "sgd": torch.optim.SGD,
            "adamw": torch.optim.AdamW,
            "adam": torch.optim.Adam,
            "rmsprop": torch.optim.RMSprop,
            "adagrad": torch.optim.Adagrad,
        }
        optimizer_class = optimizer_classes[name.lower()]
        parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        self.optimizer = optimizer_class(parameters, **options)

    def build_scheduler(self) -> None:
        """Construct self.scheduler, advanced once per optimizer update."""
        if self.train_cfg.scheduler is None:
            self.scheduler = None
            return

        name = self.train_cfg.scheduler["name"]
        options = dict(self.train_cfg.scheduler["params"])
        scheduler_classes = {
            "steplr": torch.optim.lr_scheduler.StepLR,
            "multisteplr": torch.optim.lr_scheduler.MultiStepLR,
            "exponentiallr": torch.optim.lr_scheduler.ExponentialLR,
            "cosineannealinglr": torch.optim.lr_scheduler.CosineAnnealingLR,
            "linearlr": torch.optim.lr_scheduler.LinearLR,
            "constantlr": torch.optim.lr_scheduler.ConstantLR,
        }
        scheduler_class = scheduler_classes[name.lower()]
        self.scheduler = scheduler_class(self.optimizer, **options)

    def build_normalizers(
        self, batches: Sequence[Any]
    ) -> dict[str, int | float]:
        """Aggregate each task-defined denominator before any forward."""
        normalizers: dict[str, int | float] = {}
        if self.normalizer_fn is not None:
            for batch in batches:
                for name, count in self.normalizer_fn(batch).items():
                    normalizers[name] = normalizers.get(name, 0) + count
        return normalizers

    def train_step(
        self, batch: Any, *, normalizers: dict[str, int | float]
    ) -> dict[str, Any]:
        """Run one micro-batch and accumulate gradients without an update."""
        self.setup()
        self.model.train()

        assert normalizers is not None, "normalizers is required!"

        task_output = self.task_fn(self.model, batch, normalizers=normalizers)
        if task_output.loss is not None:
            task_output.loss.backward()
        self.micro_step += 1

        return {
            "loss": 0.0 if task_output.loss is None else float(task_output.loss.detach()),
            "has_loss": task_output.loss is not None,
            "statistics": {
                name: (float(total.detach()), count)
                for name, (total, count) in task_output.statistics.items()
            },
            "micro_step": self.micro_step,
        }

    def optimizer_step(self) -> None:
        """Clip, update, advance the scheduler, and clear accumulated grads."""
        self._clip_gradients()
        self.optimizer.step()
        self.global_step += 1
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def fit(
        self, batches: Iterable[Any], num_steps: int
    ) -> dict[str, dict[int | str, Any]]:
        """Run additional optimizer updates with sequential micro-batches.

        The source is consumed once. A partial final window uses its actual
        counts. Windows without any objective do not update the optimizer.
        Return per-window micro-step statistics and optional window losses.
        """
        history: dict[str, dict[int | str, Any]] = {}

        assert num_steps >= 0, "num_steps must be nonnegative"
        assert self.train_cfg.gradient_accumulation_steps > 0, (
            "gradient_accumulation_steps must be positive"
        )

        self.setup()
        self.optimizer.zero_grad(set_to_none=True)
        iterator = iter(batches)
        curr_steps = 0
        window_index = 0

        while curr_steps < num_steps:
            window = list(
                islice(iterator, self.train_cfg.gradient_accumulation_steps)
            )
            if not window:
                break
            normalizers = self.build_normalizers(window)
            window_loss = 0.0
            has_loss = False

            window_index += 1
            idx = str(window_index)
            history[idx] = {}

            for batch in window:
                report = self.train_step(batch, normalizers=normalizers)
                window_loss += report["loss"]
                has_loss |= report["has_loss"]

                history[idx][report["micro_step"]] = report["statistics"]

            if has_loss:
                self.optimizer_step()
                curr_steps += 1

                history[idx]["window_loss"] = window_loss

        return history

    def eval(
        self, batches: Iterable[Any], *, max_batches: int | None = None
    ) -> dict[str, float]:
        """Return count-weighted task metrics without changing training state.

        max_batches limits consumption; None consumes the finite source.
        Preserve existing gradients, update counters, and module modes.
        """
        self.setup()
        modes = [(module, module.training) for module in self.model.modules()]
        totals: dict[str, torch.Tensor] = {}
        counts: dict[str, int | float] = {}
        iterator = iter(batches)
        if max_batches is not None:
            iterator = islice(iterator, max_batches)

        try:
            self.model.eval()
            with torch.no_grad():
                for batch in iterator:
                    normalizers = self.build_normalizers([batch])
                    task_output = self.task_fn(self.model, batch, normalizers=normalizers)
                    for name, (total, count) in task_output.statistics.items():
                        totals[name] = totals.get(name, 0.0) + total.detach()
                        counts[name] = counts.get(name, 0) + count

                return {
                    name: float(total / counts[name])
                    for name, total in totals.items()
                    if counts[name] > 0
                }
        finally:
            for module, training in modes:
                module.training = training

    def _clip_gradients(self) -> None:
        """Apply configured clipping; override for a custom strategy."""
        parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]

        cfg = self.runtime_cfg
        if cfg.grad_clip_type == "norm":
            if cfg.grad_max_norm is not None and parameters:
                nn.utils.clip_grad_norm_(
                    parameters, cfg.grad_max_norm, norm_type=cfg.grad_norm_type
                )
        elif cfg.grad_clip_type == "value":
            if cfg.grad_clip_value is not None and parameters:
                nn.utils.clip_grad_value_(parameters, cfg.grad_clip_value)
        elif cfg.grad_clip_type is not None:
            raise ValueError(f"Unsupported grad_clip_type {cfg.grad_clip_type}")

