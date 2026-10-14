"""Build and initialize a dense causal LM with shared training services."""

from ModernLM.config import ModelConfig, RuntimeConfig, TrainConfig
from ModernLM.initialization import initialize_parameters
from ModernLM.model import CausalLM
from TrainingSystems.runner.base import BaseRunner


class LMRunner(BaseRunner):
    """Select the LM architecture and initialization policy from configs."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        runtime_cfg: RuntimeConfig,
        task_fn,
        *,
        normalizer_fn=None,
    ) -> None:
        super().__init__(train_cfg, runtime_cfg, task_fn, normalizer_fn=normalizer_fn)
        self.model_cfg = model_cfg

    def build_model(self) -> None:
        """Allocate self.model on the configured device and parameter dtype."""
        self.model = CausalLM(
            self.model_cfg,
            device=self.runtime_cfg.device,
            dtype=self.runtime_cfg.dtype,
        )

    def prepare_model(self) -> None:
        """Initialize allocated parameters with the configured policy."""
        initialize_parameters(self.model, mode=self.runtime_cfg.init_mode)
