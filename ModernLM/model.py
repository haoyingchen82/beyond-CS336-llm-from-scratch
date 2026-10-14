"""Dense Transformer backbone and causal language model composition."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ModernLM.blocks.decoder import DecoderBlock
from ModernLM.config import ModelConfig
from ModernLM.layers.embedding import Embedding
from ModernLM.layers.linear import Linear
from ModernLM.layers.norm import RMSNorm
from ModernLM.outputs import BackboneOutput, LMOutput


class TransformerModel(nn.Module):
    """Combine token embeddings, decoder blocks, and final RMSNorm."""

    def __init__(
        self,
        cfg: ModelConfig,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Allocate model components; initialize parameters externally."""
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = False

        self.token_embeddings = Embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            device=device,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            [DecoderBlock(cfg, device=device, dtype=dtype) for _ in range(cfg.num_layers)]
        )

        self.final_norm = RMSNorm(
            cfg.hidden_size,
            cfg.norm_eps,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        token_positions: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
    ) -> BackboneOutput:
        """Run the decoder stack and normalize its final hidden states.

        Args:
            input_ids: Integer token IDs with shape [..., seq_len].
            token_positions: Integer positions shaped [seq_len] or
                [..., seq_len]; defaults to consecutive positions from zero.
            mask: Optional boolean visibility mask intersected with causality.
            inputs_embeds: Features shaped [..., seq_len, hidden_size],
                provided instead of input_ids.

        Returns:
            Final-normalized features and metadata keyed by layer.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")

        hidden_states = (
            self.token_embeddings(input_ids) if input_ids is not None else inputs_embeds
        )
        if token_positions is None:
            token_positions = torch.arange(
                hidden_states.shape[-2], device=hidden_states.device, dtype=torch.long
            )

        auxiliary_losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}
        for layer_index, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                block_output = checkpoint(
                    layer,
                    hidden_states,
                    token_positions,
                    mask,
                    use_reentrant=False,
                )
            else:
                block_output = layer(hidden_states, token_positions, mask=mask)
            hidden_states = block_output.hidden_states
            auxiliary_losses.update(
                {
                    f"layers.{layer_index}.{name}": loss
                    for name, loss in block_output.auxiliary_losses.items()
                }
            )
            metrics.update(
                {
                    f"layers.{layer_index}.{name}": value
                    for name, value in block_output.metrics.items()
                }
            )

        return BackboneOutput(
            last_hidden_states=self.final_norm(hidden_states),
            auxiliary_losses=auxiliary_losses,
            metrics=metrics,
        )


class CausalLM(nn.Module):
    """Project final backbone features into next-token vocabulary logits."""

    def __init__(
        self,
        cfg: ModelConfig,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Allocate the backbone and an independent, bias-free LM head.

        Args:
            cfg: Model dimensions and numerical settings.
            device: Optional parameter and buffer device.
            dtype: Optional parameter dtype.

        Parameters must be initialized externally or loaded before use.
        """
        super().__init__()

        self.cfg = cfg

        self.backbone = TransformerModel(cfg, device=device, dtype=dtype)
        self.lm_head = Linear(
            in_dim=cfg.hidden_size,
            out_dim=cfg.vocab_size,
            device=device,
            dtype=dtype,
        )

    def gradient_checkpointing_enable(self) -> None:
        """Enable activation checkpointing for training decoder blocks."""
        self.backbone.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable activation checkpointing for training decoder blocks."""
        self.backbone.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        token_positions: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
    ) -> LMOutput:
        """Return vocabulary logits for every input position.

        Args:
            input_ids: Integer token IDs with shape [..., seq_len].
            token_positions: Integer positions shaped [seq_len] or
                [..., seq_len]; defaults are supplied by the backbone.
            mask: Optional boolean visibility mask intersected with causality.
            inputs_embeds: Features shaped [..., seq_len, hidden_size],
                provided instead of input_ids.

        Returns:
            Unnormalized logits [..., seq_len, vocab_size] and unchanged
            backbone state, auxiliary losses, and metrics.
        """
        backbone_output = self.backbone(
            input_ids=input_ids,
            token_positions=token_positions,
            mask=mask,
            inputs_embeds=inputs_embeds,
        )

        logits = self.lm_head(backbone_output.last_hidden_states)

        return LMOutput(
            logits=logits,
            state=backbone_output.state,
            auxiliary_losses=backbone_output.auxiliary_losses,
            metrics=backbone_output.metrics,
        )
