"""Next-token objectives and their accumulation-window counts."""

from typing import Literal

from torch import nn

from DataPipeline.batch import TextBatch
from ModernLM.objectives.ce import token_cross_entropy
from ModernLM.outputs import TaskOutput


def pretrain_normalizers(batch: TextBatch) -> dict[str, int]:
    """Count supervised target tokens without running the model."""
    token_count = (
        batch.targets.numel()
        if batch.loss_mask is None
        else int(batch.loss_mask.sum().item())
    )
    return {"tokens": token_count}


def pretrained_task(
    model: nn.Module,
    batch: TextBatch,
    *,
    normalizers: dict[str, int | float],
    reduction: Literal["mean", "sum"] = "mean",
) -> TaskOutput:
    """Compute this batch's contribution to a next-token objective.

    Args:
        model: Prepared model returning vocabulary logits.
        batch: Aligned token IDs and targets, optionally with a loss mask.
        normalizers: Counts prepared for the update window by the runner.
        reduction: Token mean across the window, or an unnormalized sum.

    Returns:
        A scalar loss contribution and detached token-level CE statistics.
        No supervised targets produce no objective and no CE observations.
    """
    targets = batch.targets
    if batch.loss_mask is not None:
        targets = targets[batch.loss_mask]
    token_count = targets.numel()
    if token_count == 0:
        return TaskOutput(loss=None)

    logits = model(input_ids=batch.input_ids).logits
    if batch.loss_mask is not None:
        logits = logits[batch.loss_mask]

    ce_sum = token_cross_entropy(logits, targets).sum()
    if reduction == "mean":
        loss = ce_sum / normalizers["tokens"]
    else:
        loss = ce_sum

    return TaskOutput(
        loss=loss,
        statistics={"ce": (ce_sum.detach(), token_count)},
    )
