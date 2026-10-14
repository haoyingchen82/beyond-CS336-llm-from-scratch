"""Sample aligned next-token batches from an encoded token stream."""

import torch

from DataPipeline.batch import TextBatch


def get_batch(
    token_ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device | str = "cpu",
    *,
    generator: torch.Generator | None = None,
    document_offsets: torch.Tensor | None = None,
    pad_token_id: int = 0,
) -> TextBatch:
    """Sample next-token windows with replacement, optionally within documents.

    Args:
        token_ids: Encoded CPU torch.long token IDs with shape [N].
        batch_size: Number of sampled windows.
        context_length: Input length S; each window reads S+1 tokens.
        device: Device for the returned batch.
        generator: Optional CPU generator; None uses the default CPU RNG.
        document_offsets: CPU long [D+1], starting at zero and ending at N.
            None retains continuous-stream sampling. With boundaries, each long
            document contributes L-S windows; lengths 2..S contribute one padded
            sample, and lengths below two contribute none.
        pad_token_id: Valid vocabulary ID used to right-pad short documents.

    Returns:
        Aligned input IDs and targets with shape [batch_size, S], plus a boolean
        loss_mask when short documents occur. Real EOS targets remain supervised.
        Only selected windows are read from token_ids, including mapped caches.
    """
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be one-dimensional.")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive.")

    offsets = torch.arange(context_length + 1, device="cpu")
    if document_offsets is None:
        num_start_positions = token_ids.numel() - context_length
        if num_start_positions <= 0:
            raise ValueError("token_ids must contain at least context_length + 1 tokens.")
        start_indices = torch.randint(
            num_start_positions,
            (batch_size,),
            generator=generator,
            device="cpu",
        )
        windows = token_ids[start_indices[:, None] + offsets].to(device=device)
        return TextBatch(input_ids=windows[:, :-1], targets=windows[:, 1:])

    lengths = document_offsets[1:] - document_offsets[:-1]
    counts = torch.where(
        lengths >= context_length + 1,
        lengths - context_length,
        (lengths >= 2).long(),
    )
    cumulative_counts = counts.cumsum(dim=0)
    num_windows = int(cumulative_counts[-1]) if counts.numel() else 0
    if num_windows == 0:
        raise ValueError("No document contains at least two tokens.")

    ranks = torch.randint(num_windows, (batch_size,), generator=generator, device="cpu")
    documents = torch.searchsorted(cumulative_counts, ranks, right=True)
    previous_counts = cumulative_counts[documents] - counts[documents]
    start_indices = document_offsets[documents] + ranks - previous_counts
    indices = start_indices[:, None] + offsets
    valid = indices < document_offsets[documents + 1, None]

    windows = torch.full(
        (batch_size, context_length + 1), pad_token_id,
        dtype=token_ids.dtype, device="cpu",
    )
    windows[valid] = token_ids[indices[valid]]
    loss_mask = None if bool(valid.all()) else valid[:, 1:].to(device=device)
    windows = windows.to(device=device)

    return TextBatch(
        input_ids=windows[:, :-1],
        targets=windows[:, 1:],
        loss_mask=loss_mask,
    )
