"""Basic token sampling and full-context causal generation without a cache."""

import torch

from ModernLM.model import CausalLM


@torch.no_grad()
def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample long token IDs [B, 1] from vocabulary logits [B, V].

    Temperature zero selects argmax. Positive top_k restricts candidates;
    None uses the full vocabulary. The generator must match logits.device.
    """
    if temperature < 0 or (top_k is not None and top_k <= 0):
        raise ValueError("Temperature must be nonnegative and top_k positive or None.")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    scores = logits.float() / temperature
    if top_k is not None:
        values, indices = torch.topk(scores, min(top_k, scores.shape[-1]), dim=-1)
        choices = torch.multinomial(values.softmax(dim=-1), 1, generator=generator)
        return indices.gather(dim=-1, index=choices)
    return torch.multinomial(scores.softmax(dim=-1), 1, generator=generator)


@torch.no_grad()
def generate(
    model: CausalLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Append tokens to a nonempty long prompt [B, L] on the model device.

    Forward the latest max_seq_len tokens with local positions each time.
    Return the entire prompt and generated suffix [B, L + N]. Finished
    batch rows repeat EOS until every row finishes or the token limit is
    reached. Preserve module modes and existing gradients.
    """
    if input_ids.ndim != 2 or input_ids.shape[-1] == 0 or max_new_tokens < 0:
        raise ValueError("Use nonempty prompts [B, L] and nonnegative max_new_tokens.")
    generated = input_ids.clone()
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        for _ in range(max_new_tokens):
            context = generated[:, -model.cfg.max_seq_len :]
            logits = model(input_ids=context).logits[:, -1, :]
            next_ids = sample_next_token(
                logits, temperature=temperature, top_k=top_k, generator=generator
            )
            if eos_token_id is not None:
                next_ids = torch.where(finished[:, None], eos_token_id, next_ids)
                finished |= next_ids.squeeze(-1) == eos_token_id
            generated = torch.cat((generated, next_ids), dim=-1)
            if eos_token_id is not None and bool(finished.all()):
                break
        return generated
    finally:
        for module, training in modes:
            module.training = training
