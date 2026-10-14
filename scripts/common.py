"""Load project BPE parameters with the selected tokenizer backend."""
import json
from pathlib import Path
from typing import Iterable

from Tokenizer.base import Tokenizer
from Tokenizer.common import load_params
from Tokenizer.tokenizer import BPETokenizer, RustTokenizer

"""Configuration for scripts that assemble a language-model experiment."""

from dataclasses import asdict, dataclass, is_dataclass

from ModernLM.config import ModelConfig, RuntimeConfig, TrainConfig

@dataclass(frozen=True)
class ExperimentConfig:
    """Combine component configs with data, budgets, and output settings.

    num_updates is an additional optimizer-update budget, including on resume.
    Encoded streams stay on CPU; only sampled batches move to the model device.
    """

    model_cfg: ModelConfig
    train_cfg: TrainConfig
    runtime_cfg: RuntimeConfig

    tokenizer_path: str | Path
    train_paths: list[str | Path]
    val_paths: list[str | Path]
    text_column: str = "text"
    output_dir: str | Path = "outputs/lm_experiment"
    resume_from: str | Path | None = None

    max_chars: int | None = None            # nums of chars from raw text (to corpus)
    max_train_tokens: int | None = None           # nums of tokens encoded
    max_val_tokens: int | None = 262_144
    token_cache_dir: str | Path | None = None
    seed: int = 42
    context_length: int = 256

    num_updates: int = 20_000
    eval_interval: int = 100
    eval_batches: int = 64

    generate_interval: int = 500
    generate_prompts: tuple[str, ...] = (
        "Artificial intelligence is",
        "The history of the Roman Empire",
        "Once upon a time,",
    )
    max_new_tokens: int = 96

    temperature: float = 0.7
    top_k: int | None = 50
    eos_token: str | None = "<|endoftext|>"

def load_bpe_tokenizer(
    params_path: str | Path, *, backend: str = "rust"
) -> Tokenizer:
    """Load exact project parameters and preserve IDs across backends."""
    tokenizers = {"python": BPETokenizer, "rust": RustTokenizer}
    if backend not in tokenizers:
        raise ValueError("backend must be python or rust")
    return tokenizers[backend](load_params(params_path))

def encode_limited_tokens(
    tokenizer: Tokenizer,
    max_tokens: int | None,
    text_iter: Iterable[str],
    *,
    eos_token: str | None,
    document_mode: str = "wikitext",
) -> Iterable:
    """Lazily yield one complete document per CPU long token tensor.

    Args:
        tokenizer: Project tokenizer with encode(text) -> list[int].
        max_tokens: Total token budget including document EOS; None is unlimited.
        text_iter: Text records matching the selected corpus format.
        eos_token: Single-token document terminator, or None to omit it.
        document_mode: Corpus format preset from document_patterns below.
            Record presets require one complete document per input string;
            flattened TXT instead needs headings, separators, or document tags.
            wikiextractor expects opening and closing tags on separate lines.

    Yields:
        CPU long [L] for one document, with a terminal EOS when configured.
        Append its length to the cache total before recording a document offset.
        Stop before a document that exceeds the budget; never truncate it.
    """
    import re
    import torch

    if max_tokens is not None and max_tokens < 0:
        raise ValueError("max_tokens must be nonnegative or None")
    if max_tokens == 0:
        return
    # (start pattern, end pattern); (None, None) preserves record boundaries.
    document_patterns = {
        "wikitext": (r"=\s+[^=\s](?:.*?[^=\s])?\s+=", None),
        "tinystories_txt": (None, r"<\|endoftext\|>"),
        "eot_text": (None, r"<\|endoftext\|>"),
        "wikiextractor": (r"<doc\b[^>]*>", r"</doc>"),
        "documents": (None, None),
        "wikipedia": (None, None),
        "fineweb": (None, None),
        "fineweb_edu": (None, None),
        "c4": (None, None),
        "openwebtext": (None, None),
        "tinystories": (None, None),
    }
    if document_mode not in document_patterns:
        raise ValueError(f"Unknown document_mode: {document_mode}")
    start_pattern, end_pattern = document_patterns[document_mode]

    eos_indices = [] if eos_token is None else tokenizer.encode(eos_token)
    if eos_token is not None and len(eos_indices) != 1:
        raise ValueError("eos_token must encode to exactly one token")

    def iter_documents():
        if start_pattern is None and end_pattern is None:
            yield from text_iter
            return

        if start_pattern is None:
            separator = re.compile(end_pattern)
            parts = []
            for text in text_iter:
                text = text if text.endswith("\n") else text + "\n"
                fragments = separator.split(text)
                for index, fragment in enumerate(fragments):
                    parts.append(fragment)
                    if index < len(fragments) - 1:
                        yield "".join(parts)
                        parts = []
            if parts:
                yield "".join(parts)
            return

        start = re.compile(start_pattern)
        end = None if end_pattern is None else re.compile(end_pattern)
        parts = []
        inside_document = end is None
        for text in text_iter:
            for line in text.splitlines(keepends=True) or [""]:
                if start.fullmatch(line.strip()):
                    if end is not None and inside_document:
                        raise ValueError("Tagged document is missing its closing marker")
                    if end is None and parts:
                        yield "".join(parts)
                    parts = []
                    inside_document = True
                    if end is None:
                        parts.append(line if line.endswith("\n") else line + "\n")
                elif end is not None and end.fullmatch(line.strip()):
                    if inside_document:
                        yield "".join(parts)
                    parts = []
                    inside_document = False
                elif inside_document:
                    parts.append(line if line.endswith("\n") else line + "\n")
        if parts and end is None:
            yield "".join(parts)
        elif inside_document:
            raise ValueError("Tagged document is missing its closing marker")

    num_tokens = 0
    for document in iter_documents():
        if eos_token is not None and document.rstrip().endswith(eos_token):
            document = document.rstrip()[:-len(eos_token)]
        if not document.strip():
            continue
        indices = tokenizer.encode(document)
        if eos_indices and (not indices or indices[-1] != eos_indices[0]):
            indices.extend(eos_indices)
        if not indices:
            continue
        if max_tokens is not None and num_tokens + len(indices) > max_tokens:
            return
        num_tokens += len(indices)
        yield torch.tensor(indices, dtype=torch.long, device="cpu")
        if max_tokens is not None and num_tokens == max_tokens:
            return

def write_json(path: str | Path, data: object) -> None:
    """Save results or dataclass configs as readable UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(data):
        data = asdict(data)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")






