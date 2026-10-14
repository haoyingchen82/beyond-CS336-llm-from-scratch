"""Train and export byte-level BPE from local text, Parquet, or JSONL files."""

import sys
from itertools import chain
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DataPipeline.text import iter_texts, collect_corpus
from Tokenizer.base import TokenizerParams
from Tokenizer.common import save_params
from Tokenizer.tokenizer import BPETokenizer, train_bpe



def run_bpe_train(
    input_paths: list[str | Path],
    vocab_size: int,
    special_tokens: list[str],
    *,
    text_column: str = "text",
    max_chars: int | None = None,
) -> TokenizerParams:
    """Load a corpus and train BPE to the requested total vocabulary size.

    Args:
        input_paths: Training files in their intended concatenation order.
        vocab_size: Target size including all 256 bytes and special tokens.
        special_tokens: Ordered, unique tokens reserved outside BPE merges.
        text_column: Text field for Parquet and JSONL records.
        max_chars: Total collected character budget; None reads all records.

    Returns:
        Byte vocabulary, ordered merges, and registered special tokens.
        Training can stop below vocab_size when no pairs remain.
    """
    if not input_paths:
        raise ValueError("input_paths must contain at least one training file")
    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError("special_tokens must be unique")
    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        raise ValueError("vocab_size must include all bytes and special tokens")

    texts = chain.from_iterable(
        iter_texts(path, text_column=text_column) for path in input_paths
    )
    corpus = collect_corpus(texts, max_chars=max_chars)
    print(f"Training BPE on {len(corpus):,} characters", flush=True)
    params = train_bpe(corpus, num_merges, special_tokens)
    print(f"Vocabulary size: {len(params.vocab)}; merges: {len(params.merges)}")
    return params


def main() -> None:
    """Train on a WikiText-103 training prefix and check short round trips."""
    data_dir = PROJECT_ROOT / "data" / "wikitext-103" / "wikitext-103-raw-v1"
    input_paths = [
        data_dir / "train-00000-of-00002.parquet",
        data_dir / "train-00001-of-00002.parquet",
    ]
    output_dir = PROJECT_ROOT / "outputs" / "Tokenizer" / "wikitext103_bpe"
    vocab_size = 16000
    special_tokens = ["<|endoftext|>"]
    text_column = "text"
    max_chars = None

    params = run_bpe_train(
        input_paths,
        vocab_size,
        special_tokens,
        text_column=text_column,
        max_chars=max_chars,
    )
    tokenizer = BPETokenizer(params)
    for sample in (
        "A small language model learns from WikiText.\n",
        "\u4f60\u597d, BPE! \U0001f680<|endoftext|>Next paragraph.",
    ):
        assert tokenizer.decode(tokenizer.encode(sample)) == sample
    for token in special_tokens:
        assert tokenizer.encode(token) == [tokenizer.special_tokens_map[token]]
    print("Tokenizer round-trip and special-token checks passed")
    save_params(params, output_dir)

if __name__ == "__main__":
    main()
