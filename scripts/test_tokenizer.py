import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Tokenizer.base import TokenizerParams
from Tokenizer.common import save_params
from Tokenizer.tokenizer import (
    BPETokenizer,
    ByteTokenizer,
    CharacterTokenizer,
    WordTokenizer,
    train_bpe,
)


def run_bpe_train(
    input_paths: list[str | Path],
    vocab_size: int,
    special_tokens: list[str],
    document_separator: str = "<|endoftext|>",
) -> TokenizerParams:
    if not input_paths:
        raise ValueError("Add at least one training file to input_paths.")
    if len(input_paths) > 1 and document_separator not in special_tokens:
        raise ValueError("The document separator must be registered as a special token.")

    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        raise ValueError("vocab_size must include all 256 bytes and special tokens.")

    corpus_parts = []
    for input_path in input_paths:
        print(f"Reading training data: {input_path}", flush=True)
        corpus_parts.append(Path(input_path).read_text(encoding="utf-8"))
    content = document_separator.join(corpus_parts)
    del corpus_parts

    print("Training BPE on the combined corpus", flush=True)
    params = train_bpe(content, num_merges, special_tokens)
    print(f"Vocabulary size: {len(params.vocab)}; merges: {len(params.merges)}")
    return params


def valid_tokenizer(
    tokenizer_type: str,
    sample_text: str,
    params: TokenizerParams | None = None,
):
    tokenizer_classes = {
        "word": WordTokenizer,
        "character": CharacterTokenizer,
        "byte": ByteTokenizer,
    }
    if tokenizer_type == "bpe":
        if params is None:
            raise ValueError("BPE validation requires trained params.")
        tokenizer = BPETokenizer(params)
    elif tokenizer_type in tokenizer_classes:
        tokenizer = tokenizer_classes[tokenizer_type]()
    else:
        raise ValueError("tokenizer_type must be bpe, word, character, or byte.")

    token_ids = tokenizer.encode(sample_text)
    if token_ids is None:
        raise NotImplementedError(f"{type(tokenizer).__name__}.encode() is not implemented.")
    decoded_text = tokenizer.decode(token_ids)
    if decoded_text is None:
        raise NotImplementedError(f"{type(tokenizer).__name__}.decode() is not implemented.")
    if decoded_text != sample_text:
        raise AssertionError("Decoding did not reproduce the sample text.")

    print("\nTokenizer:", type(tokenizer).__name__)
    print("\nSample text:", sample_text)
    print("Token IDs:", token_ids)
    print("Decoded text:", decoded_text)
    return tokenizer


if __name__ == "__main__":
    input_paths = [
        PROJECT_ROOT / "data" / "tinystories_sample_5M.txt",
        # Add more UTF-8 training files here:
        PROJECT_ROOT / "data" / "corpus.en",
        PROJECT_ROOT / "data" / "wikitext2_sample_5M.txt",
        PROJECT_ROOT / "data" / "fineweb_edu_sample_10M.txt"
    ]
    output_dir = PROJECT_ROOT / "outputs" / "Tokenizer"
    vocab_size = 4096
    document_separator = "<|endoftext|>"
    special_tokens = [document_separator]
    tokenizer_type = "bpe"  # Choose: bpe, word, character, byte.
    sample_text = (
        "Once upon a time, a little fox found a blue kite.\n"
        "He shared it with his friend, and they played together."
        "<|endoftext|>The next morning, they went back to the park."
    )

    params = run_bpe_train(input_paths, vocab_size, special_tokens, document_separator)
    save_params(params, output_dir)
    valid_tokenizer(tokenizer_type, sample_text, params)
