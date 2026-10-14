"""Persist and identify the project's byte-level BPE parameters."""

import hashlib
import json
from pathlib import Path

from Tokenizer.base import TokenizerParams
from Tokenizer.bpe_utils import PRETOKEN_PATTERN


def _params_payload(params: TokenizerParams) -> dict:
    """Convert parameters to the existing JSON representation."""
    return {
        "vocab": {
            str(token_id): list(token_bytes)
            for token_id, token_bytes in params.vocab.items()
        },
        "merges": [
            {"pair": list(pair), "new_id": new_id}
            for pair, new_id in params.merges.items()
        ],
        "special_tokens": list(params.special_tokens),
    }


def save_params(params: TokenizerParams, output_dir: str | Path) -> None:
    """Write exact parameters and a readable vocabulary preview."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decoded_vocab = {}
    for token_id, token_bytes in params.vocab.items():
        try:
            decoded_vocab[str(token_id)] = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            decoded_vocab[str(token_id)] = None

    for filename, data in (
        ("bpe_params.json", _params_payload(params)),
        ("bpe_vocab.json", decoded_vocab),
    ):
        output_path = output_dir / filename
        output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved: {output_path}")


def load_params(params_path: str | Path) -> TokenizerParams:
    """Restore byte values, integer IDs, and merge order from JSON."""
    payload = json.loads(Path(params_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Tokenizer parameters must be a JSON object")
    vocab_data = payload.get("vocab")
    merges_data = payload.get("merges")
    special_tokens = payload.get("special_tokens")
    if not isinstance(vocab_data, dict) or not isinstance(merges_data, list):
        raise ValueError("vocab must be an object and merges must be a list")
    if not isinstance(special_tokens, list) or any(
        not isinstance(token, str) for token in special_tokens
    ):
        raise ValueError("special_tokens must be a list of strings")

    vocab = {}
    for key, token_bytes in vocab_data.items():
        if not isinstance(token_bytes, list) or any(
            type(value) is not int for value in token_bytes
        ):
            raise ValueError("Vocabulary values must be lists of byte integers")
        token_id = int(key)
        if token_id in vocab:
            raise ValueError("Vocabulary IDs must be unique")
        vocab[token_id] = bytes(token_bytes)

    merges = {}
    for item in merges_data:
        if not isinstance(item, dict):
            raise ValueError("Each merge must be an object")
        pair = item.get("pair")
        new_id = item.get("new_id")
        if not isinstance(pair, list) or len(pair) != 2 or any(
            type(token_id) is not int for token_id in pair
        ) or type(new_id) is not int:
            raise ValueError("Each merge needs two integer IDs and an integer new_id")
        pair = tuple(pair)
        if pair in merges:
            raise ValueError("Merge pairs must be unique")
        merges[pair] = new_id

    return TokenizerParams(vocab, merges, tuple(special_tokens))


def tokenizer_fingerprint(params: TokenizerParams) -> str:
    """Hash exact IDs, ordered merges, and the fixed BPE text profile."""
    payload = {
        "profile": "project_byte_bpe_v1",
        "pretoken_pattern": PRETOKEN_PATTERN,
        "params": _params_payload(params),
    }
    canonical_json = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
