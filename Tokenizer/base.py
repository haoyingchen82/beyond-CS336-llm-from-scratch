"""Project tokenizer interfaces and shared byte-level BPE parameters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError

@dataclass(frozen=True)
class TokenizerParams:
    """Store byte vocabulary and merges in priority order.

    IDs are contiguous: 0..255 are raw bytes, followed by special tokens.
    Dictionary insertion order defines merge rank, independently of new IDs.
    Treat the dictionaries as fixed after constructing a tokenizer.
    """

    vocab: dict[int, bytes]
    merges: dict[tuple[int, int], int]
    special_tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if set(self.vocab) != set(range(len(self.vocab))):
            raise ValueError("Vocabulary IDs must be contiguous from zero")
        if any(self.vocab.get(index) != bytes([index]) for index in range(256)):
            raise ValueError("Vocabulary IDs 0..255 must contain the raw bytes")
        if any(not token for token in self.special_tokens) or len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("Special tokens must be nonempty and unique")
        special_ids = set(range(256, 256 + len(self.special_tokens)))
        for index, token in enumerate(self.special_tokens, start=256):
            if self.vocab.get(index) != token.encode("utf-8"):
                raise ValueError("Special-token IDs must follow the raw-byte vocabulary")
        ordinary = [value for index, value in self.vocab.items() if index not in special_ids]
        if len(set(ordinary)) != len(ordinary):
            raise ValueError("Ordinary vocabulary entries must have unique byte sequences")
        for (left, right), new_id in self.merges.items():
            if any(index not in self.vocab for index in (left, right, new_id)):
                raise ValueError("Merge IDs must exist in the vocabulary")
            if special_ids.intersection((left, right, new_id)):
                raise ValueError("Special tokens cannot participate in BPE merges")
            if self.vocab[new_id] != self.vocab[left] + self.vocab[right]:
                raise ValueError("A merge result must concatenate its input bytes")
