from dataclasses import dataclass
from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError

@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int, bytes]
    merges: dict[tuple[int, int], int]
    special_tokens: tuple[str, ...]