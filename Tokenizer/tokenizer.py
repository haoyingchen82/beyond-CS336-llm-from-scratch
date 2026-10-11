import re
from itertools import combinations

from Tokenizer.base import Tokenizer, BPETokenizerParams
from Tokenizer.bpe_utils import _pretokenize, merge


class WordTokenizer(Tokenizer):
    def encode(self, string: str) -> list[int]:
        pass

    def decode(self, indices: list[int]) -> str:
        pass

class CharacterTokenizer(Tokenizer):
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))

    def decode(self, indices: list[int]) -> str:
        return "".join((map(chr, indices)))

class ByteTokenizer(Tokenizer):
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode('utf-8')
        return list(map(int, string_bytes))

    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)
        return string_bytes.decode("utf-8")


class BPETokenizer(Tokenizer):
    def __init__(self, params: BPETokenizerParams):
        self.vocab = params.vocab
        self.merges = params.merges
        self.special_tokens = params.special_tokens

        bytes_to_id = {token: token_id for token_id, token in self.vocab.items()}

        self.special_tokens_map = {token: bytes_to_id[token.encode('utf-8')] for token in self.special_tokens}



    def encode(self, string: str) -> list[int]:
        if self.special_tokens:
            pattern = "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
            segments = re.split(f"({pattern})", string) # reserve special tokens
        else:
            segments = [string]

        results = []

        for seg in segments:
            if seg in self.special_tokens:
                results.append(self.special_tokens_map[seg])
                continue

            for word in _pretokenize(seg):
                indices = tuple(map(int, word.encode('utf-8')))
                for pair, new_id in self.merges.items():
                    indices = merge(indices, pair, new_id)

                results.extend(indices)

        return results

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.vocab.get, indices))
        return b"".join(bytes_list).decode("utf-8")


def train_bpe(string: str, num_merges: int, special_tokens: list[str] | None = None) -> BPETokenizerParams:
    special_tokens = special_tokens or []
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    merges: dict[tuple[int, int], int] = {}

    # assign IDs to the special tokens first
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')

    # split the whole text in special tokens
    if special_tokens:
        pattern = "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
        segments = re.split(pattern, string)
    else:
        segments = [string]

    pre_token: dict[tuple[int, ...], int] = {}
    for seg in segments:
        for word in _pretokenize(seg):
            indices = tuple(map(int, word.encode('utf-8')))
            pre_token[indices] = pre_token.get(indices, 0) + 1

    for _ in range(num_merges):

        def _count_adjacent_pairs(pre_token: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
            counts = {}
            for indices, frequency in pre_token.items():
                for pair_0, pair_1 in zip(indices, indices[1:]):
                    counts[(pair_0, pair_1)] = counts.get((pair_0, pair_1), 0) + frequency

            return counts

        counts = _count_adjacent_pairs(pre_token)

        if not counts:
            break

        pair = max(counts, key=lambda p : (counts[p], vocab[p[0]], vocab[p[1]]))
        new_id = len(vocab)
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        merges[pair] = new_id

        new_pre_token: dict[tuple[int, ...], int] = {}

        for indices, frequency in pre_token.items():
            indices = merge(indices, pair, new_id)
            new_pre_token[indices] = frequency

        pre_token = new_pre_token

    return BPETokenizerParams(vocab=vocab, merges=merges, special_tokens=tuple(special_tokens))












