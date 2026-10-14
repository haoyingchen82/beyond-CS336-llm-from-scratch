import json
import re
from collections.abc import Iterable
from itertools import islice

from tokenizers import Regex, Tokenizer as HFTokenizer
from tokenizers import models, pre_tokenizers, trainers

from Tokenizer.base import Tokenizer, TokenizerParams
from Tokenizer.bpe_utils import PRETOKEN_PATTERN, _pretokenize, merge, _build_maps


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
    def __init__(self, params: TokenizerParams):
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


def train_bpe(string: str, num_merges: int, special_tokens: list[str] | None = None) -> TokenizerParams:
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

    return TokenizerParams(vocab=vocab, merges=merges, special_tokens=tuple(special_tokens))

class RustTokenizer(Tokenizer):
    def __init__(self, params: TokenizerParams):
        self.params = params
        self.vocab = params.vocab
        self.merges = params.merges
        self.special_tokens = params.special_tokens
        self.special_tokens_map = {
            token: token_id
            for token_id, token in enumerate(self.special_tokens, start=256)
        }

        _BYTE2CHAR, _ = _build_maps()

        def bytes_to_bytelevel(raw_bytes: bytes) -> str:
            return "".join(_BYTE2CHAR[b] for b in raw_bytes)

        hf_vocabs = {}
        hf_merges = []

        for _id, token_bytes in self.vocab.items():
            if _id not in self.special_tokens_map.values():
                hf_vocabs[bytes_to_bytelevel(token_bytes)] = _id

        for merge, new_id in self.merges.items():
            left_id = merge[0]
            right_id = merge[1]

            hf_merges.append(
                (
                    bytes_to_bytelevel(self.vocab[left_id]),
                    bytes_to_bytelevel(self.vocab[right_id])
                )
            )

        bpe_model = models.BPE(vocab=hf_vocabs, merges=hf_merges)
        self.backend = HFTokenizer(bpe_model)
        self.backend.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(PRETOKEN_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])

    def encode(self, string: str) -> list[int]:
        """Encode ordinary text with Rust BPE and preserve special-token IDs."""
        if self.special_tokens:
            pattern = "|".join(
                re.escape(token)
                for token in sorted(self.special_tokens, key=len, reverse=True)
            )
            segments = re.split(f"({pattern})", string)
        else:
            segments = [string]

        indices = []
        for segment in segments:
            if segment in self.special_tokens_map:
                indices.append(self.special_tokens_map[segment])
            else:
                indices.extend(
                    self.backend.encode(segment, add_special_tokens=False).ids
                )
        return indices

    def decode(self, indices: list[int]) -> str:
        """Join project vocabulary bytes, preserving special tokens."""
        return b"".join(self.vocab[index] for index in indices).decode(
            "utf-8", errors="replace"
        )

    @staticmethod
    def train(
        texts: Iterable[str],
        vocab_size: int,
        special_tokens: tuple[str, ...] = (),
        *,
        min_frequency: int = 1,
        batch_size: int = 1024,
    ) -> "RustTokenizer":
        """Train from text records and return a tokenizer with project IDs.

        vocab_size includes all 256 bytes and the reserved special tokens.
        Save the returned instance's params with Tokenizer.common.save_params.
        """
        special_tokens = tuple(special_tokens)
        backend = HFTokenizer(models.BPE())
        backend.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(PRETOKEN_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size - len(special_tokens),
            min_frequency=min_frequency,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        pattern = (
            re.compile("|".join(
                re.escape(token)
                for token in sorted(special_tokens, key=len, reverse=True)
            ))
            if special_tokens else None
        )

        def pre_seg():
            for text in texts:
                segments = pattern.split(text) if pattern else [text]
                for segment in segments:
                    if segment:
                        yield segment

        def batches():
            iterator = pre_seg()
            while batch := list(islice(iterator, batch_size)):
                yield batch

        backend.train_from_iterator(batches(), trainer=trainer)
        model_data = json.loads(backend.to_str())["model"]
        hf_vocab = model_data["vocab"]
        _, char2byte = _build_maps()

        # Reserve raw-byte IDs first, then special-token IDs.
        vocab = {index: bytes([index]) for index in range(256)}
        for token in special_tokens:
            vocab[len(vocab)] = token.encode("utf-8")

        id_map = {}
        for token, old_id in sorted(hf_vocab.items(), key=lambda item: item[1]):
            token_bytes = bytes(char2byte[char] for char in token)
            if len(token_bytes) == 1:
                id_map[old_id] = token_bytes[0]
            else:
                new_id = len(vocab)
                vocab[new_id] = token_bytes
                id_map[old_id] = new_id

        # Keep native merge rank while translating all three IDs.
        merges = {
            (id_map[hf_vocab[left]], id_map[hf_vocab[right]]):
                id_map[hf_vocab[left + right]]
            for left, right in model_data["merges"]
        }
        params = TokenizerParams(vocab, merges, special_tokens)
        return RustTokenizer(params)


