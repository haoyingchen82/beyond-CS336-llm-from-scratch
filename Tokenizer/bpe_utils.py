from collections.abc import Iterable
import regex

PRETOKEN_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)"
    r"| ?\p{L}+"
    r"| ?\p{N}+"
    r"| ?[^\s\p{L}\p{N}]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

def merge(indices: tuple[int, ...], pair: tuple[int, int], new_index: int) -> tuple[int, ...]:
    new_indices = []
    i = 0
    while i < len(indices):
        if i+1 < len(indices) and indices[i] == pair[0] and indices[i+1]==pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1

    return tuple(new_indices)

def _pretokenize(string: str) -> Iterable[str]:
    _PRETOKEN_REGEX = regex.compile(PRETOKEN_PATTERN)

    return (
        match.group(0)
        for match in _PRETOKEN_REGEX.finditer(string)
    )

if __name__ == "__main__":
    print(list(_pretokenize("hello world")))

