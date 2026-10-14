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

def _build_maps():

    bs = (
        list(range(ord("!"), ord("~") + 1))    # 33 ~ 126
        + list(range(ord("¡"), ord("¬") + 1))  # 161 ~ 172
        + list(range(ord("®"), ord("ÿ") + 1))  # 174 ~ 255
    )
    cs = bs.copy()


    offset = 0
    for byte_val in range(256):
        if byte_val not in bs:
            bs.append(byte_val)
            cs.append(256 + offset)
            offset += 1

    byte2char = {byte_val: chr(code_point) for byte_val, code_point in zip(bs, cs)}
    char2byte = {chr(code_point): byte_val for byte_val, code_point in zip(bs, cs)}
    return byte2char, char2byte

if __name__ == "__main__":
    print(list(_pretokenize("hello world")))

