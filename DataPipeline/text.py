import json
from pathlib import Path
from typing import Iterator, Iterable


def iter_texts(
    path: str | Path, *, text_column: str = "text"
) -> Iterator[str]:
    """Yield text records in file order, preserving whitespace and empty rows.

    Args:
        path: UTF-8 TXT/JSONL file or a Parquet file.
        text_column: Text field for Parquet and JSONL records.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        with path.open("r", encoding="utf-8", newline="") as stream:
            yield from stream

    elif suffix == ".parquet":
        from pyarrow.parquet import ParquetFile

        with ParquetFile(path) as parquet_file:
            for batch in parquet_file.iter_batches(
                batch_size=8192, columns=[text_column]
            ):
                for text in batch.column(text_column).to_pylist():
                    if text is not None:
                        yield text

    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    text = json.loads(line)[text_column]
                    if text is not None:
                        yield text
    else:
        raise ValueError(f"Unsupported corpus format: {path.suffix}")


def collect_corpus(
    texts: Iterable[str], *, max_chars: int | None = None
) -> str:
    """Collect a prefix, adding a newline only when a record lacks one. Specifically Used for tiny costumed bpe experiments.

    max_chars counts Unicode characters, including added newlines.
    None consumes the whole source. Empty records preserve blank lines;
    records are not automatically treated as separate documents.
    """
    if max_chars is not None and max_chars < 0:
        raise ValueError("max_chars must be nonnegative or None")
    if max_chars == 0:
        return ""

    parts = []
    num_chars = 0
    for text in texts:
        part = text if text.endswith("\n") else text + "\n"
        if max_chars is not None:
            part = part[: max_chars - num_chars]
        parts.append(part)
        num_chars += len(part)
        if max_chars is not None and num_chars >= max_chars:
            break
    return "".join(parts)