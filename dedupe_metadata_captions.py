"""Utilities for removing repeated sentences from JSONL metadata captions.

This script reads a JSON Lines (JSONL) file—commonly used for datasets such as
DiffusionDB or other image caption corpora—and normalises a specific text field
by dropping duplicate sentences while preserving their original order.

Example usage::

    python dedupe_metadata_captions.py metadata.jsonl \
        --output metadata_deduped.jsonl --field text

The default field is ``text`` to match the common schema produced by
``create_caption_sidecars.py`` and similar scripts.  Pass ``--field test`` if
your metadata uses the typo mentioned in the support question.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


SENTENCE_PATTERN = re.compile(r"[^.!?\s][^.!?]*[.!?]+|[^.!?]+$", re.MULTILINE)


def iter_sentences(text: str) -> Iterable[str]:
    """Yield sentences from ``text`` while preserving trailing punctuation."""

    for match in SENTENCE_PATTERN.finditer(text):
        sentence = match.group(0).strip()
        if sentence:
            yield sentence


def dedupe_sentences(text: str) -> str:
    """Remove repeated sentences from ``text``.

    Deduplication is case-insensitive and treats runs of whitespace as a single
    space so that superficial formatting differences do not prevent matches.
    """

    seen: set[str] = set()
    unique: list[str] = []

    for sentence in iter_sentences(text):
        normalised = re.sub(r"\s+", " ", sentence).strip().lower()
        if normalised and normalised not in seen:
            seen.add(normalised)
            unique.append(sentence.strip())

    # Join sentences with a space while respecting the original punctuation.
    return " ".join(unique)


def clean_record(record: dict[str, Any], field: str) -> dict[str, Any]:
    """Return ``record`` with the specified ``field`` deduplicated if present."""

    cleaned = dict(record)

    if field in cleaned and isinstance(cleaned[field], str):
        cleaned[field] = dedupe_sentences(cleaned[field])
    return cleaned


def process_file(input_path: Path, output_path: Path, field: str) -> None:
    """Read ``input_path`` JSONL, dedupe the chosen field, and write ``output_path``."""

    with input_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number} of {input_path}: {exc}"
                ) from exc

            cleaned = clean_record(record, field)
            outfile.write(json.dumps(cleaned, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove repeated sentences from a caption field in metadata.jsonl"
    )
    parser.add_argument("input", type=Path, help="Path to the source metadata.jsonl file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination for the processed JSONL (defaults to <input>.deduped)",
    )
    parser.add_argument(
        "--field",
        default="text",
        help="Name of the JSON field containing captions (default: text)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input
    output_path: Path = args.output or input_path.with_suffix(input_path.suffix + ".deduped")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    process_file(input_path, output_path, args.field)


if __name__ == "__main__":
    main()
