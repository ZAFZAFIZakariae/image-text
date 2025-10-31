"""Utility to export captions from a metadata manifest into per-image text files.

This helper is meant for workflows that require a folder layout where every
image has a sidecar caption file (for example the Kohya-based RealVis trainers).

Given a ``metadata.jsonl`` manifest produced by ``prepare_text_image_dataset.py``
or any file containing ``{"file": ..., "text": ...}`` JSONL entries, the script
writes a ``.txt`` (or custom extension) file next to each referenced image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-image caption files next to the dataset images using "
            "entries from a metadata manifest."
        )
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Root directory that contains the dataset images referenced in the manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("metadata.jsonl"),
        help=(
            "Path to the JSONL manifest. Defaults to 'metadata.jsonl' inside "
            "the image directory."
        ),
    )
    parser.add_argument(
        "--caption-extension",
        default=".txt",
        help=(
            "File extension to use for generated caption files (default: .txt). "
            "The leading dot is optional."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing caption files instead of leaving them untouched.",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help=(
            "Silently skip manifest entries whose images are missing instead of "
            "aborting with an error."
        ),
    )
    return parser.parse_args()


def _normalize_extension(extension: str) -> str:
    extension = extension.strip()
    if not extension:
        raise ValueError("Caption extension cannot be empty.")
    if not extension.startswith("."):
        extension = f".{extension}"
    return extension


def _iter_manifest_entries(manifest_path: Path) -> Iterable[Tuple[str, str]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
                raise ValueError(
                    f"Failed to parse JSON on line {line_number} of {manifest_path}: {exc}"
                ) from exc
            if "file" not in record or "text" not in record:
                raise ValueError(
                    f"Manifest entry on line {line_number} must contain 'file' and 'text' fields."
                )
            yield str(record["file"]), str(record["text"])


def main() -> None:
    args = parse_args()

    image_dir = args.image_dir
    if not image_dir.is_dir():
        raise SystemExit(f"Image directory {image_dir} does not exist or is not a directory.")

    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = image_dir / manifest_path
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest file {manifest_path} does not exist.")

    caption_extension = _normalize_extension(args.caption_extension)

    created = 0
    skipped_existing = 0
    missing_images = 0

    for relative_image, caption in _iter_manifest_entries(manifest_path):
        image_path = image_dir / relative_image
        if not image_path.is_file():
            if args.skip_missing_images:
                missing_images += 1
                continue
            raise SystemExit(
                f"Image referenced in manifest does not exist: {image_path}"
            )

        caption_path = image_path.with_suffix(caption_extension)
        if caption_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        caption_path.parent.mkdir(parents=True, exist_ok=True)
        caption_path.write_text(caption + "\n", encoding="utf-8")
        created += 1

    summary = (
        f"Created {created} caption file{'s' if created != 1 else ''}."
        f" Skipped {skipped_existing} existing file{'s' if skipped_existing != 1 else ''}."
    )
    if args.skip_missing_images and missing_images:
        summary += f" {missing_images} missing image entry{'ies' if missing_images != 1 else ''} ignored."

    output_location = image_dir.resolve()
    summary += f" Caption files written alongside images under {output_location}."

    print(summary)


if __name__ == "__main__":
    main()
