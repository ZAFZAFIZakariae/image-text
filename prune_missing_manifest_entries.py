"""CLI helper to drop manifest rows that point to missing image files.

This utility scans a ``metadata.jsonl`` manifest and rewrites it in place,
removing any entries whose ``file`` path cannot be resolved to an existing
image inside the dataset directory. Use it after deleting images so that your
manifest only lists files that are still present on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

from create_caption_sidecars import _resolve_image_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove entries from a metadata.jsonl manifest when their referenced "
            "images no longer exist in the dataset directory."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Directory that contains the dataset images referenced in the manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("metadata.jsonl"),
        help=(
            "Path to the metadata manifest. Defaults to 'metadata.jsonl' inside "
            "the dataset directory."
        ),
    )
    return parser.parse_args()


def _resolve_manifest_path(dataset_dir: Path, manifest: Path) -> Path:
    if manifest.is_absolute():
        return manifest

    cwd_manifest = (Path.cwd() / manifest).resolve()
    if cwd_manifest.is_file():
        return cwd_manifest

    dataset_manifest = (dataset_dir / manifest).resolve()
    if dataset_manifest.is_file():
        return dataset_manifest

    raise SystemExit(
        "Manifest file {0} does not exist. Tried {1} and {2}.".format(
            manifest, cwd_manifest, dataset_manifest
        )
    )


def prune_manifest(dataset_dir: Path, manifest_path: Path) -> Tuple[int, int, int]:
    """Rewrite ``manifest_path`` keeping only entries whose images exist.

    Returns a tuple ``(kept, removed, total)`` where ``total`` counts the number
    of non-empty records inspected.
    """

    total = kept = removed = 0

    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")

    with manifest_path.open("r", encoding="utf-8") as source, tmp_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for raw_line in source:
            line = raw_line.strip()
            if not line:
                continue

            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
                tmp_path.unlink(missing_ok=True)
                raise ValueError(
                    f"Failed to parse JSON on line {total} of {manifest_path}: {exc}"
                ) from exc

            if "file" not in record:
                tmp_path.unlink(missing_ok=True)
                raise ValueError(
                    f"Manifest entry on line {total} of {manifest_path} is missing a 'file' field."
                )

            image_path = _resolve_image_path(dataset_dir, str(record["file"]))
            if image_path.is_file():
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            else:
                removed += 1

    tmp_path.replace(manifest_path)
    return kept, removed, total


def main() -> None:
    args = parse_args()

    dataset_dir = args.dataset_dir
    if not dataset_dir.is_dir():
        raise SystemExit(
            f"Dataset directory {dataset_dir} does not exist or is not a directory."
        )

    manifest_path = _resolve_manifest_path(dataset_dir, args.manifest)

    kept, removed, total = prune_manifest(dataset_dir, manifest_path)
    print(
        (
            f"Examined {total} manifest entr{'y' if total == 1 else 'ies'}. "
            f"Removed {removed} missing entr{'y' if removed == 1 else 'ies'}. "
            f"Kept {kept} valid entr{'y' if kept == 1 else 'ies'}."
        )
    )


if __name__ == "__main__":
    main()
