"""Utility for turning an image directory into a text-image dataset manifest.

The script walks a directory tree, generates captions for each image using the
BLIP helper in :mod:`image_caption`, and writes a ``metadata.jsonl`` file that
``fine_tune_text2image.py`` can consume.  Existing manifests are reused unless
``--refresh`` is supplied, which is handy when iterating on captions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from tqdm.auto import tqdm

from image_caption import caption_image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Caption images and create a metadata.jsonl manifest for training."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory that contains the raw training images.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Where to write the manifest. Defaults to IMAGE-DIR/metadata.jsonl. "
            "Each line will contain {'file': relative_path, 'text': caption}."
        ),
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default="jpg,jpeg,png,webp",
        help="Comma-separated list of image extensions to include (default: jpg,jpeg,png,webp).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore existing captions and regenerate everything from scratch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of images to caption (useful for smoke tests).",
    )

    return parser.parse_args()


def _gather_images(image_dir: Path, extensions: Iterable[str]) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory {image_dir} does not exist.")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"{image_dir} is not a directory.")

    normalized_exts = {ext.lower().lstrip(".") for ext in extensions if ext}
    paths = [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower().lstrip(".") in normalized_exts
    ]

    if not paths:
        raise ValueError(
            "No images were found. Double-check the directory and extensions provided."
        )

    return paths


def _load_existing_manifest(manifest_path: Path) -> Dict[str, str]:
    if not manifest_path.exists():
        return {}

    captions: Dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            captions[str(record["file"])] = str(record["text"])
    return captions


def _write_manifest(manifest_path: Path, captions: Dict[str, str]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for image_key in sorted(captions.keys()):
            json.dump({"file": image_key, "text": captions[image_key]}, handle, ensure_ascii=False)
            handle.write("\n")


def main() -> None:
    args = _parse_args()

    output_path = args.output_path or (args.image_dir / "metadata.jsonl")

    existing = {} if args.refresh else _load_existing_manifest(output_path)

    image_paths = _gather_images(args.image_dir, args.extensions.split(","))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    captions: Dict[str, str] = {}
    skipped = 0
    for path in tqdm(image_paths, desc="Captioning", unit="image"):
        relative_key = path.relative_to(args.image_dir).as_posix()
        if not args.refresh and relative_key in existing:
            captions[relative_key] = existing[relative_key]
            skipped += 1
            continue

        caption = caption_image(str(path))
        captions[relative_key] = caption

    _write_manifest(output_path, captions)

    print(
        f"Wrote {len(captions)} captions to {output_path}."
        + (f" Reused {skipped} existing entries." if skipped else "")
    )


if __name__ == "__main__":
    main()
