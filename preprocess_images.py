"""Normalise a directory of images to consistent 1024×1024 outputs."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageOps
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}


@dataclass
class ProcessingStats:
    """Counters summarising how many images were touched."""

    processed: int = 0
    skipped: int = 0
    failed: int = 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resize, crop, or pad all images in a directory so that they end up as "
            "1024×1024 squares. The script preserves the directory layout and can "
            "either operate in-place or write to a separate output directory."
        )
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory that contains the source images to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where processed images will be written. Defaults to "
            "processing the files in-place."
        ),
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=1024,
        help="Final width/height of the square output images (default: 1024).",
    )
    parser.add_argument(
        "--strategy",
        choices={"auto", "crop", "pad"},
        default="auto",
        help=(
            "How to turn non-square images into a square. 'crop' always uses a "
            "center crop, 'pad' mirrors the borders, and 'auto' crops when the "
            "aspect ratio is reasonable but falls back to padding for extremely "
            "wide or tall inputs (default: auto)."
        ),
    )
    parser.add_argument(
        "--max-crop-ratio",
        type=float,
        default=2.5,
        help=(
            "When --strategy=auto, images with a long-to-short side ratio above "
            "this threshold will be padded instead of cropped (default: 2.5)."
        ),
    )
    parser.add_argument(
        "--convert-mode",
        type=str,
        default="RGB",
        help=(
            "Optional Pillow mode to convert every image into before processing. "
            "Use 'None' (case-insensitive) to keep the original mode (default: RGB)."
        ),
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default="jpg,jpeg,png,webp,bmp",
        help=(
            "Comma-separated list of file extensions to process (default: "
            "jpg,jpeg,png,webp,bmp)."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist in the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which files would be processed without writing anything.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for troubleshooting.",
    )
    return parser.parse_args()


def _gather_images(root: Path, extensions: Iterable[str]) -> List[Path]:
    normalised_exts = {ext.lower().lstrip(".") for ext in extensions if ext}
    if not normalised_exts:
        normalised_exts = SUPPORTED_EXTENSIONS

    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower().lstrip(".") in normalised_exts
    ]


def _upscale_smallest_edge(img: Image.Image, target: int) -> Image.Image:
    width, height = img.size
    if min(width, height) >= target:
        return img

    scale = target / float(min(width, height))
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    if width <= height:
        new_width = max(new_width, target)
    if height <= width:
        new_height = max(new_height, target)

    logger.debug(
        "Upscaling image from %s to %s using bicubic resampling.",
        (width, height),
        (new_width, new_height),
    )

    return img.resize((new_width, new_height), Image.BICUBIC)


def _pad_to_square(img: Image.Image, target: int) -> Image.Image:
    width, height = img.size
    max_side = max(width, height)
    pad_left = (max_side - width) // 2
    pad_top = (max_side - height) // 2
    pad_right = max_side - width - pad_left
    pad_bottom = max_side - height - pad_top

    logger.debug(
        "Padding image %s with (left=%d, top=%d, right=%d, bottom=%d).",
        (width, height),
        pad_left,
        pad_top,
        pad_right,
        pad_bottom,
    )

    array = np.asarray(img)
    pad_width: Tuple[Tuple[int, int], ...]
    if array.ndim == 2:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
    elif array.ndim == 3:
        pad_width = (
            (pad_top, pad_bottom),
            (pad_left, pad_right),
            (0, 0),
        )
    else:
        raise ValueError(f"Unsupported image array with ndim={array.ndim}")

    if any(pad > 0 for pads in pad_width for pad in pads):
        array = np.pad(array, pad_width, mode="reflect")
    padded = Image.fromarray(array)

    if padded.size != (target, target):
        padded = padded.resize((target, target), Image.BICUBIC)

    return padded


def _crop_to_square(img: Image.Image, target: int) -> Image.Image:
    logger.debug("Cropping image %s to %dx%d square.", img.size, target, target)
    return ImageOps.fit(
        img,
        (target, target),
        method=Image.BICUBIC,
        centering=(0.5, 0.5),
    )


def _square_image(
    img: Image.Image,
    target: int,
    strategy: str,
    max_crop_ratio: float,
) -> Image.Image:
    width, height = img.size
    if width == height == target:
        return img

    aspect_ratio = max(width, height) / float(min(width, height)) if min(width, height) else float("inf")
    logger.debug("Image size %s has aspect ratio %.3f.", img.size, aspect_ratio)

    if strategy == "crop":
        return _crop_to_square(img, target)
    if strategy == "pad":
        return _pad_to_square(img, target)

    # Auto strategy
    if aspect_ratio <= max_crop_ratio:
        return _crop_to_square(img, target)
    return _pad_to_square(img, target)


def _prepare_image(img: Image.Image, convert_mode: str | None) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if convert_mode is not None:
        img = img.convert(convert_mode)
    return img


def _ensure_output_path(path: Path, image_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return path
    return output_dir / path.relative_to(image_dir)


def _save_image(img: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    ext = destination.suffix.lower()
    format_map = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".webp": "WEBP",
        ".bmp": "BMP",
    }
    fmt = format_map.get(ext, img.format)

    if fmt == "JPEG" and img.mode == "RGBA":
        img = img.convert("RGB")

    img.save(destination, format=fmt, quality=95 if fmt == "JPEG" else None)


def process_directory(args: argparse.Namespace) -> ProcessingStats:
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)
    logger.setLevel(log_level)

    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory {args.image_dir} does not exist.")
    if not args.image_dir.is_dir():
        raise NotADirectoryError(f"{args.image_dir} is not a directory.")

    convert_mode = None if str(args.convert_mode).lower() == "none" else args.convert_mode

    image_paths = _gather_images(args.image_dir, args.extensions.split(","))
    if not image_paths:
        raise ValueError("No matching images were found in the provided directory.")

    stats = ProcessingStats()

    for path in tqdm(image_paths, desc="Processing", unit="image"):
        if args.skip_existing and args.output_dir is not None:
            destination = args.output_dir / path.relative_to(args.image_dir)
            if destination.exists():
                stats.skipped += 1
                continue
        else:
            destination = _ensure_output_path(path, args.image_dir, args.output_dir)

        if args.dry_run:
            logger.info("Would process %s -> %s", path, destination)
            stats.skipped += 1
            continue

        try:
            with Image.open(path) as img:
                img = _prepare_image(img, convert_mode)
                img = _upscale_smallest_edge(img, args.target_size)
                img = _square_image(img, args.target_size, args.strategy, args.max_crop_ratio)
                _save_image(img, destination)
                stats.processed += 1
        except Exception:  # pragma: no cover - guard rail for unexpected image formats
            logger.exception("Failed to process %s", path)
            stats.failed += 1

    return stats


def main() -> None:
    args = _parse_args()
    stats = process_directory(args)
    logger.info(
        "Finished processing images. Processed=%d, Skipped=%d, Failed=%d",
        stats.processed,
        stats.skipped,
        stats.failed,
    )


if __name__ == "__main__":
    main()
