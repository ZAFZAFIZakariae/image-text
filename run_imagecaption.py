"""Command-line helper for generating image captions with BLIP.

This script relies on :mod:`image_caption` for the heavy lifting.  It accepts
an image reference (either a filesystem path or an HTTP(S) URL) and prints the
caption produced by :func:`image_caption.caption_image`.  When no argument is
provided, the script falls back to a sample image hosted on Hugging Face so it
can be exercised without additional setup.
"""

from __future__ import annotations

import argparse

from image_caption import caption_image


DEFAULT_IMAGE = (
    "https://huggingface.co/datasets/Narsil/image_dummy/resolve/main/parrots.png"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the image captioning helper."""

    parser = argparse.ArgumentParser(
        description="Generate a caption for a local image or a URL using BLIP."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=DEFAULT_IMAGE,
        help=(
            "Path or URL to the image to caption. "
            "Defaults to a demo image hosted by Hugging Face."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the command-line image captioning script."""

    args = parse_args()
    image_reference: str = args.image

    print(f"Captioning image: {image_reference}")
    caption = caption_image(image_reference)
    print(f"Caption: {caption}")


if __name__ == "__main__":
    main()
