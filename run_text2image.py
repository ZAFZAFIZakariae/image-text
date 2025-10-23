"""Command-line script to generate images from text prompts.

This script relies on :mod:`text2image_pipeline` to load the Stable Diffusion
pipeline and expose a convenient :func:`generate_image` function.  The script
provides a simple command-line interface for specifying the prompt and output
file.  It defaults to a sample prompt so functionality can be tested without
additional arguments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from text2image_pipeline import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL_NAME,
    generate_image,
)


DEFAULT_PROMPT = "A robot painting a portrait"
DEFAULT_OUTPUT = "output.png"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the text-to-image script."""

    parser = argparse.ArgumentParser(
        description="Generate an image from a text prompt using Stable Diffusion."
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default=DEFAULT_PROMPT,
        help="Text prompt describing the desired image.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help="Filename for saving the generated image.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help=(
            "Model alias or Hugging Face model ID to use for image generation. "
            f"Known aliases: {', '.join(AVAILABLE_MODELS)}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for text-to-image generation from the command line."""

    args = parse_args()
    prompt: str = args.prompt
    output_path = Path(args.output).expanduser()

    print(f"Prompt: {prompt}")
    model_choice = args.model
    selected_model = model_choice or DEFAULT_MODEL_NAME
    print(f"Using model: {selected_model}")

    image = generate_image(prompt, model=model_choice)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved image to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
