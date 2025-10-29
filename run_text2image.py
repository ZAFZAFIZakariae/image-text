"""Command-line script to generate images from text prompts.

This script relies on :mod:`text2image_pipeline` to load the Stable Diffusion
pipeline and expose a convenient :func:`generate_image` function.  The script
provides a simple command-line interface for specifying the prompt and output
file.  It defaults to a sample prompt so functionality can be tested without
additional arguments.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
from text2image_pipeline import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL_NAME,
    WORKFLOW_CHOICES,
    generate_image,
)


DEFAULT_PROMPT = "A robot painting a portrait"
DEFAULT_OUTPUT = "output.png"


def _parse_pipeline_arg(text: str) -> Tuple[str, Any]:
    """Parse ``KEY=VALUE`` CLI pairs for forwarding to the pipeline."""

    if "=" not in text:
        raise argparse.ArgumentTypeError(
            f"Invalid pipeline argument {text!r}. Expected the form KEY=VALUE."
        )

    key, raw_value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Pipeline argument keys must be non-empty.")

    raw_value = raw_value.strip()
    if not raw_value:
        raise argparse.ArgumentTypeError(
            f"Pipeline argument {text!r} is missing a value after '='."
        )

    try:
        value = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        value = raw_value

    return key, value


def _build_pipeline_kwargs(
    pairs: Iterable[Tuple[str, Any]],
) -> Dict[str, Any]:
    """Convert parsed ``(key, value)`` pairs into a kwargs dictionary."""

    kwargs: Dict[str, Any] = {}
    for key, value in pairs:
        if key in kwargs:
            existing = kwargs[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                kwargs[key] = [existing, value]
        else:
            kwargs[key] = value
    return kwargs


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
            "Model alias to use for image generation. Supported aliases: "
            f"{', '.join(AVAILABLE_MODELS)}"
        ),
    )
    parser.add_argument(
        "--workflow",
        "-w",
        default="auto",
        choices=WORKFLOW_CHOICES,
        help=(
            "Diffusion workflow to execute. 'auto' runs the refiner when the "
            "selected model provides one, 'base-only' skips the refiner, and "
            "'base+refiner' requires the two-stage SDXL pass."
        ),
    )
    parser.set_defaults(use_custom_vae=None)
    parser.add_argument(
        "--use-custom-vae",
        dest="use_custom_vae",
        action="store_true",
        help=(
            "Force the pipeline to attach any configured external VAE. This "
            "enables the full SDXL base+refiner+VAE stack when available."
        ),
    )
    parser.add_argument(
        "--no-custom-vae",
        dest="use_custom_vae",
        action="store_false",
        help=(
            "Disable the configured external VAE and use the pipeline's "
            "default autoencoder."
        ),
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Optional negative prompt to steer the denoising process.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        dest="num_inference_steps",
        default=None,
        help="Total number of denoising steps for the diffusion process.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale (CFG).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Requested image width for generation.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Requested image height for generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic generation.",
    )
    parser.add_argument(
        "--refiner-start",
        type=float,
        default=None,
        help=(
            "Fraction of steps after which the SDXL refiner should take over. "
            "Defaults to the model configuration (0.8 for bundled SDXL)."
        ),
    )
    parser.add_argument(
        "--pipeline-arg",
        action="append",
        default=[],
        type=_parse_pipeline_arg,
        metavar="KEY=VALUE",
        help=(
            "Additional keyword argument to forward to the diffusion pipeline. "
            "Use multiple times for several parameters (e.g. --pipeline-arg "
            "eta=0.0 --pipeline-arg clip_skip=2). Values are parsed with "
            "ast.literal_eval when possible."
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

    pipeline_kwargs = _build_pipeline_kwargs(args.pipeline_arg)

    if args.negative_prompt is not None:
        pipeline_kwargs.setdefault("negative_prompt", args.negative_prompt)
    if args.num_inference_steps is not None:
        pipeline_kwargs.setdefault("num_inference_steps", args.num_inference_steps)
    if args.guidance_scale is not None:
        pipeline_kwargs.setdefault("guidance_scale", args.guidance_scale)
    if args.width is not None:
        pipeline_kwargs.setdefault("width", args.width)
    if args.height is not None:
        pipeline_kwargs.setdefault("height", args.height)

    if args.seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        pipeline_kwargs.setdefault("generator", generator)

    image = generate_image(
        prompt,
        model=model_choice,
        workflow=args.workflow,
        use_custom_vae=args.use_custom_vae,
        refiner_start=args.refiner_start,
        **pipeline_kwargs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved image to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
