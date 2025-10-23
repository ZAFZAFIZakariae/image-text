"""Text-to-image generation module using Stable Diffusion v1.5.

This module exposes a ``generate_image`` function that leverages the
``diffusers`` library to run the Stable Diffusion v1.5 pipeline.  The
pipeline is moved to a CUDA device when available so image synthesis can
take advantage of GPU acceleration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def _load_pipeline() -> StableDiffusionPipeline:
    """Load the Stable Diffusion v1.5 pipeline.

    Returns
    -------
    StableDiffusionPipeline
        An instance of the text-to-image diffusion pipeline configured for
        the best available compute device.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use half-precision when running on GPU for better performance and
    # reduced memory consumption. CPU execution should remain in float32.
    torch_dtype: Optional[torch.dtype] = (
        torch.float16 if device == "cuda" else torch.float32
    )

    pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-5", torch_dtype=torch_dtype
    )

    # Move the pipeline to the appropriate device (GPU when available).
    pipeline = pipeline.to(device)

    return pipeline


# Lazily load the pipeline so it is created once when the module is imported.
_PIPELINE: StableDiffusionPipeline = _load_pipeline()


def generate_image(prompt: str) -> Image.Image:
    """Generate an image from a text prompt.

    Parameters
    ----------
    prompt:
        The text description to feed into Stable Diffusion.

    Returns
    -------
    PIL.Image.Image
        The first generated image corresponding to the input prompt.
    """

    # The pipeline returns a ``PipelineOutput`` containing a list of PIL images.
    output = _PIPELINE(prompt)
    return output.images[0]


if __name__ == "__main__":
    sample_prompt = "A scenic landscape with mountains"
    image = generate_image(sample_prompt)

    output_path = Path("output.png")
    image.save(output_path)
    print(f"Generated image saved to {output_path.resolve()}")
