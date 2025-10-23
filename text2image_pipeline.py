"""Text-to-image generation module using Stable Diffusion v1.5.

This module exposes a ``generate_image`` function that leverages the
``diffusers`` library to run the Stable Diffusion v1.5 pipeline.  The
pipeline is moved to a CUDA device when available so image synthesis can
take advantage of GPU acceleration. The underlying model can be overridden
via the ``TEXT2IMAGE_MODEL_ID`` environment variable when a different
checkpoint is desired.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from inspect import signature

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"


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

    model_id = os.getenv("TEXT2IMAGE_MODEL_ID", DEFAULT_MODEL_ID)

    load_kwargs = {}
    if torch_dtype is not None:
        pretrained_signature = signature(StableDiffusionPipeline.from_pretrained)
        if "dtype" in pretrained_signature.parameters:
            load_kwargs["dtype"] = torch_dtype
        else:
            load_kwargs["torch_dtype"] = torch_dtype

    try:
        pipeline = StableDiffusionPipeline.from_pretrained(model_id, **load_kwargs)
    except OSError as exc:  # pragma: no cover - passthrough for clearer error message
        raise RuntimeError(
            "Failed to load Stable Diffusion pipeline. "
            "Check that the model ID is correct and that you have the "
            "necessary permissions. You can override the model via the "
            "TEXT2IMAGE_MODEL_ID environment variable."
        ) from exc

    # Move the pipeline to the appropriate device (GPU when available).
    pipeline = pipeline.to(device)

    return pipeline


# Lazily load the pipeline so it is created once when the module is imported.
_PIPELINE: StableDiffusionPipeline = _load_pipeline()
# Disable the default NSFW safety checker so the pipeline returns images unfiltered.
# The callable mirrors the expected signature and always reports that the content is safe.


def _disable_safety_checker(images, **kwargs):
    """Return the images unchanged and flag them all as safe.

    The diffusers pipelines expect the safety checker to return a tuple of
    (images, has_nsfw_concept) where ``has_nsfw_concept`` is an iterable with
    one boolean per generated image. Some versions of diffusers later iterate
    over this value, so returning a bare ``False`` would raise a ``TypeError``.
    """

    # ``images`` is typically a list of PIL images; mirror that structure when
    # reporting that every generated image passed the (disabled) safety check.
    return images, [False] * len(images)


_PIPELINE.safety_checker = _disable_safety_checker


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
