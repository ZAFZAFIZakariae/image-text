"""Text-to-image generation module supporting multiple diffusion backends.

This module exposes a :func:`generate_image` helper that wraps models from the
`diffusers <https://github.com/huggingface/diffusers>`_ library.  The module
now ships with built-in configurations for Stable Diffusion XL 1.0 and
Animagine XL 3.0 while still allowing callers to supply any other Hugging Face
model identifier.  Pipelines are cached per model so the heavy weights are only
loaded once per process, and the built-in safety checker is disabled to avoid
automatic censoring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

from inspect import signature

import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration describing how to construct a diffusion pipeline."""

    model_id: str
    pipeline_cls: Type[DiffusionPipeline]
    variant: Optional[str] = None


DEFAULT_MODEL_NAME = "stable-diffusion-xl-1.0"

_MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "stable-diffusion-v1-5": ModelConfig(
        model_id="runwayml/stable-diffusion-v1-5",
        pipeline_cls=StableDiffusionPipeline,
    ),
    "stable-diffusion-xl-1.0": ModelConfig(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        pipeline_cls=StableDiffusionXLPipeline,
        variant="fp16",
    ),
    "animagine-xl-3.0": ModelConfig(
        model_id="cagliostrolab/animagine-xl-3.0",
        pipeline_cls=StableDiffusionXLPipeline,
        variant="fp16",
    ),
}

AVAILABLE_MODELS = tuple(sorted(_MODEL_REGISTRY))

_PIPELINE_CACHE: Dict[str, DiffusionPipeline] = {}


def _normalise_model_name(model_name: str) -> str:
    """Normalise a model alias for dictionary lookup."""

    # Hugging Face model identifiers include `/`. Treat those as canonical IDs
    # and return them unchanged so custom checkpoints are supported.
    if "/" in model_name:
        return model_name

    normalised = model_name.strip().lower()
    # Replace spaces with dashes and drop parentheses so names like
    # "Stable Diffusion XL (1.0)" resolve to the registered alias.
    normalised = normalised.replace(" ", "-")
    normalised = re.sub(r"[()]+", "", normalised)
    return normalised


def _resolve_model(model_name: Optional[str]) -> Tuple[str, ModelConfig]:
    """Determine which model configuration should be used."""

    requested = model_name or os.getenv("TEXT2IMAGE_MODEL_ID", DEFAULT_MODEL_NAME)
    normalised = _normalise_model_name(requested)

    if normalised in _MODEL_REGISTRY:
        return normalised, _MODEL_REGISTRY[normalised]

    # Fall back to treating the provided value as a direct Hugging Face model ID.
    # Use the Stable Diffusion v1.5 pipeline class as a sensible default.
    return requested, ModelConfig(model_id=requested, pipeline_cls=StableDiffusionPipeline)


def _load_pipeline(model_name: Optional[str] = None) -> DiffusionPipeline:
    """Load (or retrieve from cache) a diffusion pipeline instance."""

    cache_key, config = _resolve_model(model_name)

    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use half-precision when running on GPU for better performance and
    # reduced memory consumption. CPU execution should remain in float32.
    torch_dtype: Optional[torch.dtype] = (
        torch.float16 if device == "cuda" else torch.float32
    )

    model_id = config.model_id

    load_kwargs = {}
    if torch_dtype is not None:
        pretrained_signature = signature(config.pipeline_cls.from_pretrained)
        if "dtype" in pretrained_signature.parameters:
            load_kwargs["dtype"] = torch_dtype
        else:
            load_kwargs["torch_dtype"] = torch_dtype

    if config.variant is not None and torch_dtype == torch.float16:
        load_kwargs.setdefault("variant", config.variant)

    try:
        pipeline = config.pipeline_cls.from_pretrained(model_id, **load_kwargs)
    except ValueError as exc:
        should_retry_without_variant = (
            "variant" in load_kwargs
            and config.variant is not None
            and "variant=" in str(exc)
        )

        if should_retry_without_variant:
            logger.warning(
                "Model %s does not provide the '%s' variant; falling back to default "
                "weights. This may increase memory usage and slightly reduce "
                "performance compared to native %s weights.",
                model_id,
                config.variant,
                config.variant,
            )
            load_kwargs.pop("variant", None)
            pipeline = config.pipeline_cls.from_pretrained(model_id, **load_kwargs)
        else:
            raise
    except OSError as exc:  # pragma: no cover - passthrough for clearer error message
        raise RuntimeError(
            "Failed to load Stable Diffusion pipeline. "
            "Check that the model ID is correct and that you have the "
            "necessary permissions. You can override the model via the "
            "TEXT2IMAGE_MODEL_ID environment variable."
        ) from exc

    # Move the pipeline to the appropriate device (GPU when available).
    pipeline = pipeline.to(device)

    # Disable the default NSFW safety checker so the pipeline returns images
    # unfiltered. Some pipelines expose ``None`` when no checker exists, hence
    # the attribute guard.
    if hasattr(pipeline, "safety_checker"):
        pipeline.safety_checker = _disable_safety_checker

    _PIPELINE_CACHE[cache_key] = pipeline
    return pipeline


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


def generate_image(prompt: str, model: Optional[str] = None) -> Image.Image:
    """Generate an image from a text prompt.

    Parameters
    ----------
    prompt:
        The text description to feed into Stable Diffusion.
    model:
        Optional alias or Hugging Face model identifier designating which
        diffusion pipeline to run. When omitted the default configured model
        (Stable Diffusion XL 1.0) is used.

    Returns
    -------
    PIL.Image.Image
        The first generated image corresponding to the input prompt.
    """

    # The pipeline returns a ``PipelineOutput`` containing a list of PIL images.
    pipeline = _load_pipeline(model)
    output = pipeline(prompt)
    return output.images[0]


if __name__ == "__main__":
    sample_prompt = "A scenic landscape with mountains"
    image = generate_image(sample_prompt)

    output_path = Path("output.png")
    image.save(output_path)
    print(f"Generated image saved to {output_path.resolve()}")

