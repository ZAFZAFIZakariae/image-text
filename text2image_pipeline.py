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
    StableDiffusionXLRefinerPipeline,
)
from PIL import Image


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration describing how to construct a diffusion pipeline."""

    model_id: str
    pipeline_cls: Type[DiffusionPipeline]
    variant: Optional[str] = None
    refiner_model_id: Optional[str] = None
    refiner_cls: Optional[Type[DiffusionPipeline]] = None
    refiner_variant: Optional[str] = None
    refiner_high_noise_frac: Optional[float] = None


@dataclass
class LoadedPipelines:
    """Container holding the primary pipeline and optional refiner."""

    base: DiffusionPipeline
    refiner: Optional[DiffusionPipeline] = None


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
        refiner_model_id="stabilityai/stable-diffusion-xl-refiner-1.0",
        refiner_cls=StableDiffusionXLRefinerPipeline,
        refiner_variant="fp16",
        refiner_high_noise_frac=0.8,
    ),
    "animagine-xl-3.0": ModelConfig(
        model_id="cagliostrolab/animagine-xl-3.0",
        pipeline_cls=StableDiffusionXLPipeline,
        variant="fp16",
    ),
}

AVAILABLE_MODELS = tuple(sorted(_MODEL_REGISTRY))

_PIPELINE_CACHE: Dict[str, LoadedPipelines] = {}


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


def _load_pipeline(model_name: Optional[str] = None) -> Tuple[ModelConfig, LoadedPipelines]:
    """Load (or retrieve from cache) the pipelines for a given model."""

    cache_key, config = _resolve_model(model_name)

    if cache_key in _PIPELINE_CACHE:
        return config, _PIPELINE_CACHE[cache_key]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use half-precision when running on GPU for better performance and
    # reduced memory consumption. CPU execution should remain in float32.
    torch_dtype: Optional[torch.dtype] = (
        torch.float16 if device == "cuda" else torch.float32
    )

    def _instantiate_pipeline(
        model_id: str,
        pipeline_cls: Type[DiffusionPipeline],
        variant: Optional[str],
        stage_label: str,
    ) -> DiffusionPipeline:
        load_kwargs = {}
        if torch_dtype is not None:
            pretrained_signature = signature(pipeline_cls.from_pretrained)
            if "dtype" in pretrained_signature.parameters:
                load_kwargs["dtype"] = torch_dtype
            else:
                load_kwargs["torch_dtype"] = torch_dtype

        if variant is not None and torch_dtype == torch.float16:
            load_kwargs.setdefault("variant", variant)

        try:
            pipeline = pipeline_cls.from_pretrained(model_id, **load_kwargs)
        except ValueError as exc:
            should_retry_without_variant = (
                "variant" in load_kwargs
                and variant is not None
                and "variant=" in str(exc)
            )

            if should_retry_without_variant:
                logger.warning(
                    "%s model %s does not provide the '%s' variant; falling back to "
                    "default weights. This may increase memory usage and slightly "
                    "reduce performance compared to native %s weights.",
                    stage_label.capitalize(),
                    model_id,
                    variant,
                    variant,
                )
                load_kwargs.pop("variant", None)
                pipeline = pipeline_cls.from_pretrained(model_id, **load_kwargs)
            else:
                raise
        except OSError as exc:  # pragma: no cover - passthrough for clearer error message
            raise RuntimeError(
                "Failed to load Stable Diffusion pipeline. "
                "Check that the model ID is correct and that you have the "
                "necessary permissions. You can override the model via the "
                "TEXT2IMAGE_MODEL_ID environment variable."
            ) from exc

        pipeline = pipeline.to(device)

        if hasattr(pipeline, "safety_checker"):
            pipeline.safety_checker = _disable_safety_checker

        return pipeline

    base_pipeline = _instantiate_pipeline(
        config.model_id, config.pipeline_cls, config.variant, stage_label="base"
    )

    refiner_pipeline: Optional[DiffusionPipeline] = None
    if config.refiner_model_id and config.refiner_cls:
        refiner_pipeline = _instantiate_pipeline(
            config.refiner_model_id,
            config.refiner_cls,
            config.refiner_variant,
            stage_label="refiner",
        )

    loaded = LoadedPipelines(base=base_pipeline, refiner=refiner_pipeline)
    _PIPELINE_CACHE[cache_key] = loaded
    return config, loaded


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
    config, pipelines = _load_pipeline(model)

    base_pipeline = pipelines.base
    refiner_pipeline = pipelines.refiner

    if refiner_pipeline is None:
        output = base_pipeline(prompt)
        return output.images[0]

    # When a refiner is available, run a two-stage SDXL workflow. The base
    # pipeline handles the majority of denoising and produces a latent image
    # that the refiner then sharpens and enhances.
    high_noise_frac = config.refiner_high_noise_frac or 0.8

    base_output = base_pipeline(
        prompt=prompt,
        output_type="latent",
        denoising_end=high_noise_frac,
    )
    refined_output = refiner_pipeline(
        prompt=prompt,
        image=base_output.images,
        denoising_start=high_noise_frac,
    )
    return refined_output.images[0]


if __name__ == "__main__":
    sample_prompt = "A scenic landscape with mountains"
    image = generate_image(sample_prompt)

    output_path = Path("output.png")
    image.save(output_path)
    print(f"Generated image saved to {output_path.resolve()}")

