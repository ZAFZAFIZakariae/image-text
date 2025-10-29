"""Model configuration for the official Stable Diffusion XL weights."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Type

from diffusers import DiffusionPipeline, StableDiffusionXLPipeline

from .base import ModelConfig


DEFAULT_ALIAS = "stable-diffusion-xl-1.0"
ALIASES: Sequence[str] = (
    "sdxl",
    "sdxl-base",
    "stable-diffusion-xl",
    "stable-diffusion-xl-1.0",
)


def build_config(refiner_cls: Optional[Type[DiffusionPipeline]]) -> ModelConfig:
    """Return the configuration for the SDXL base checkpoint."""

    refiner_model_id: Optional[str]
    refiner_variant: Optional[str]
    if refiner_cls is None:
        refiner_model_id = None
        refiner_variant = None
    else:
        refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
        refiner_variant = "fp16"

    return ModelConfig(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        pipeline_cls=StableDiffusionXLPipeline,
        variant="fp16",
        vae_model_id="madebyollin/sdxl-vae-fp16-fix",
        vae_variant="fp16",
        refiner_model_id=refiner_model_id,
        refiner_cls=refiner_cls,
        refiner_variant=refiner_variant,
        refiner_high_noise_frac=0.8,
    )


__all__: Iterable[str] = [
    "ALIASES",
    "DEFAULT_ALIAS",
    "build_config",
]
