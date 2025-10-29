"""Model configuration for Animagine XL 4.0."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Type

from diffusers import DiffusionPipeline, StableDiffusionXLPipeline

from .base import ModelConfig

ALIASES: Sequence[str] = (
    "animagine-xl",
    "animagine-xl-4",
    "animagine-xl-4.0",
)


def build_config(refiner_cls: Optional[Type[DiffusionPipeline]]) -> ModelConfig:
    """Return the configuration for the Animagine XL 4.0 checkpoint."""

    return ModelConfig(
        model_id="cagliostrolab/animagine-xl-4.0",
        pipeline_cls=StableDiffusionXLPipeline,
        variant="fp16",
    )


__all__: Iterable[str] = [
    "ALIASES",
    "build_config",
]
