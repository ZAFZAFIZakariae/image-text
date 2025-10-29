"""Model configuration for Juggernaut XL v10."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Type

from diffusers import DiffusionPipeline, StableDiffusionXLPipeline

from .base import ModelConfig

ALIASES: Sequence[str] = (
    "juggernautxl",
    "juggernaut-xl",
    "juggernautxl-v10",
    "juggernaut-xl-v10",
)


def build_config(refiner_cls: Optional[Type[DiffusionPipeline]]) -> ModelConfig:
    """Return the configuration for the Juggernaut XL v10 checkpoint."""

    return ModelConfig(
        model_id="RunDiffusion/Juggernaut-XL-v10",
        pipeline_cls=StableDiffusionXLPipeline,
        refiner_cls=None,
    )


__all__: Iterable[str] = [
    "ALIASES",
    "build_config",
]
