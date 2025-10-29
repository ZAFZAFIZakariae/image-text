"""Model configuration for the RealVis XL 5.0 checkpoint."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Type

from diffusers import DiffusionPipeline, StableDiffusionXLPipeline

from .base import ModelConfig

ALIASES: Sequence[str] = (
    "realvisxl",
    "realvisxl-v5",
    "realvisxl-v5.0",
    "realvis-xl",
    "realvis-xl-5",
    "realvis-xl-5.0",
)


def build_config(refiner_cls: Optional[Type[DiffusionPipeline]]) -> ModelConfig:
    """Return the configuration for the RealVis XL 5.0 checkpoint."""

    # RealVis XL ships with a fused refiner; skip SDXL's second-stage weights.
    _ = refiner_cls  # Unused – kept for a consistent function signature.

    return ModelConfig(
        model_id="SG161222/RealVisXL_V5.0",
        pipeline_cls=StableDiffusionXLPipeline,
        vae_model_id="SG161222/RealVisXL_VAE",
    )


__all__: Iterable[str] = [
    "ALIASES",
    "build_config",
]
