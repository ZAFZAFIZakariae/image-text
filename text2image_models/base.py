"""Shared data structures for describing text-to-image model pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Type

from diffusers import DiffusionPipeline


@dataclass(frozen=True)
class ModelConfig:
    """Configuration describing how to construct a diffusion pipeline."""

    model_id: str
    pipeline_cls: Type[DiffusionPipeline]
    variant: Optional[str] = None
    vae_model_id: Optional[str] = None
    vae_variant: Optional[str] = None
    refiner_model_id: Optional[str] = None
    refiner_cls: Optional[Type[DiffusionPipeline]] = None
    refiner_variant: Optional[str] = None
    refiner_high_noise_frac: Optional[float] = None
    scheduler_setup: Optional[Callable[[DiffusionPipeline], None]] = None
