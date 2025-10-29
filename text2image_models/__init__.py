"""Canonical model registry for text-to-image generation."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Type

from diffusers import DiffusionPipeline

from .base import ModelConfig
from . import animagine_xl_4, juggernautxl_v10, realvisxl_v5, sdxl


def build_model_registry(
    refiner_cls: Optional[Type[DiffusionPipeline]],
) -> Tuple[str, Dict[str, ModelConfig]]:
    """Return the default alias and model registry."""

    registry: Dict[str, ModelConfig] = {}

    def _register(aliases, config: ModelConfig) -> None:
        for alias in aliases:
            registry[alias] = config

    # Stable Diffusion XL base weights
    sdxl_config = sdxl.build_config(refiner_cls)
    _register(sdxl.ALIASES, sdxl_config)

    # RealVis XL 5.0
    realvis_config = realvisxl_v5.build_config(refiner_cls)
    _register(realvisxl_v5.ALIASES, realvis_config)

    # Juggernaut XL v10
    juggernaut_config = juggernautxl_v10.build_config(refiner_cls)
    _register(juggernautxl_v10.ALIASES, juggernaut_config)

    # Animagine XL 4.0
    animagine_config = animagine_xl_4.build_config(refiner_cls)
    _register(animagine_xl_4.ALIASES, animagine_config)

    return sdxl.DEFAULT_ALIAS, registry


__all__ = ["ModelConfig", "build_model_registry"]
