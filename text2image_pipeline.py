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
from typing import Any, Dict, Optional, Tuple, Type

from inspect import signature

import torch
try:
    from diffusers import (
        AutoencoderKL,
        DiffusionPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
    )
except ImportError:  # pragma: no cover - diffusers must be installed
    raise

try:  # pragma: no cover - optional dependency in older diffusers versions
    from diffusers import StableDiffusionXLImg2ImgPipeline
except ImportError:  # pragma: no cover - gracefully degrade when refiner is unavailable
    StableDiffusionXLImg2ImgPipeline = None  # type: ignore[assignment]

# diffusers 0.20 added StableDiffusionXLImg2ImgPipeline, while some downstream
# forks still expose the earlier StableDiffusionXLRefinerPipeline alias. Import
# the alias when available so the rest of the module can treat both classes the
# same way.
try:  # pragma: no cover - optional dependency in older diffusers versions
    from diffusers import StableDiffusionXLRefinerPipeline  # type: ignore
except ImportError:  # pragma: no cover - silently fall back to the img2img class
    StableDiffusionXLRefinerPipeline = None  # type: ignore[assignment]
from PIL import Image


logger = logging.getLogger(__name__)

_REFINER_PIPELINE_CLS = None
if StableDiffusionXLImg2ImgPipeline is not None:
    _REFINER_PIPELINE_CLS = StableDiffusionXLImg2ImgPipeline
elif StableDiffusionXLRefinerPipeline is not None:
    _REFINER_PIPELINE_CLS = StableDiffusionXLRefinerPipeline

if _REFINER_PIPELINE_CLS is None:  # pragma: no cover - logging only
    logger.warning(
        "Stable Diffusion XL refiner support is unavailable because the "
        "img2img/refiner pipeline class could not be imported. Install "
        "diffusers>=0.20.0 (and its optional dependencies) to enable the "
        "two-stage `base+refiner` workflow."
    )

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


@dataclass
class LoadedPipelines:
    """Container holding the primary pipeline, optional refiner, and VAEs."""

    base: DiffusionPipeline
    refiner: Optional[DiffusionPipeline] = None
    base_default_vae: Optional[AutoencoderKL] = None
    custom_vae: Optional[AutoencoderKL] = None


DEFAULT_MODEL_NAME = "stable-diffusion-xl-1.0"

_REALVISXL_V4_CONFIG = ModelConfig(
    model_id="SG161222/RealVisXL_V4.0",
    pipeline_cls=StableDiffusionXLPipeline,
)

_REALVISXL_V5_CONFIG = ModelConfig(
    model_id="SG161222/RealVisXL_V5.0",
    pipeline_cls=StableDiffusionXLPipeline,
)

_JUGGERNAUTXL_V8_CONFIG = ModelConfig(
    model_id="RunDiffusion/Juggernaut-XL-v8",
    pipeline_cls=StableDiffusionXLPipeline,
)

_JUGGERNAUTXL_V10_NSFW_CONFIG = ModelConfig(
    model_id="RunDiffusion/Juggernaut-XL-v10-nsfw",
    pipeline_cls=StableDiffusionXLPipeline,
)

_MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "stable-diffusion-v1-5": ModelConfig(
        model_id="runwayml/stable-diffusion-v1-5",
        pipeline_cls=StableDiffusionPipeline,
    ),
    "stable-diffusion-xl-1.0": ModelConfig(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        pipeline_cls=StableDiffusionXLPipeline,
        variant="fp16",
        vae_model_id="madebyollin/sdxl-vae-fp16-fix",
        vae_variant="fp16",
        refiner_model_id=(
            "stabilityai/stable-diffusion-xl-refiner-1.0"
            if _REFINER_PIPELINE_CLS is not None
            else None
        ),
        refiner_cls=_REFINER_PIPELINE_CLS,
        refiner_variant="fp16" if _REFINER_PIPELINE_CLS is not None else None,
        refiner_high_noise_frac=0.8,
    ),
    "animagine-xl-3.0": ModelConfig(
        model_id="cagliostrolab/animagine-xl-3.0",
        pipeline_cls=StableDiffusionXLPipeline,
        variant="fp16",
    ),
    "realvisxl": _REALVISXL_V4_CONFIG,
    "realvisxl-v4": _REALVISXL_V4_CONFIG,
    "realvisxl-v4.0": _REALVISXL_V4_CONFIG,
    "realvis-xl": _REALVISXL_V4_CONFIG,
    "realvis-xl-4": _REALVISXL_V4_CONFIG,
    "realvis-xl-4.0": _REALVISXL_V4_CONFIG,
    "realvisxl-v5": _REALVISXL_V5_CONFIG,
    "realvisxl-v5.0": _REALVISXL_V5_CONFIG,
    "realvis-xl-5": _REALVISXL_V5_CONFIG,
    "realvis-xl-5.0": _REALVISXL_V5_CONFIG,
    "juggernautxl": _JUGGERNAUTXL_V8_CONFIG,
    "juggernaut-xl": _JUGGERNAUTXL_V8_CONFIG,
    "juggernautxl-v8": _JUGGERNAUTXL_V8_CONFIG,
    "juggernaut-xl-v8": _JUGGERNAUTXL_V8_CONFIG,
    "juggernautxl-v10-nsfw": _JUGGERNAUTXL_V10_NSFW_CONFIG,
    "juggernaut-xl-v10-nsfw": _JUGGERNAUTXL_V10_NSFW_CONFIG,
}

AVAILABLE_MODELS = tuple(sorted(_MODEL_REGISTRY))

_PIPELINE_CACHE: Dict[str, LoadedPipelines] = {}
_VAE_CACHE: Dict[Tuple[str, Optional[str], str, str], AutoencoderKL] = {}


def _load_vae(
    model_id: str,
    variant: Optional[str],
    torch_dtype: Optional[torch.dtype],
    device: str,
) -> AutoencoderKL:
    """Load (or retrieve from cache) an AutoencoderKL VAE."""

    dtype_token = (
        str(torch_dtype).split(".")[-1]
        if torch_dtype is not None
        else "default"
    )
    cache_key = (model_id, variant, dtype_token, device)

    if cache_key in _VAE_CACHE:
        return _VAE_CACHE[cache_key]

    load_kwargs = {}
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype

    if variant is not None and torch_dtype == torch.float16:
        load_kwargs["variant"] = variant

    try:
        vae = AutoencoderKL.from_pretrained(model_id, **load_kwargs)
    except ValueError as exc:
        should_retry_without_variant = (
            "variant" in load_kwargs
            and variant is not None
            and "variant=" in str(exc)
        )

        if should_retry_without_variant:
            logger.warning(
                "VAE model %s does not provide the '%s' variant; falling back to "
                "default weights. This may increase memory usage and slightly "
                "reduce performance compared to native %s weights.",
                model_id,
                variant,
                variant,
            )
            load_kwargs.pop("variant", None)
            vae = AutoencoderKL.from_pretrained(model_id, **load_kwargs)
        else:
            raise

    vae = vae.to(device)
    _VAE_CACHE[cache_key] = vae
    return vae


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

    base_default_vae: Optional[AutoencoderKL]
    base_default_vae = getattr(base_pipeline, "vae", None)

    custom_vae: Optional[AutoencoderKL] = None
    if config.vae_model_id:
        try:
            custom_vae = _load_vae(
                config.vae_model_id,
                config.vae_variant,
                torch_dtype,
                device,
            )
        except OSError as exc:  # pragma: no cover - passthrough for clearer error message
            raise RuntimeError(
                "Failed to load the configured VAE weights. Check that the model ID "
                "is correct and that you have the necessary permissions."
            ) from exc

    refiner_pipeline: Optional[DiffusionPipeline] = None
    if config.refiner_model_id and config.refiner_cls:
        refiner_pipeline = _instantiate_pipeline(
            config.refiner_model_id,
            config.refiner_cls,
            config.refiner_variant,
            stage_label="refiner",
        )

    loaded = LoadedPipelines(
        base=base_pipeline,
        refiner=refiner_pipeline,
        base_default_vae=base_default_vae,
        custom_vae=custom_vae,
    )
    _PIPELINE_CACHE[cache_key] = loaded
    return config, loaded


def _configure_pipeline_vae(pipelines: LoadedPipelines, use_custom_vae: Optional[bool]) -> None:
    """Attach the requested VAE to the cached base pipeline."""

    base_pipeline = pipelines.base
    target_vae: Optional[AutoencoderKL]

    if use_custom_vae is True:
        if pipelines.custom_vae is None:
            raise ValueError(
                "A custom VAE was requested, but the selected model does not "
                "provide one."
            )
        target_vae = pipelines.custom_vae
    elif use_custom_vae is False:
        target_vae = pipelines.base_default_vae
    else:
        target_vae = pipelines.custom_vae or pipelines.base_default_vae

    if target_vae is None:
        return

    if getattr(base_pipeline, "vae", None) is not target_vae:
        base_pipeline.vae = target_vae


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


_WORKFLOW_AUTO = "auto"
_WORKFLOW_BASE_ONLY = "base-only"
_WORKFLOW_BASE_REFINER = "base+refiner"
_VALID_WORKFLOWS = {
    _WORKFLOW_AUTO,
    _WORKFLOW_BASE_ONLY,
    _WORKFLOW_BASE_REFINER,
}

WORKFLOW_CHOICES = tuple(sorted(_VALID_WORKFLOWS))


def _filter_kwargs_for_pipeline(
    pipeline: DiffusionPipeline, extra_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Return only the keyword arguments accepted by ``pipeline``."""

    call_signature = signature(pipeline.__call__)
    has_var_kwargs = any(
        parameter.kind == parameter.VAR_KEYWORD
        for parameter in call_signature.parameters.values()
    )

    if has_var_kwargs:
        return dict(extra_kwargs)

    valid_params = set(call_signature.parameters)
    return {k: v for k, v in extra_kwargs.items() if k in valid_params}


def generate_image(
    prompt: str,
    model: Optional[str] = None,
    workflow: Optional[str] = None,
    use_custom_vae: Optional[bool] = None,
    refiner_start: Optional[float] = None,
    **pipeline_kwargs: Any,
) -> Image.Image:
    """Generate an image from a text prompt.

    Parameters
    ----------
    prompt:
        The text description to feed into Stable Diffusion.
    model:
        Optional alias or Hugging Face model identifier designating which
        diffusion pipeline to run. When omitted the default configured model
        (Stable Diffusion XL 1.0) is used.

    workflow:
        Controls which parts of the diffusion pipeline should execute. The
        supported values are ``"auto"`` (default behaviour), ``"base-only"``
        to skip any loaded refiner, and ``"base+refiner"`` to require a
        two-pass SDXL run. The comparison is case-insensitive.

    use_custom_vae:
        When ``True`` the loader forces attachment of any configured external
        VAE (such as the SDXL FP16 fix). When ``False`` the pipeline reverts to
        its default autoencoder. ``None`` (default) chooses the external VAE
        when available and otherwise keeps the pipeline default.

    refiner_start:
        Optional fraction in the ``[0, 1]`` range indicating when the SDXL
        refiner should take over the denoising process. When not provided, the
        value defined by the model configuration is used (default: ``0.8`` for
        the bundled SDXL weights).

    **pipeline_kwargs:
        Additional keyword arguments forwarded to the underlying diffusers
        pipelines. This exposes the full set of Stable Diffusion parameters,
        such as ``negative_prompt``, ``guidance_scale``, ``num_inference_steps``,
        ``width``/``height``, callbacks, or custom generators.

    Returns
    -------
    PIL.Image.Image
        The first generated image corresponding to the input prompt.
    """

    # The pipeline returns a ``PipelineOutput`` containing a list of PIL images.
    config, pipelines = _load_pipeline(model)

    _configure_pipeline_vae(pipelines, use_custom_vae)

    base_pipeline = pipelines.base
    refiner_pipeline = pipelines.refiner

    selected_workflow = (workflow or _WORKFLOW_AUTO).strip().lower()

    if selected_workflow not in _VALID_WORKFLOWS:
        raise ValueError(
            "Invalid workflow requested. Expected one of "
            f"{sorted(_VALID_WORKFLOWS)}, got {workflow!r}."
        )

    if selected_workflow == _WORKFLOW_BASE_REFINER and refiner_pipeline is None:
        raise ValueError(
            "The requested base+refiner workflow requires a refiner pipeline, "
            "but the selected model does not provide one. "
            "Install diffusers>=0.20.0 (with accelerate and safetensors) and "
            "ensure the model includes `stabilityai/stable-diffusion-xl-refiner-1.0`."
        )

    should_use_refiner = (
        selected_workflow == _WORKFLOW_BASE_REFINER
        or (selected_workflow == _WORKFLOW_AUTO and refiner_pipeline is not None)
    )

    extra_kwargs: Dict[str, Any] = dict(pipeline_kwargs)
    if "prompt" in extra_kwargs:
        logger.warning(
            "Ignoring 'prompt' specified via pipeline keyword arguments; the "
            "value provided to generate_image() is always used instead."
        )
        extra_kwargs.pop("prompt", None)

    if not should_use_refiner:
        call_kwargs = _filter_kwargs_for_pipeline(base_pipeline, extra_kwargs)
        output = base_pipeline(prompt=prompt, **call_kwargs)
        return output.images[0]

    # When a refiner is available, run a two-stage SDXL workflow. The base
    # pipeline handles the majority of denoising and produces a latent image
    # that the refiner then sharpens and enhances.
    high_noise_frac = (
        refiner_start
        if refiner_start is not None
        else config.refiner_high_noise_frac or 0.8
    )

    base_kwargs = _filter_kwargs_for_pipeline(base_pipeline, extra_kwargs)
    base_kwargs.pop("output_type", None)
    base_denoising_end = base_kwargs.pop("denoising_end", high_noise_frac)

    base_output = base_pipeline(
        prompt=prompt,
        output_type="latent",
        denoising_end=base_denoising_end,
        **base_kwargs,
    )

    assert refiner_pipeline is not None  # Narrow type for static checkers
    refiner_kwargs = _filter_kwargs_for_pipeline(refiner_pipeline, extra_kwargs)
    refiner_kwargs.pop("image", None)
    refiner_denoising_start = refiner_kwargs.pop(
        "denoising_start", base_denoising_end
    )

    refined_output = refiner_pipeline(
        prompt=prompt,
        image=base_output.images,
        denoising_start=refiner_denoising_start,
        **refiner_kwargs,
    )
    return refined_output.images[0]


if __name__ == "__main__":
    sample_prompt = "A scenic landscape with mountains"
    image = generate_image(sample_prompt)

    output_path = Path("output.png")
    image.save(output_path)
    print(f"Generated image saved to {output_path.resolve()}")

