"""Runtime patches for deterministic and resumable PyTorch DataLoaders."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Optional


def _load_state() -> Optional[dict[str, Any]]:
    state_path = os.environ.get("IMAGETEXT_DATA_LOADER_STATE")
    if not state_path:
        return None

    path = Path(state_path)
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _seed_global_rngs(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - numpy is optional at runtime
        np = None
    else:
        np.random.seed(seed)

    try:
        import torch
    except Exception:  # pragma: no cover - torch is optional in unit tests
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CUDA is not available in tests
        torch.cuda.manual_seed_all(seed)


def _patch_random_sampler(seed: Optional[int], skip_once: int) -> None:
    try:
        import torch
        from torch.utils.data import RandomSampler
    except Exception:  # pragma: no cover - torch missing during tests
        return

    if seed is not None:
        # ``RandomSampler`` uses its own generator. Seeding it keeps the shuffle order stable.
        def _seed_sampler_generator(sampler: RandomSampler) -> None:
            if hasattr(sampler, "generator") and sampler.generator is not None:
                sampler.generator.manual_seed(int(seed))

        seed_sampler = _seed_sampler_generator
    else:
        def _noop_seed(_: RandomSampler) -> None:
            return None

        seed_sampler = _noop_seed

    original_iter = RandomSampler.__iter__
    skip_box = [max(0, int(skip_once))]

    def iterator_with_resume(self: RandomSampler):  # type: ignore[override]
        seed_sampler(self)
        iterator = original_iter(self)
        if skip_box[0] > 0:
            remaining = skip_box[0]
            for _ in range(remaining):
                try:
                    next(iterator)
                except StopIteration:
                    skip_box[0] = 0
                    return
            skip_box[0] = 0
        for index in iterator:
            yield index

    RandomSampler.__iter__ = iterator_with_resume  # type: ignore[assignment]


def _enable_truncated_images() -> None:
    if not os.environ.get("IMAGETEXT_ALLOW_TRUNCATED_IMAGES"):
        return

    try:
        from PIL import ImageFile  # type: ignore
    except Exception:  # pragma: no cover - Pillow may be absent in unit tests
        return

    ImageFile.LOAD_TRUNCATED_IMAGES = True


def _install() -> None:
    _enable_truncated_images()

    state = _load_state()
    if not state or not state.get("enabled", False):
        return

    seed = state.get("seed")
    skip_once = int(state.get("skip_samples_once", 0))

    if seed is not None:
        _seed_global_rngs(int(seed))

    _patch_random_sampler(seed, skip_once)


_install()
