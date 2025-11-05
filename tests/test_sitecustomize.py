"""Tests for helper hooks installed via sitecustomize."""

from __future__ import annotations

import importlib
import sys
import types


def test_sitecustomize_enables_truncated_images(monkeypatch):
    monkeypatch.delenv("IMAGETEXT_DATA_LOADER_STATE", raising=False)
    monkeypatch.setenv("IMAGETEXT_ALLOW_TRUNCATED_IMAGES", "1")

    imagefile_module = types.ModuleType("PIL.ImageFile")
    imagefile_module.LOAD_TRUNCATED_IMAGES = False

    pil_module = types.ModuleType("PIL")
    pil_module.ImageFile = imagefile_module

    monkeypatch.setitem(sys.modules, "PIL", pil_module)
    monkeypatch.setitem(sys.modules, "PIL.ImageFile", imagefile_module)

    sys.modules.pop("sitecustomize", None)
    importlib.import_module("sitecustomize")

    assert imagefile_module.LOAD_TRUNCATED_IMAGES is True

