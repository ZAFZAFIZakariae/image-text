import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import create_caption_sidecars


def test_create_caption_sidecars_handles_subdirectories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset_dir = tmp_path / "dataset"
    nested_dir = dataset_dir / "text2imagedataset"
    nested_dir.mkdir(parents=True)

    image_path = nested_dir / "example.jpg"
    image_path.write_bytes(b"not an actual image but sufficient for path handling")

    manifest_path = dataset_dir / "metadata.jsonl"
    manifest_path.write_text(
        json.dumps({"file": "text2imagedataset/example.jpg", "text": "A caption."}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_caption_sidecars.py",
            str(dataset_dir),
            "--manifest",
            str(manifest_path),
        ],
    )

    create_caption_sidecars.main()

    caption_path = image_path.with_suffix(".txt")
    assert caption_path.exists()
    assert caption_path.read_text(encoding="utf-8") == "A caption.\n"


def test_create_caption_sidecars_supports_external_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images"
    nested_dir = images_dir / "text2imagedataset"
    nested_dir.mkdir(parents=True)

    image_path = nested_dir / "example.jpg"
    image_path.write_bytes(b"binary image placeholder")

    manifest_dir = dataset_root / "manifests"
    manifest_dir.mkdir(parents=True)
    manifest_path = manifest_dir / "metadata.jsonl"
    manifest_path.write_text(
        json.dumps({"file": "text2imagedataset/example.jpg", "text": "Another caption."})
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(dataset_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_caption_sidecars.py",
            str(images_dir),
            "--manifest",
            "manifests/metadata.jsonl",
        ],
    )

    create_caption_sidecars.main()

    caption_path = image_path.with_suffix(".txt")
    assert caption_path.exists()
    assert caption_path.read_text(encoding="utf-8") == "Another caption.\n"
