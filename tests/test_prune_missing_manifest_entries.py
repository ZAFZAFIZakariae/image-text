from __future__ import annotations

import json
from pathlib import Path

import pytest

from prune_missing_manifest_entries import prune_manifest


def _write_manifest(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_prune_manifest_removes_missing_images(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    existing = dataset_dir / "existing.png"
    existing.write_bytes(b"fake")

    nested_dir = dataset_dir / "subdir"
    nested_dir.mkdir()
    nested_image = nested_dir / "nested.png"
    nested_image.write_bytes(b"fake")

    manifest_path = dataset_dir / "metadata.jsonl"
    _write_manifest(
        manifest_path,
        [
            {"file": "existing.png", "text": "kept"},
            {"file": "missing.png", "text": "dropped"},
            {"file": "subdir/nested.png", "text": "kept"},
            {"file": "dataset/subdir/nested.png", "text": "kept"},
        ],
    )

    kept, removed, total = prune_manifest(dataset_dir, manifest_path)

    assert (kept, removed, total) == (3, 1, 4)

    with manifest_path.open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]

    assert [record["file"] for record in lines] == [
        "existing.png",
        "subdir/nested.png",
        "dataset/subdir/nested.png",
    ]


def test_prune_manifest_raises_when_missing_file_field(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    manifest_path = dataset_dir / "metadata.jsonl"
    _write_manifest(manifest_path, [{"text": "no file field"}])

    with pytest.raises(ValueError):
        prune_manifest(dataset_dir, manifest_path)
