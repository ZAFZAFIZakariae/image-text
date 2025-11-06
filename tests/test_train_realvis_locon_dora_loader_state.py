import argparse
import json
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

from train_realvis_locon_dora import (
    DEFAULT_DATA_LOADER_SEED,
    DataLoaderState,
    collect_dataset_entries,
    infer_resume_step,
    prepare_dataloader_state,
)


def make_namespace(**kwargs):
    defaults = dict(
        base_model="/tmp/model.safetensors",
        data_dir="",
        caption_extension=".txt",
        output_dir="",
        output_name="test",
        seed=None,
        resume=None,
        resume_step=None,
        disable_dataloader_state=False,
        force_cache_latents=False,
        cache_latents_skip_threshold=20000,
        no_cache_latents_to_disk=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_infer_resume_step_extracts_trailing_digits():
    assert infer_resume_step("/foo/bar/loral-00001000.safetensors") == 1000
    assert infer_resume_step("checkpoint-step9999.pt") == 9999
    assert infer_resume_step("model-final") is None


def test_collect_dataset_entries_filters_missing_captions(tmp_path, capsys):
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()

    image_with_caption = data_dir / "img001.png"
    image_with_caption.write_bytes(b"fake")
    (data_dir / "img001.txt").write_text("caption", encoding="utf-8")

    image_without_caption = data_dir / "img002.png"
    image_without_caption.write_bytes(b"fake")

    entries = collect_dataset_entries(data_dir, ".txt")
    # Only the image with a caption should be returned.
    assert [path.name for path in entries] == ["img001.png"]

    stderr = capsys.readouterr().err
    assert "Skipping 1 images" in stderr


def test_prepare_dataloader_state_creates_state_file(tmp_path):
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()

    for idx in range(3):
        image_path = data_dir / f"image{idx:02d}.png"
        image_path.write_bytes(b"fake")
        caption_path = data_dir / f"image{idx:02d}.txt"
        caption_path.write_text("caption", encoding="utf-8")

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    args = make_namespace(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        output_name="loras",
    )

    resolved = {"train_batch_size": "2", "gradient_accumulation_steps": "4"}

    images = collect_dataset_entries(data_dir, ".txt")

    state = prepare_dataloader_state(args, resolved, images)
    assert isinstance(state, DataLoaderState)
    assert state.seed == DEFAULT_DATA_LOADER_SEED
    assert state.skip_samples_once == 0
    assert state.dataset_size == 3

    with Path(state.path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["seed"] == DEFAULT_DATA_LOADER_SEED
    assert payload["dataset_size"] == 3


def test_prepare_dataloader_state_honours_resume_step(tmp_path):
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()

    for idx in range(5):
        image_path = data_dir / f"sample{idx}.png"
        image_path.write_bytes(b"fake")
        (data_dir / f"sample{idx}.txt").write_text("caption", encoding="utf-8")

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    args = make_namespace(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        output_name="loras",
        resume="/checkpoints/loras-00000123.safetensors",
        resume_step=123,
    )

    resolved = {"train_batch_size": "3", "gradient_accumulation_steps": "2"}
    images = collect_dataset_entries(data_dir, ".txt")
    state = prepare_dataloader_state(args, resolved, images)

    # samples_per_step = 3 * 2 = 6 -> skip = (123 * 6) % 5 = 3
    assert state.skip_samples_once == (123 * 6) % 5

    with Path(state.path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["skip_samples_once"] == state.skip_samples_once
