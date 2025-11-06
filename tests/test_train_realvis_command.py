import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from train_realvis_locon_dora import (
    build_command,
    detect_cache_latents_flag,
    parse_args,
    resolve_training_config,
)


MINIMAL_ARGS = [
    "--base-model",
    "/models/RealVisXL_V5.0_fp16.safetensors",
    "--data-dir",
    "/data",
    "--output-dir",
    "/out",
]


def make_args(extra=None):
    argv = list(MINIMAL_ARGS)
    if extra:
        argv.extend(extra)
    return parse_args(argv)


def build(extra=None, cache_flag=Ellipsis):
    args = make_args(extra)
    resolved = resolve_training_config(args)
    kwargs = {}
    if cache_flag is not Ellipsis:
        kwargs["cache_latents_flag"] = cache_flag
    command = build_command(args, resolved, Path("/train_network.py"), **kwargs)
    return command


def test_command_includes_cache_latents_by_default():
    command = build()
    assert "--cache_latents_to_disk" in command


def test_command_includes_multiple_cache_latents_flags():
    command = build(cache_flag=("--cache_latents_to_disk", "--cache_latents"))
    assert "--cache_latents_to_disk" in command
    assert "--cache_latents" in command


def test_parse_args_supports_allow_truncated_images_flag():
    args = make_args(["--allow-truncated-images"])
    assert args.allow_truncated_images is True


def test_command_omits_cache_latents_when_disabled():
    command = build(["--no-cache-latents-to-disk"])
    assert "--cache_latents_to_disk" not in command


def test_command_uses_cache_latents_flag_when_specified():
    command = build(cache_flag="--cache_latents")
    assert "--cache_latents" in command
    assert "--cache_latents_to_disk" not in command


def _write_script(tmp_path, body: str) -> Path:
    train_script = tmp_path / "train_network.py"
    train_script.write_text(body, encoding="utf-8")
    return train_script


def test_detect_cache_latents_prefers_disk(tmp_path):
    train_script = tmp_path / "train_network.py"
    train_script.write_text(
        """
from argparse import ArgumentParser


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--cache_latents_to_disk", action="store_true")
    return parser
""",
        encoding="utf-8",
    )

    flag = detect_cache_latents_flag(train_script)
    assert flag == ("--cache_latents_to_disk",)


def test_detect_cache_latents_falls_back_to_cache_latents(tmp_path):
    train_script = _write_script(
        tmp_path,
        """
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--cache_latents", action="store_true")
""",
    )

    flag = detect_cache_latents_flag(train_script)
    assert flag == ("--cache_latents",)


def test_detect_cache_latents_handles_missing_flag(tmp_path):
    train_script = _write_script(tmp_path, "print('hello world')\n")

    flag = detect_cache_latents_flag(train_script)
    assert flag is None


def test_detect_cache_latents_returns_both_when_present(tmp_path):
    train_script = _write_script(
        tmp_path,
        """
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--cache_latents_to_disk", action="store_true")
parser.add_argument("--cache_latents", action="store_true")
""",
    )

    flag = detect_cache_latents_flag(train_script)
    assert flag == ("--cache_latents_to_disk", "--cache_latents")


def test_detect_cache_latents_ignores_comments(tmp_path):
    train_script = _write_script(
        tmp_path,
        """
from argparse import ArgumentParser


parser = ArgumentParser()
# parser.add_argument("--cache_latents_to_disk", action="store_true")
parser.add_argument("--cache_latents", action="store_true")
""",
    )

    flag = detect_cache_latents_flag(train_script)
    assert flag == ("--cache_latents",)
