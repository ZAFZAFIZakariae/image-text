import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from train_realvis_locon_dora import build_command, parse_args, resolve_training_config


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


def build(extra=None):
    args = make_args(extra)
    resolved = resolve_training_config(args)
    command = build_command(args, resolved, Path("/train_network.py"))
    return command


def test_command_includes_cache_latents_by_default():
    command = build()
    assert "--cache_latents_to_disk" in command


def test_command_omits_cache_latents_when_disabled():
    command = build(["--no-cache-latents-to-disk"])
    assert "--cache_latents_to_disk" not in command
