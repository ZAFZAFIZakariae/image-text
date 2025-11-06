import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from train_realvis_locon_dora import (
    adjust_mixed_precision_for_hardware,
    build_command,
    detect_argument_flag,
    detect_cache_latents_flag,
    ensure_accelerate_config,
    ensure_optimizer_compatibility,
    evaluate_cache_latents_policy,
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


def build(
    extra=None,
    cache_flag=Ellipsis,
    supports_sdxl=Ellipsis,
    supports_v2=Ellipsis,
):
    args = make_args(extra)
    resolved = resolve_training_config(args)
    kwargs = {}
    if cache_flag is not Ellipsis:
        kwargs["cache_latents_flag"] = cache_flag
    if supports_sdxl is not Ellipsis:
        kwargs["supports_sdxl"] = supports_sdxl
    if supports_v2 is not Ellipsis:
        kwargs["supports_v2"] = supports_v2
    command = build_command(args, resolved, Path("/train_network.py"), **kwargs)
    return command


def test_command_includes_cache_latents_by_default():
    command = build()
    assert "--cache_latents_to_disk" in command


def test_command_includes_sdxl_and_v2_by_default():
    command = build()
    assert "--sdxl" in command
    assert "--v2" in command


def test_command_includes_multiple_cache_latents_flags():
    command = build(cache_flag=("--cache_latents_to_disk", "--cache_latents"))
    assert "--cache_latents_to_disk" in command
    assert "--cache_latents" in command


def test_parse_args_supports_allow_truncated_images_flag():
    args = make_args(["--allow-truncated-images"])
    assert args.allow_truncated_images is True


def test_parse_args_supports_cache_latents_threshold():
    args = make_args(["--cache-latents-skip-threshold", "1000"])
    assert args.cache_latents_skip_threshold == 1000


def test_command_omits_cache_latents_when_disabled():
    command = build(["--no-cache-latents-to-disk"])
    assert "--cache_latents_to_disk" not in command


def test_command_uses_cache_latents_flag_when_specified():
    command = build(cache_flag="--cache_latents")
    assert "--cache_latents" in command
    assert "--cache_latents_to_disk" not in command


def test_command_omits_sdxl_when_not_supported():
    command = build(supports_sdxl=False)
    assert "--sdxl" not in command


def test_command_omits_v2_when_not_supported():
    command = build(supports_v2=False)
    assert "--v2" not in command


def test_command_includes_mixed_precision_setting():
    command = build()
    assert "--mixed_precision=bf16" in command


def test_command_allows_mixed_precision_override():
    command = build(["--mixed-precision", "fp16"])
    assert "--mixed_precision=fp16" in command


def test_command_allows_optimizer_override():
    command = build(["--optimizer-type", "adamw"])
    assert "--optimizer_type=adamw" in command


def test_evaluate_cache_latents_policy_disables_large_dataset():
    args = parse_args(list(MINIMAL_ARGS))
    should_cache, message = evaluate_cache_latents_policy(
        args, 25000, ("--cache_latents_to_disk",)
    )
    assert should_cache is False
    assert message is not None
    assert "25000" in message
    assert "auto-disable threshold" in message


def test_evaluate_cache_latents_policy_respects_force_override():
    args = parse_args(list(MINIMAL_ARGS) + ["--force-cache-latents"])
    should_cache, message = evaluate_cache_latents_policy(
        args, 50000, ("--cache_latents_to_disk",)
    )
    assert should_cache is True
    assert message is None


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


def test_detect_argument_flag_identifies_present_flag(tmp_path):
    train_script = _write_script(
        tmp_path,
        """
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--sdxl", action="store_true")
""",
    )

    assert detect_argument_flag(train_script, "--sdxl") is True


def test_detect_argument_flag_identifies_missing_flag(tmp_path):
    train_script = _write_script(
        tmp_path,
        """
from argparse import ArgumentParser


parser = ArgumentParser()
""",
    )

    assert detect_argument_flag(train_script, "--sdxl") is False


def test_ensure_accelerate_config_creates_file(tmp_path, monkeypatch):
    config_path = tmp_path / "accelerate" / "default_config.yaml"
    recorded: dict[str, str] = {}

    accelerate_module = types.ModuleType("accelerate")
    commands_module = types.ModuleType("accelerate.commands")
    config_module = types.ModuleType("accelerate.commands.config")
    utils_module = types.ModuleType("accelerate.utils")

    config_module.default_config_file = str(config_path)

    def fake_write_basic_config(*, mixed_precision="no", save_location: str) -> None:
        recorded["mixed_precision"] = mixed_precision
        recorded["save_location"] = save_location
        Path(save_location).parent.mkdir(parents=True, exist_ok=True)
        Path(save_location).write_text("config", encoding="utf-8")

    utils_module.write_basic_config = fake_write_basic_config

    accelerate_module.commands = commands_module
    accelerate_module.utils = utils_module
    commands_module.config = config_module

    monkeypatch.setitem(sys.modules, "accelerate", accelerate_module)
    monkeypatch.setitem(sys.modules, "accelerate.commands", commands_module)
    monkeypatch.setitem(sys.modules, "accelerate.commands.config", config_module)
    monkeypatch.setitem(sys.modules, "accelerate.utils", utils_module)

    ensure_accelerate_config({"mixed_precision": "bf16"})

    assert recorded["mixed_precision"] == "bf16"
    assert Path(recorded["save_location"]).exists()


def test_adjust_mixed_precision_downgrades_without_support(monkeypatch):
    resolved = {"mixed_precision": "bf16"}

    fake_torch = types.ModuleType("torch")
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: False,
        get_device_capability=lambda device=0: (7, 5),
    )
    fake_torch.cuda = fake_cuda

    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    adjust_mixed_precision_for_hardware(resolved)

    assert resolved["mixed_precision"] == "fp16"


def test_adjust_mixed_precision_preserves_when_supported(monkeypatch):
    resolved = {"mixed_precision": "bf16"}

    fake_torch = types.ModuleType("torch")
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
    )
    fake_torch.cuda = fake_cuda

    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    adjust_mixed_precision_for_hardware(resolved)

    assert resolved["mixed_precision"] == "bf16"


def test_optimizer_fallback_when_bitsandbytes_missing(monkeypatch):
    resolved = {"optimizer_type": "adamw8bit"}

    import builtins

    real_import = builtins.__import__

    def raising_import(name, *args, **kwargs):
        if name.startswith("bitsandbytes"):
            raise ModuleNotFoundError("bitsandbytes not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", raising_import)

    ensure_optimizer_compatibility(resolved)

    assert resolved["optimizer_type"] == "adamw"


def test_optimizer_keeps_8bit_when_supported(monkeypatch):
    resolved = {"optimizer_type": "adamw8bit"}

    fake_bnb = types.ModuleType("bitsandbytes")
    fake_bnb.__version__ = "0.test"
    fake_optim = types.ModuleType("bitsandbytes.optim")

    class FakeAdamW8bit:  # pragma: no cover - minimal stub for import
        pass

    fake_optim.AdamW8bit = FakeAdamW8bit

    fake_torch = types.ModuleType("torch")
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda device=0: (8, 0),
    )
    fake_torch.cuda = fake_cuda

    monkeypatch.setitem(sys.modules, "bitsandbytes", fake_bnb)
    monkeypatch.setitem(sys.modules, "bitsandbytes.optim", fake_optim)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    ensure_optimizer_compatibility(resolved)

    assert resolved["optimizer_type"] == "adamw8bit"
