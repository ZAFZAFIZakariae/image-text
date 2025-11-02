"""Convenience launcher for RealVis XL LyCORIS fine-tuning with Kohya."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class Preset:
    """Configuration options for a hardware preset."""

    name: str
    description: str
    overrides: Dict[str, str]


COMMON_DEFAULTS: Dict[str, str] = {
    "network_dim": "192",
    "network_alpha": "96",
    "lora_dropout": "0.05",
    "learning_rate": "1e-4",
    "text_encoder_lr": "5e-6",
    "weight_decay": "0.01",
    "max_grad_norm": "1.0",
    "lr_scheduler": "cosine",
    "lr_warmup_ratio": "0.05",
    "max_train_steps": "20000",
    "train_batch_size": "2",
    "gradient_accumulation_steps": "4",
    "max_data_loader_n_workers": "8",
    "log_prefix": "L4_full60k",
}


PRESETS: Dict[str, Preset] = {
    "L4": Preset(
        name="L4",
        description="Optimised for NVIDIA L4 / 24GB GPUs",
        overrides={},
    ),
    "T4": Preset(
        name="T4",
        description="Reduced batch for NVIDIA T4 / 16GB GPUs",
        overrides={
            "train_batch_size": "1",
            "gradient_accumulation_steps": "8",
            "max_data_loader_n_workers": "6",
            "log_prefix": "T4_full60k",
        },
    ),
}


def build_command(args: argparse.Namespace) -> List[str]:
    """Assemble the kohya_ss command from CLI arguments."""

    preset = PRESETS[args.preset]

    resolved: Dict[str, str] = dict(COMMON_DEFAULTS)
    resolved.update(preset.overrides)

    # Apply user overrides when provided.
    for key in (
        "network_dim",
        "network_alpha",
        "lora_dropout",
        "learning_rate",
        "text_encoder_lr",
        "weight_decay",
        "max_grad_norm",
        "lr_scheduler",
        "lr_warmup_ratio",
        "max_train_steps",
        "train_batch_size",
        "gradient_accumulation_steps",
        "max_data_loader_n_workers",
        "log_prefix",
    ):
        value = getattr(args, key, None)
        if value is not None:
            resolved[key] = str(value)

    command = [
        sys.executable,
        "-m",
        "kohya_ss.train_network",
        "--network_module=lycoris.kohya",
        "--algo=locon_dora",
        f"--network_dim={resolved['network_dim']}",
        f"--network_alpha={resolved['network_alpha']}",
        f"--lora_dropout={resolved['lora_dropout']}",
        f"--pretrained_model_name_or_path={args.base_model}",
        f"--train_data_dir={args.data_dir}",
        f"--caption_extension={args.caption_extension}",
        "--resolution=1024",
        "--min_bucket_reso=640",
        "--max_bucket_reso=1024",
        "--bucket_reso_steps=64",
        f"--output_dir={args.output_dir}",
        f"--output_name={args.output_name}",
        f"--learning_rate={resolved['learning_rate']}",
        f"--text_encoder_lr={resolved['text_encoder_lr']}",
        "--optimizer_type=adamw8bit",
        f"--weight_decay={resolved['weight_decay']}",
        f"--max_grad_norm={resolved['max_grad_norm']}",
        f"--lr_scheduler={resolved['lr_scheduler']}",
        f"--lr_warmup_ratio={resolved['lr_warmup_ratio']}",
        "--train_unet",
        "--train_text_encoder",
        "--network_train_unet_only=0",
        "--mixed_precision=bf16",
        "--gradient_checkpointing",
        "--min_snr_gamma=5.0",
        "--noise_offset=0.02",
        f"--max_data_loader_n_workers={resolved['max_data_loader_n_workers']}",
        "--persistent_data_loader_workers",
        "--cache_latents_to_disk",
        "--save_every_n_steps=1000",
        "--save_model_as=safetensors",
        f"--log_prefix={resolved['log_prefix']}",
        f"--max_train_steps={resolved['max_train_steps']}",
        f"--train_batch_size={resolved['train_batch_size']}",
        f"--gradient_accumulation_steps={resolved['gradient_accumulation_steps']}",
    ]

    if args.seed is not None:
        command.append(f"--seed={args.seed}")

    if args.resume is not None:
        command.append(f"--resume={args.resume}")

    if args.additional_argument:
        command.extend(args.additional_argument)

    return command


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Kohya LyCORIS (LoCon + DoRA) fine-tuning for RealVis XL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--base-model",
        required=True,
        help="Path to the RealVis XL v5.0 .safetensors checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing images and matching .txt captions",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store LyCORIS checkpoints",
    )
    parser.add_argument(
        "--output-name",
        default="realvisxl_locon_dora_full60k",
        help="Name prefix for saved checkpoints",
    )
    parser.add_argument(
        "--caption-extension",
        default=".txt",
        help="Caption file extension in the dataset directory",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="L4",
        help="Hardware preset defining batch size and loader workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional RNG seed for training",
    )
    parser.add_argument(
        "--resume",
        help="Path to a previous Kohya checkpoint to resume from",
    )

    parser.add_argument(
        "--kohya-ss-package",
        default="git+https://github.com/bmaltais/kohya_ss.git@master",
        help=(
            "Pip requirement to install if kohya_ss is missing. Accepts local paths"
            " for offline environments."
        ),
    )
    parser.add_argument(
        "--lycoris-package",
        default="lycoris-lora",
        help=(
            "Pip requirement to install if lycoris is missing. Accepts local paths"
            " for offline environments."
        ),
    )
    parser.add_argument(
        "--skip-auto-install",
        action="store_true",
        help="Disable automatic pip installation of missing dependencies.",
    )

    # Optional overrides
    for name in (
        "network_dim",
        "network_alpha",
        "lora_dropout",
        "learning_rate",
        "text_encoder_lr",
        "weight_decay",
        "max_grad_norm",
        "lr_scheduler",
        "lr_warmup_ratio",
        "max_train_steps",
        "train_batch_size",
        "gradient_accumulation_steps",
        "max_data_loader_n_workers",
        "log_prefix",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name)

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the computed command without executing it",
    )
    parser.add_argument(
        "--additional-argument",
        action="append",
        metavar="ARG",
        help=(
            "Extra kohya_ss arguments (use quotes to include =). Repeat the flag to add multiple options."
        ),
    )

    return parser.parse_args(argv)


def ensure_directories(args: argparse.Namespace) -> None:
    if not os.path.exists(args.base_model):
        raise FileNotFoundError(f"Base model not found: {args.base_model}")
    if not os.path.isdir(args.data_dir):
        raise NotADirectoryError(f"Dataset directory not found: {args.data_dir}")
    os.makedirs(args.output_dir, exist_ok=True)


def _format_missing_dependency_hint(remaining: Sequence[str]) -> str:
    hint = [
        "Missing training dependencies: {}.".format(", ".join(sorted(remaining)))
    ]

    if "kohya_ss" in remaining:
        hint.append(
            "Install Kohya manually or point --kohya-ss-package to a local path."
        )
    if "lycoris" in remaining:
        hint.append(
            "Install LyCORIS manually or point --lycoris-package to a local path."
        )

    return " ".join(hint)


def ensure_dependencies(args: argparse.Namespace) -> None:
    """Ensure the required third-party modules are available."""

    dependencies = {
        "kohya_ss": args.kohya_ss_package,
        "lycoris": args.lycoris_package,
    }

    missing = [
        name for name in dependencies if importlib.util.find_spec(name) is None
    ]

    if not missing:
        return

    if args.skip_auto_install:
        raise ModuleNotFoundError(_format_missing_dependency_hint(missing))

    # Try to install the missing packages automatically to reduce friction for
    # first-time users. We keep the installation quiet to avoid excessive log
    # spam but still report the command being executed. If a package does not
    # have an install target (empty string) we simply skip the automated
    # installation attempt so the user can provide their own distribution.
    for module_name in missing:
        requirement = dependencies[module_name]
        if not requirement:
            continue
        print(
            f"Missing dependency '{module_name}'. Attempting to install via pip: {requirement}"
        )
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    requirement,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - network dependent
            raise ModuleNotFoundError(
                f"Failed to install '{module_name}'. Original error: {exc}"
            ) from exc

    # Validate that the installation succeeded before continuing. If a package
    # is still unavailable we raise a helpful error message with manual
    # installation guidance so the user can resolve the issue themselves.
    remaining = [
        name for name in dependencies if importlib.util.find_spec(name) is None
    ]

    if remaining:
        raise ModuleNotFoundError(_format_missing_dependency_hint(remaining))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    ensure_dependencies(args)
    ensure_directories(args)
    command = build_command(args)

    printable = " ".join(shlex.quote(token) for token in command)
    print("Running:")
    print(printable)

    if args.dry_run:
        return 0

    process = subprocess.run(command)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
