"""Convenience launcher for RealVis XL LyCORIS fine-tuning with Kohya."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class Preset:
    """Configuration options for a hardware preset."""

    name: str
    description: str
    overrides: Dict[str, str]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEFAULT_DATA_LOADER_SEED = 3407


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

def resolve_training_config(args: argparse.Namespace) -> Dict[str, str]:
    """Resolve the merged preset + CLI configuration values."""

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

    return resolved


def build_command(args: argparse.Namespace, resolved: Dict[str, str]) -> List[str]:
    """Assemble the kohya_ss command from CLI arguments."""

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
        "--resume-step",
        type=int,
        help=(
            "Number of optimizer steps completed in the checkpoint specified with --resume. "
            "Overrides the value inferred from the checkpoint filename."
        ),
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
        "--disable-dataloader-state",
        action="store_true",
        help=(
            "Skip installing the deterministic DataLoader patch that preserves ordering "
            "and skips batches on resume."
        ),
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


def ensure_dependencies() -> None:
    """Ensure the required third-party modules are available."""

    missing = []
    for module_name in ("kohya_ss", "lycoris"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)

    if not missing:
        return

    hint = [
        "Missing training dependencies: {}.".format(", ".join(sorted(missing)))
    ]

    if "kohya_ss" in missing:
        hint.append(
            "Install Kohya with `pip install -q git+https://github.com/bmaltais/kohya_ss.git@master`."
        )
    if "lycoris" in missing:
        hint.append(
            "Install LyCORIS with `pip install -q lycoris-lora`."
        )

    raise ModuleNotFoundError(" ".join(hint))


def _iter_image_entries(data_dir: Path) -> Iterable[Path]:
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def collect_dataset_entries(data_dir: Path, caption_extension: str) -> List[Path]:
    """Collect dataset image paths that have matching caption files."""

    if not caption_extension.startswith("."):
        caption_extension = f".{caption_extension}"

    images: List[Path] = []
    missing_captions: List[Path] = []

    for image_path in _iter_image_entries(data_dir):
        caption_path = image_path.with_suffix(caption_extension)
        if caption_path.exists():
            images.append(image_path)
        else:
            missing_captions.append(image_path)

    if missing_captions:
        print(
            "Warning: Skipping %d images without matching %s captions." % (
                len(missing_captions),
                caption_extension,
            ),
            file=sys.stderr,
        )

    images.sort()
    return images


def infer_resume_step(resume_path: str) -> Optional[int]:
    """Best-effort extraction of the completed step count from a checkpoint name."""

    stem = Path(resume_path).stem
    match = re.search(r"(\d+)$", stem)
    if match:
        return int(match.group(1))
    return None


@dataclass(frozen=True)
class DataLoaderState:
    path: Path
    seed: int
    skip_samples_once: int
    dataset_size: int


def prepare_dataloader_state(
    args: argparse.Namespace, resolved: Dict[str, str]
) -> Optional[DataLoaderState]:
    """Generate the state file consumed by ``sitecustomize`` to patch PyTorch."""

    if args.disable_dataloader_state:
        return None

    data_dir = Path(args.data_dir)
    images = collect_dataset_entries(data_dir, args.caption_extension)

    if not images:
        raise ValueError(
            "No training images with matching captions were found in the dataset directory."
        )

    dataset_size = len(images)
    state_path = Path(args.output_dir) / f"{args.output_name}_dataloader_state.json"

    existing_seed: Optional[int] = None
    if state_path.exists():
        try:
            with state_path.open("r", encoding="utf-8") as fh:
                state_data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            state_data = {}
        existing_seed = state_data.get("seed")

    seed = args.seed if args.seed is not None else existing_seed or DEFAULT_DATA_LOADER_SEED

    resume_step = None
    if args.resume:
        if args.resume_step is not None:
            resume_step = args.resume_step
        else:
            resume_step = infer_resume_step(args.resume)
        if resume_step is None:
            raise ValueError(
                "Unable to infer the completed step count from the --resume checkpoint. "
                "Provide --resume-step explicitly to enable DataLoader skipping."
            )

    skip_samples_once = 0
    if resume_step is not None:
        train_batch_size = int(resolved["train_batch_size"])
        grad_accum = int(resolved["gradient_accumulation_steps"])
        samples_per_step = train_batch_size * grad_accum
        skip_samples_once = (resume_step * samples_per_step) % dataset_size

    state_payload = {
        "enabled": True,
        "seed": seed,
        "skip_samples_once": skip_samples_once,
        "dataset_size": dataset_size,
    }

    with state_path.open("w", encoding="utf-8") as fh:
        json.dump(state_payload, fh)

    if skip_samples_once:
        print(
            f"DataLoader resume: skipping the first {skip_samples_once} samples "
            f"out of {dataset_size} to align with checkpoint progress.",
            file=sys.stderr,
        )

    print(f"DataLoader seed fixed at {seed} for deterministic ordering.")

    return DataLoaderState(state_path, seed, skip_samples_once, dataset_size)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    ensure_dependencies()
    ensure_directories(args)
    resolved = resolve_training_config(args)

    loader_state = prepare_dataloader_state(args, resolved)
    if loader_state is not None:
        args.seed = loader_state.seed

    command = build_command(args, resolved)

    printable = " ".join(shlex.quote(token) for token in command)
    print("Running:")
    print(printable)

    if args.dry_run:
        return 0

    env = os.environ.copy()
    if loader_state is not None:
        env["IMAGETEXT_DATA_LOADER_STATE"] = str(loader_state.path)
        project_root = str(Path(__file__).resolve().parent)
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{project_root}:{existing_pythonpath}" if existing_pythonpath else project_root
        )

    process = subprocess.run(command, env=env)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
