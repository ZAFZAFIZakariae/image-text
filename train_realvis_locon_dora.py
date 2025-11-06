"""Convenience launcher for RealVis XL LyCORIS fine-tuning with Kohya."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import ast
import re
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class Preset:
    """Configuration options for a hardware preset."""

    name: str
    description: str
    overrides: Dict[str, str]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEFAULT_DATA_LOADER_SEED = 3407


def log_step(message: str) -> None:
    """Emit a progress message that flushes immediately."""

    print(f"[train_realvis] {message}", flush=True)


COMMON_DEFAULTS: Dict[str, str] = {
    "network_dim": "192",
    "network_alpha": "96",
    "network_dropout": "0.05",
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
        "network_dropout",
        "learning_rate",
        "text_encoder_lr",
        "weight_decay",
        "max_grad_norm",
        "lr_scheduler",
        "lr_warmup_ratio",
        "lr_warmup_steps",
        "max_train_steps",
        "train_batch_size",
        "gradient_accumulation_steps",
        "max_data_loader_n_workers",
        "log_prefix",
    ):
        value = getattr(args, key, None)
        if value is not None:
            resolved[key] = str(value)

    if "network_dropout" not in resolved:
        # Backward compatibility with older presets that still reference lora_dropout.
        lora_dropout = resolved.pop("lora_dropout", None)
        if lora_dropout is not None:
            resolved["network_dropout"] = lora_dropout

    resolved.pop("lora_dropout", None)

    # Convert ratio-based warmup configuration to an explicit step count for
    # the modern kohya_ss argument format.
    warmup_steps = resolved.get("lr_warmup_steps")
    if warmup_steps in (None, "None"):
        ratio = resolved.get("lr_warmup_ratio")
        try:
            if ratio is not None:
                warmup_steps = int(
                    round(float(ratio) * int(resolved["max_train_steps"]))
                )
        except (ValueError, KeyError):
            warmup_steps = None

    if warmup_steps is not None:
        resolved["lr_warmup_steps"] = str(int(warmup_steps))
    else:
        resolved.pop("lr_warmup_steps", None)

    resolved.pop("lr_warmup_ratio", None)

    return resolved


CacheLatentsFlags = Optional[Sequence[str]]


def _extract_cache_flags(script_text: str) -> Tuple[bool, bool] | None:
    """Return availability of cache flags via AST inspection.

    ``train_network.py`` is large and occasionally changes how it documents
    command line arguments. Earlier versions referenced ``--cache_latents`` in
    comments even though the option itself was removed, which tricked a plain
    string search into assuming the flag still existed. By walking the AST and
    checking for real ``add_argument`` calls we can confirm that an option is
    genuinely registered with ``argparse``.
    """

    try:
        tree = ast.parse(script_text)
    except SyntaxError:
        return None

    has_cache_latents_to_disk = False
    has_cache_latents = False

    class _ArgumentVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - argparse API
            nonlocal has_cache_latents_to_disk, has_cache_latents

            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
                self.generic_visit(node)
                return

            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value == "--cache_latents_to_disk":
                        has_cache_latents_to_disk = True
                    elif arg.value == "--cache_latents":
                        has_cache_latents = True

            self.generic_visit(node)

    _ArgumentVisitor().visit(tree)

    return has_cache_latents_to_disk, has_cache_latents


def detect_cache_latents_flag(train_script: Path) -> CacheLatentsFlags:
    """Detect the appropriate cache latents CLI flag for ``train_network.py``.

    Historically Kohya exposed ``--cache_latents_to_disk`` while newer versions
    renamed the option to ``--cache_latents``. Reading the training script avoids
    passing a flag that is no longer recognised, which would otherwise abort the
    launch with an ``unrecognized arguments`` error.
    """

    try:
        script_text = train_script.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        # When the script cannot be inspected fall back to the legacy option
        # which matches the default behaviour prior to detection support.
        return ("--cache_latents_to_disk",)

    extracted = _extract_cache_flags(script_text)
    if extracted is None:
        has_cache_latents_to_disk = bool(
            re.search(r"add_argument\([^#\n]*--cache_latents_to_disk", script_text)
        )
        has_cache_latents = bool(
            re.search(r"add_argument\([^#\n]*--cache_latents(?!_to_disk)", script_text)
        )
    else:
        has_cache_latents_to_disk, has_cache_latents = extracted

    flags: Tuple[str, ...]
    if has_cache_latents_to_disk and has_cache_latents:
        flags = ("--cache_latents_to_disk", "--cache_latents")
    elif has_cache_latents_to_disk:
        flags = ("--cache_latents_to_disk",)
    elif has_cache_latents:
        flags = ("--cache_latents",)
    else:
        return None

    return flags


def build_command(
    args: argparse.Namespace,
    resolved: Dict[str, str],
    train_script: Path,
    cache_latents_flag: Optional[Union[str, Sequence[str]]] = ("--cache_latents_to_disk",),
) -> List[str]:
    """Assemble the kohya_ss command from CLI arguments."""

    command = [
        sys.executable,
        "-u",
        str(train_script),
        "--network_module=lycoris.kohya",
        "--network_args",
        "algo=locon_dora",
        f"--network_dim={resolved['network_dim']}",
        f"--network_alpha={resolved['network_alpha']}",
        f"--network_dropout={resolved['network_dropout']}",
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
        f"--max_grad_norm={resolved['max_grad_norm']}",
        f"--lr_scheduler={resolved['lr_scheduler']}",
        "--mixed_precision=bf16",
        "--gradient_checkpointing",
        "--min_snr_gamma=5.0",
        "--noise_offset=0.02",
        f"--max_data_loader_n_workers={resolved['max_data_loader_n_workers']}",
        "--persistent_data_loader_workers",
        "--save_every_n_steps=1000",
        "--save_model_as=safetensors",
        f"--log_prefix={resolved['log_prefix']}",
        f"--max_train_steps={resolved['max_train_steps']}",
        f"--train_batch_size={resolved['train_batch_size']}",
        f"--gradient_accumulation_steps={resolved['gradient_accumulation_steps']}",
    ]

    cache_flags: List[str] = []
    if isinstance(cache_latents_flag, str):
        cache_flags = [cache_latents_flag]
    elif cache_latents_flag:
        cache_flags = list(cache_latents_flag)

    if cache_flags and not args.no_cache_latents_to_disk:
        command.extend(cache_flags)

    weight_decay = resolved.get("weight_decay")
    if weight_decay not in (None, "None"):
        command.extend(["--optimizer_args", f"weight_decay={weight_decay}"])

    warmup_steps = resolved.get("lr_warmup_steps")
    if warmup_steps is not None:
        command.append(f"--lr_warmup_steps={warmup_steps}")

    if args.network_train_unet_only:
        command.append("--network_train_unet_only")

    if getattr(args, "network_train_text_encoder_only", False):
        command.append("--network_train_text_encoder_only")

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

    parser.add_argument(
        "--network-train-unet-only",
        action="store_true",
        help="Disable text encoder network weights and train the UNet adapters only.",
    )
    parser.add_argument(
        "--network-train-text-encoder-only",
        action="store_true",
        help="Disable UNet network weights and train the text encoder adapters only.",
    )

    parser.add_argument(
        "--lora-dropout",
        dest="network_dropout",
        help="Deprecated alias for --network-dropout.",
    )

    # Optional overrides
    for name in (
        "network_dim",
        "network_alpha",
        "network_dropout",
        "learning_rate",
        "text_encoder_lr",
        "weight_decay",
        "max_grad_norm",
        "lr_scheduler",
        "lr_warmup_ratio",
        "lr_warmup_steps",
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
        "--no-cache-latents-to-disk",
        action="store_true",
        help=(
            "Skip adding --cache_latents_to_disk to the kohya_ss command. "
            "Disable this when temporary storage is limited or the option is not available."
        ),
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
        "--allow-truncated-images",
        action="store_true",
        help=(
            "Enable PIL's LOAD_TRUNCATED_IMAGES flag so that partially downloaded or "
            "otherwise truncated images do not abort latent caching."
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


COLAB_INSTALL_SNIPPET = textwrap.dedent(
    """
    # Install training dependencies
    !pip install --upgrade pip
    !pip install git+https://github.com/bmaltais/kohya_ss.git@master
    !pip install lycoris-lora
    """
).strip()


def _format_missing_dependency_hint(remaining: Sequence[str]) -> str:
    missing_display = ", ".join(sorted(remaining))
    lines = [f"Missing training dependencies: {missing_display}."]

    if "kohya_ss" in remaining:
        lines.append(
            "Install Kohya (kohya_ss) before launching the trainer or point the"
            " loader at an existing clone by setting the KOHYA_SS_PATH environment"
            " variable."
        )
        lines.append(
            "Local checkouts named 'kohya_ss' or 'kohya-ss' next to this script are"
            " detected automatically."
        )
    if "lycoris" in remaining:
        lines.append(
            "Install LyCORIS before launching the trainer."
            " If you cloned lycoris manually, set LYCORIS_PATH to the clone root"
            " so it can be discovered."
        )

    lines.append(
        "In Google Colab you can install the required packages by running the following cell:"
    )
    lines.append("\n" + COLAB_INSTALL_SNIPPET)

    return "\n".join(lines)


def _collect_subprocess_output(
    error: subprocess.CalledProcessError, limit: int = 20
) -> str:
    """Summarise stdout/stderr captured from a ``CalledProcessError``."""

    combined: List[str] = []

    for stream in (error.output, error.stderr):
        if not stream:
            continue

        if isinstance(stream, bytes):
            text = stream.decode("utf-8", errors="replace")
        else:
            text = str(stream)

        combined.extend(text.splitlines())

    if len(combined) > limit:
        combined = combined[-limit:]

    return "\n".join(combined)


def _iter_candidate_dependency_paths(package: str, repo_names: Sequence[str]) -> Iterable[Path]:
    """Yield potential directories that may contain a local clone of a dependency."""

    env_var = f"{package.upper()}_PATH"
    env_path = os.environ.get(env_var)
    if env_path:
        yield Path(env_path)

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()

    for base in {script_dir, cwd, script_dir.parent, cwd.parent}:
        for repo_name in repo_names:
            yield base / repo_name


_ADDED_DEPENDENCY_PATHS: List[str] = []


def _maybe_add_local_dependency(package: str, repo_names: Sequence[str]) -> None:
    """Add a local clone of ``package`` to ``sys.path`` when detected.

    Google Colab users often clone ``kohya_ss`` next to this repository instead of
    installing it via ``pip``. Import discovery fails in that scenario because the
    clone is not added to ``PYTHONPATH``. By probing for common locations we can
    provide a smoother out-of-the-box experience while still allowing explicit
    installations to take precedence when available.
    """

    for candidate in _iter_candidate_dependency_paths(package, repo_names):
        if not candidate.exists():
            continue

        # ``kohya_ss`` is importable from the repository root whereas ``lycoris``
        # exposes the package from the ``lycoris`` directory inside the clone.
        candidate_package = candidate / package
        if candidate_package.exists():
            path_to_add = candidate_package
        else:
            path_to_add = candidate

        path_str = str(path_to_add)

        if path_str not in sys.path:
            sys.path.insert(0, path_str)

        if path_str not in _ADDED_DEPENDENCY_PATHS:
            _ADDED_DEPENDENCY_PATHS.append(path_str)

        if importlib.util.find_spec(package) is not None:
            return


def ensure_dependencies(_: argparse.Namespace) -> None:
    """Ensure the required third-party modules are available."""

    _maybe_add_local_dependency("kohya_ss", ("kohya_ss", "kohya-ss"))
    _maybe_add_local_dependency("lycoris", ("lycoris", "lycoris-lora", "lycoris_lora"))

    missing = [
        name
        for name in ("kohya_ss", "lycoris")
        if importlib.util.find_spec(name) is None
    ]

    if missing:
        raise ModuleNotFoundError(_format_missing_dependency_hint(missing))


def resolve_kohya_train_script() -> Path:
    """Return the filesystem path to ``train_network.py`` inside ``kohya_ss``."""

    spec = importlib.util.find_spec("kohya_ss")
    if spec is None:
        raise ModuleNotFoundError(
            "kohya_ss is not installed or discoverable; install it before launching."
        )

    if spec.submodule_search_locations:
        package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    elif spec.origin:
        package_root = Path(spec.origin).resolve().parent
    else:
        raise ModuleNotFoundError("Unable to determine kohya_ss installation path.")

    candidates = [
        package_root / "sd-scripts" / "train_network.py",
        package_root.parent / "sd-scripts" / "train_network.py",
        package_root / "train_network.py",
        package_root.parent / "kohya_ss" / "sd-scripts" / "train_network.py",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate kohya_ss/sd-scripts/train_network.py. Ensure the repository is "
        "installed with its training scripts."
    )


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
    log_step(
        f"Discovered {len(images)} training images with captions in {data_dir}."
    )
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

    log_step(f"Wrote DataLoader state to {state_path}.")

    if skip_samples_once:
        print(
            f"DataLoader resume: skipping the first {skip_samples_once} samples "
            f"out of {dataset_size} to align with checkpoint progress.",
            file=sys.stderr,
        )

    print(f"DataLoader seed fixed at {seed} for deterministic ordering.", flush=True)

    return DataLoaderState(state_path, seed, skip_samples_once, dataset_size)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    log_step("Checking for required dependencies (kohya_ss, lycoris)...")
    ensure_dependencies(args)

    log_step("Validating base model, dataset, and output directories...")
    ensure_directories(args)

    log_step(f"Resolving training configuration for preset '{args.preset}'...")
    resolved = resolve_training_config(args)

    loader_state = prepare_dataloader_state(args, resolved)
    if loader_state is not None:
        args.seed = loader_state.seed
        log_step(
            f"DataLoader deterministic state prepared (seed={loader_state.seed}, "
            f"dataset_size={loader_state.dataset_size})."
        )
    else:
        log_step("DataLoader deterministic state disabled or unavailable.")

    train_script = resolve_kohya_train_script()
    log_step(f"Resolved kohya_ss training script at {train_script}.")
    cache_latents_flag = detect_cache_latents_flag(train_script)

    def _normalise_flags(flag: CacheLatentsFlags) -> Sequence[str]:
        if not flag:
            return ()
        if isinstance(flag, str):
            return (flag,)
        return tuple(flag)

    cache_flag_tokens = _normalise_flags(cache_latents_flag)
    if cache_flag_tokens and not args.no_cache_latents_to_disk:
        flags_display = ", ".join(cache_flag_tokens)
        log_step(
            "Latent caching enabled ({}). Kohya will pre-compute .npz files inside {} "
            "or its 'latents' subdirectory before training logs appear.".format(
                flags_display, args.data_dir
            )
        )
    elif cache_flag_tokens:
        flags_display = ", ".join(cache_flag_tokens)
        log_step(
            "Detected kohya_ss cache flag(s) ({}), but --no-cache-latents-to-disk was "
            "requested so latents will be encoded on the fly.".format(flags_display)
        )
    else:
        log_step(
            "kohya_ss version does not expose a cache-latents flag; trainer will encode "
            "latents on the fly."
        )

    command = build_command(args, resolved, train_script, cache_latents_flag)

    printable = " ".join(shlex.quote(token) for token in command)
    log_step("Launching kohya_ss training command:")
    print(printable, flush=True)

    if args.dry_run:
        log_step("Dry run requested; exiting before launching kohya_ss.")
        return 0

    env = os.environ.copy()

    needs_sitecustomize = loader_state is not None or args.allow_truncated_images

    if loader_state is not None:
        env["IMAGETEXT_DATA_LOADER_STATE"] = str(loader_state.path)

    if args.allow_truncated_images:
        env["IMAGETEXT_ALLOW_TRUNCATED_IMAGES"] = "1"

    pythonpath_entries: List[str] = []

    if needs_sitecustomize:
        pythonpath_entries.append(str(Path(__file__).resolve().parent))

    pythonpath_entries.extend(_ADDED_DEPENDENCY_PATHS)

    if pythonpath_entries:
        existing_pythonpath = env.get("PYTHONPATH")
        combined = pythonpath_entries.copy()
        if existing_pythonpath:
            combined.append(existing_pythonpath)
        env["PYTHONPATH"] = ":".join(combined)

    log_step("Starting kohya_ss training subprocess...")
    process = subprocess.run(command, env=env)
    log_step(f"kohya_ss subprocess exited with return code {process.returncode}.")
    if process.returncode != 0:
        print(
            "The training command exited abnormally. Re-run with --dry-run to inspect the "
            "assembled command or execute the printed command manually for more details.",
            file=sys.stderr,
            flush=True,
        )
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
