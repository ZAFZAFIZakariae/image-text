"""End-to-end text-to-image fine-tuning entry point.

This module upgrades the original placeholder so it can fine-tune a Stable
Diffusion style pipeline using paired text/image data stored in a JSON Lines
manifest.  The implementation intentionally mirrors the official Diffusers
examples but keeps the dependencies minimal so it can run in a vanilla PyTorch
environment without :mod:`accelerate`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


@dataclass
class Sample:
    """Single text-image training example."""

    image_path: Path
    text: str


class TextImageDataset(Dataset):
    """Dataset that loads text prompts and their corresponding images."""

    def __init__(
        self,
        samples: Sequence[Sample],
        tokenizer: CLIPTokenizer,
        image_size: int,
        center_crop: bool,
    ) -> None:
        self._samples = list(samples)
        self._tokenizer = tokenizer
        self._image_size = image_size
        self._center_crop = center_crop

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._samples)

    def _process_image(self, path: Path) -> torch.FloatTensor:
        image = Image.open(path).convert("RGB")

        if self._center_crop:
            width, height = image.size
            min_dimension = min(width, height)
            left = (width - min_dimension) // 2
            top = (height - min_dimension) // 2
            image = image.crop((left, top, left + min_dimension, top + min_dimension))

        image = image.resize((self._image_size, self._image_size), resample=Image.BICUBIC)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = image_tensor * 2.0 - 1.0
        return image_tensor

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self._samples[index]

        pixel_values = self._process_image(sample.image_path)
        tokenized = self._tokenizer(
            sample.text,
            padding="max_length",
            truncation=True,
            max_length=self._tokenizer.model_max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
        }


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the fine-tuning script."""

    parser = argparse.ArgumentParser(
        description="Fine-tune a Stable Diffusion style text-to-image model on custom data."
    )

    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help=(
            "Path to a directory containing images and a metadata.jsonl file. "
            "Each line of the manifest must include 'file' and 'text' keys."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where checkpoints and logs will be written.",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Model identifier or path to use as the fine-tuning starting point.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer (default: 1e-4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of samples per batch (default: 4).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs to run (default: 1).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=25,
        help="How often to log training metrics in optimizer steps (default: 25).",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=500,
        help="How often to save checkpoints in optimizer steps (default: 500).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before an optimizer step.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training when possible.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the most recent checkpoint in --output-dir.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Resolution to resize training images to (default: 512).",
    )
    parser.add_argument(
        "--center-crop",
        action="store_true",
        help="Crop images to square before resizing (default: off).",
    )

    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_manifest(dataset_path: Path) -> List[Sample]:
    manifest_path = dataset_path / "metadata.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Could not locate metadata.jsonl in {dataset_path}. "
            "Use prepare_text_image_dataset.py to generate it."
        )

    samples: List[Sample] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            try:
                image_path = dataset_path / record["file"]
                text = record["text"]
            except KeyError as exc:
                raise KeyError(
                    f"Malformed entry on line {line_number} of {manifest_path}: missing {exc.args[0]!r}"
                ) from exc

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image {image_path} referenced on line {line_number} does not exist."
                )
            samples.append(Sample(image_path=image_path, text=text))

    if not samples:
        raise ValueError(f"Manifest {manifest_path} did not contain any records.")

    return samples


def _save_training_state(
    output_dir: Path,
    step: int,
    unet: UNet2DConditionModel,
    optimizer: AdamW,
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-step-{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    unet.save_pretrained(checkpoint_dir / "unet")
    torch.save({"optimizer": optimizer.state_dict(), "step": step}, checkpoint_dir / "training_state.pt")

    latest_path = output_dir / "latest.txt"
    latest_path.write_text(str(step), encoding="utf-8")


def _load_latest_checkpoint(output_dir: Path) -> Optional[tuple[Path, int]]:
    latest_path = output_dir / "latest.txt"
    if latest_path.exists():
        step = int(latest_path.read_text(encoding="utf-8").strip())
        checkpoint_dir = output_dir / f"checkpoint-step-{step:06d}"
        if checkpoint_dir.exists():
            return checkpoint_dir, step
    checkpoints = sorted(output_dir.glob("checkpoint-step-*/unet"))
    if checkpoints:
        checkpoint_dir = checkpoints[-1].parent
        step = int(checkpoint_dir.name.split("-")[-1])
        return checkpoint_dir, step
    return None


def main(args: argparse.Namespace) -> None:
    """Fine-tune a Stable Diffusion UNet using paired text and images."""

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    samples = _load_manifest(args.dataset_path)
    dataset = TextImageDataset(samples, tokenizer, args.image_size, args.center_crop)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet").to(device)

    checkpoint_info = None
    if args.resume:
        checkpoint_info = _load_latest_checkpoint(args.output_dir)
        if checkpoint_info is None:
            raise FileNotFoundError(
                "--resume was specified but no checkpoints were found in the output directory."
            )
        checkpoint_dir, _ = checkpoint_info
        unet = UNet2DConditionModel.from_pretrained(checkpoint_dir / "unet").to(device)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    optimizer = AdamW(unet.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision and torch.cuda.is_available())

    global_step = 0
    if checkpoint_info is not None:
        checkpoint_dir, global_step = checkpoint_info
        state_path = checkpoint_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            global_step = state.get("step", global_step)

    unet.train()

    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_steps = updates_per_epoch * args.epochs

    progress_bar = tqdm(total=total_steps, desc="Training", leave=True)
    progress_bar.update(global_step)

    optimizer.zero_grad(set_to_none=True)
    accumulation_counter = 0

    for epoch in range(args.epochs):
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device=device, dtype=vae.dtype)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (pixel_values.shape[0],),
                device=device,
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(input_ids, attention_mask=attention_mask)[0]

            with torch.cuda.amp.autocast(enabled=args.mixed_precision and torch.cuda.is_available()):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

            accumulation_counter += 1
            if accumulation_counter % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                accumulation_counter = 0
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item() * args.gradient_accumulation_steps})

                if global_step % args.log_interval == 0:
                    print(
                        f"Step {global_step}: loss={loss.item() * args.gradient_accumulation_steps:.4f}"
                    )

                if global_step % args.save_interval == 0:
                    _save_training_state(args.output_dir, global_step, unet, optimizer)

        progress_bar.refresh()

    if global_step == 0 or global_step % args.save_interval != 0:
        _save_training_state(args.output_dir, global_step, unet, optimizer)
    progress_bar.close()


if __name__ == "__main__":
    parser = build_parser()
    arguments = parser.parse_args()
    main(arguments)
