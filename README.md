# Image-Text Inference and Fine-Tuning

This repository contains scripts for text-to-image generation and image captioning using Hugging Face Diffusers and Transformers. It includes inference utilities (`run_text2image.py`, `run_imagecaption.py`) as well as a full fine-tuning pipeline (`prepare_text_image_dataset.py`, `fine_tune_text2image.py`).

## Installation

1. Create and activate a Python 3.9+ virtual environment (recommended).
2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

   The requirements include the SDXL refiner dependencies (`accelerate` and `safetensors`) and follow the setup recommended in the Diffusers documentation. Make sure you have a recent version of `pip`. If you are targeting GPU execution (e.g., CUDA 11.8 on Colab), install PyTorch from the official index first using the command shown in the Colab section below so that `pip install -r requirements.txt` reuses that compatible wheel.

## Running Inference

### Text-to-Image Generation

Use the provided script to generate images from prompts:

```bash
python run_text2image.py \
    --prompt "A futuristic cityscape at sunset" \
    --output outputs/text2image.png \
    --model "Animagine XL 4.0" \
    --workflow base-only
```

The ``--model`` flag accepts aliases for the bundled configurations only. The integrated registry now focuses on four SDXL-era checkpoints:

- ``stable-diffusion-xl-1.0`` / ``sdxl`` – base weights with optional refiner and fp16 VAE fix.
- ``realvisxl-v5.0`` / ``realvis-xl-5.0`` – RealVis XL V5.0 with its fused VAE (no refiner stage).
- ``juggernautxl-v10`` / ``juggernaut-xl-v10`` – Juggernaut XL v10 with its bundled VAE.
- ``animagine-xl-4.0`` / ``animagine-xl`` – Animagine XL 4.0 tuned for anime-style renders.

Leaving ``--model`` blank uses Stable Diffusion XL 1.0. Pipelines are automatically moved to CUDA when available, which is strongly recommended for reasonable generation speed and quality.

Control the diffusion pass with ``--workflow``:

- ``auto`` (default) runs the refiner whenever the selected model includes one. SDXL users should ensure the `stabilityai/stable-diffusion-xl-refiner-1.0` checkpoint is accessible (log in with `huggingface_hub` if required) so the workflow can download the weights on first use.
- ``base-only`` forces a single pass through the base pipeline.
- ``base+refiner`` requires the two-stage SDXL workflow and errors if a refiner is unavailable.

#### Advanced SDXL and Diffusers parameters

The CLI exposes the most common Stable Diffusion XL settings directly:

- ``--negative-prompt`` to suppress unwanted concepts.
- ``--num-inference-steps`` to control the denoising iterations.
- ``--guidance-scale`` to adjust classifier-free guidance strength.
- ``--width`` / ``--height`` to override the generated resolution.
- ``--seed`` to produce deterministic results when paired with a fixed prompt.
- ``--refiner-start`` to choose when the SDXL refiner takes over in a two-stage run.

Any additional keyword arguments supported by the underlying diffusers pipeline can be forwarded with ``--pipeline-arg KEY=VALUE``. Repeat the flag to send multiple values (for example, ``--pipeline-arg scheduler=EulerAncestralDiscreteScheduler --pipeline-arg clip_skip=2``). Values are parsed with ``ast.literal_eval`` when possible, so numbers, lists, and booleans can be passed without surrounding quotes.

### Image Captioning

Generate captions for images with:

```bash
python run_imagecaption.py \
    --image_path path/to/your/image.jpg
```

Captions are saved alongside the input image unless an output path is specified.

## Fine-Tuning

Follow these steps to adapt Stable Diffusion to your own images:

### Normalise your dataset images (optional but recommended)

Stable Diffusion XL works best when every training image shares the same
dimensions. The repository now includes a helper that mirrors the recommended
1024×1024 preprocessing pipeline—small images are upscaled with bicubic
interpolation, reasonably sized photos receive a centered smart crop, and
extreme aspect ratios fall back to reflect padding so no important content is
lost.

```bash
python preprocess_images.py \
    /path/to/raw/images \
    --output-dir /path/to/processed/images \
    --strategy auto  # crop unless the aspect ratio is extreme
```

The command preserves your folder structure and writes the processed assets to
the destination directory (omit `--output-dir` to operate in place). Adjust
`--strategy` to `crop` or `pad` if you want to force one behaviour, and use
`--target-size` when you need a square other than 1024×1024.

1. **Generate captions for the dataset** – point the helper at a directory that contains the raw JPEG/PNG files. The script uses the BLIP captioning model to create a `metadata.jsonl` manifest with prompts for every image.

   ```bash
   python prepare_text_image_dataset.py \
       --image-dir /path/to/your/images \
       --output-path /path/to/dataset/metadata.jsonl
   ```

   The manifest stores relative file paths, so placing it next to the image directory keeps the structure portable. Re-run the command with `--refresh` if you need to regenerate captions.

   If you delete or move images after the fact, clean up stale rows so training will not stumble over missing files:

   ```bash
   python prune_missing_manifest_entries.py /path/to/dataset
   ```

   The helper rewrites `metadata.jsonl` in place, keeping only entries whose images still exist next to the manifest.

2. **Launch fine-tuning** – once you have the manifest, call the training script and point it at the dataset folder (containing the images and the newly created `metadata.jsonl`).

   ```bash
   python fine_tune_text2image.py \
       --dataset-path /path/to/dataset \
       --output-dir /path/to/checkpoints \
       --pretrained-model runwayml/stable-diffusion-v1-5 \
       --batch-size 4 \
       --epochs 1 \
       --mixed-precision
   ```

   The script saves checkpoints every `--save-interval` updates and resumes automatically when `--resume` is supplied. Increase `--epochs` and adjust the learning rate, batch size, and gradient accumulation to match your hardware.

### Fine-tuning RealVis XL 5.0 with LyCORIS (LoCon + DoRA)

You can reproduce the Colab workflow below to fine-tune the `realvisxl-v5` base weights with LyCORIS (LoCon + DoRA) adapters using the Kohya trainer. The commands are designed for a fresh Google Colab session with a GPU runtime.

1. **Setup (Colab)** – install the training dependencies and LyCORIS extension packages:

   ```bash
   pip install -q accelerate bitsandbytes==0.43.3 xformers==0.0.27.post2 triton==3.0.0 \
     datasets==2.19.1 peft==0.11.1 einops==0.8.0 pillow==10.3.0 \
     safetensors==0.4.3 transformers==4.44.2 lycoris-lora==2.3.2.dev6 \
     sentencepiece==0.2.0
   pip install -q git+https://github.com/bmaltais/kohya_ss.git@master
   ```

   Use a T4 or L4 GPU runtime and start from a clean session to avoid conflicts with cached wheels.

2. **Paths & basics** – configure the RealVis checkpoint, dataset directory, and output path for the trained adapters:

   ```bash
   BASE_MODEL="/content/realvisxl-v5.safetensors"   # put your RealVis XL v5 here
   DATA_DIR="/content/my60k"                         # images + per-image .txt captions
   OUT_DIR="/content/output_loras"

   mkdir -p "$OUT_DIR"
   ```

   Arrange the dataset so each image has a matching caption file (for example `img00001.jpg` and `img00001.txt`) inside `DATA_DIR`. If your captions are stored in a single file, run a caption splitter to generate one `.txt` file per image before training.

3. **One-pass LyCORIS (LoCon + DoRA) training** – launch the Kohya trainer with the preset that matches your GPU memory budget. The configuration below covers the 60k-image run recommended by the user:

   **L4 (24GB) – better throughput**

   ```bash
   python -u /usr/local/lib/python3.10/dist-packages/kohya_ss/train_network.py \
     --network_module=lycoris.kohya \
     --algo=locon_dora \
     --network_dim=192 --network_alpha=96 --lora_dropout=0.05 \
     --pretrained_model_name_or_path="$BASE_MODEL" \
     --train_data_dir="$DATA_DIR" --caption_extension=".txt" \
     --resolution=1024 --min_bucket_reso=640 --max_bucket_reso=1024 --bucket_reso_steps=64 \
     --output_dir="$OUT_DIR" --output_name="realvisxl_locon_dora_full60k" \
     --learning_rate=1e-4 --text_encoder_lr=5e-6 \
     --optimizer_type=adamw8bit --weight_decay=0.01 --max_grad_norm=1.0 \
     --lr_scheduler=cosine --lr_warmup_ratio=0.05 \
     --train_unet --train_text_encoder \
     --network_train_unet_only=0 \
     --mixed_precision=bf16 --gradient_checkpointing \
     --min_snr_gamma=5.0 --noise_offset=0.02 \
     --max_data_loader_n_workers=8 --persistent_data_loader_workers \
     --cache_latents_to_disk \
     --save_every_n_steps=1000 --save_model_as=safetensors \
     --log_prefix="L4_full60k" \
     --max_train_steps=20000 \
     --train_batch_size=2 --gradient_accumulation_steps=4
   ```

   **T4 (16GB) – tighter memory (smaller batch & more accumulation)**

   ```bash
   python -u /usr/local/lib/python3.10/dist-packages/kohya_ss/train_network.py \
     --network_module=lycoris.kohya \
     --algo=locon_dora \
     --network_dim=192 --network_alpha=96 --lora_dropout=0.05 \
     --pretrained_model_name_or_path="$BASE_MODEL" \
     --train_data_dir="$DATA_DIR" --caption_extension=".txt" \
     --resolution=1024 --min_bucket_reso=640 --max_bucket_reso=1024 --bucket_reso_steps=64 \
     --output_dir="$OUT_DIR" --output_name="realvisxl_locon_dora_full60k" \
     --learning_rate=1e-4 --text_encoder_lr=5e-6 \
     --optimizer_type=adamw8bit --weight_decay=0.01 --max_grad_norm=1.0 \
     --lr_scheduler=cosine --lr_warmup_ratio=0.05 \
     --train_unet --train_text_encoder \
     --network_train_unet_only=0 \
     --mixed_precision=bf16 --gradient_checkpointing \
     --min_snr_gamma=5.0 --noise_offset=0.02 \
     --max_data_loader_n_workers=6 --persistent_data_loader_workers \
     --cache_latents_to_disk \
     --save_every_n_steps=1000 --save_model_as=safetensors \
     --log_prefix="T4_full60k" \
     --max_train_steps=20000 \
     --train_batch_size=1 --gradient_accumulation_steps=8
   ```

`--cache_latents_to_disk` keeps long runs stable on Colab. Monitor the quality of intermediate checkpoints (saved every 1,000 steps) and stop early if your results plateau—many 60k-image runs converge between 16k and 24k steps. Keep both EMA and non-EMA outputs if you plan to perform a full fine-tune later.

### Resuming LyCORIS runs and controlling DataLoader order

Supplying `--resume` alongside a saved checkpoint restores the network weights,
optimizer state, and global step counter so training continues from the exact
step where it stopped. Historically, the PyTorch `DataLoader` would rebuild in a
freshly shuffled order every time, so the resumed run might replay samples near
the resume point. The `train_realvis_locon_dora.py` launcher now installs a
runtime patch that keeps the shuffle order deterministic and skips the
already-seen mini-batches the first time the loader iterates after a resume.

* The helper persists a seed (default `3407`) the first time it runs. Override
  it with `--seed` if you prefer a specific value, or supply
  `--disable-dataloader-state` to fall back to Kohya's native behaviour.
* When resuming, the script infers the completed step count from the checkpoint
  filename (e.g. `*-00012000.safetensors`). If your naming scheme differs, add
  `--resume-step N` so the loader can skip the correct number of samples. The
  number of skipped samples accounts for gradient accumulation
  (`train_batch_size × gradient_accumulation_steps`).
* The temporary state file lives next to the output directory. Deleting it is a
  quick way to reseed the shuffle order between experiments.

With those safeguards in place you can resume training by simply pointing
`--resume` at the latest checkpoint—the command will continue from the next
unseen sample while keeping the optimiser state intact.

If you prefer to store the configuration locally and trigger the same training command from this repository, use the helper script:

```bash
python train_realvis_locon_dora.py \
    --base-model /content/realvisxl-v5.safetensors \
    --data-dir /content/my60k \
    --output-dir /content/output_loras \
    --preset L4
```

Swap `--preset` to `T4` for the lower-memory schedule or add overrides such as `--max-train-steps 24000` or `--train-batch-size 1`. Pass `--dry-run` to print the composed `kohya_ss.train_network` invocation without launching it.

## Running on Google Colab

The repo runs well inside a single Colab notebook cell sequence. Use the steps
below to get the SDXL base+refiner workflow running on a GPU runtime:

1. **Enable a GPU** – open **Runtime → Change runtime type** and select
   **GPU**.
2. **Clone the repository** (or upload it manually) and install dependencies.
   The snippet below keeps everything inside `/content` so the rest of the
   commands work verbatim:

   ```python
   !git clone https://github.com/<your-account>/image-text.git
   %cd image-text

   !pip install --upgrade pip
   !pip install torch --index-url https://download.pytorch.org/whl/cu118
   !pip install -r requirements.txt
   ```

   Installing PyTorch first guarantees that Colab downloads a CUDA-enabled
   wheel (matching the hosted GPU). The requirements file then installs the
   Diffusers stack used by the pipelines.
3. **Authenticate with Hugging Face (optional but recommended).** SDXL weights
   live behind the Stability AI license gate, so run the following in a cell if
   your account needs a token to download the checkpoints:

   ```python
   from huggingface_hub import login
   login()
   ```

   Paste your token when prompted. You can skip this step if you already have
   the models cached in your Colab session or are using public checkpoints.
4. **Generate an image.** Choose the workflow that fits your needs. The example
   below runs the default two-stage SDXL pipeline with refiner:

   ```python
   !python run_text2image.py \
       --prompt "A futuristic cityscape at sunset" \
       --output outputs/sdxl_refined.png \
       --workflow base+refiner
   ```

   To try a different bundled checkpoint, pass `--model` with one of the supported
   aliases above. To run SDXL without the refiner (which lowers memory usage on
   smaller GPUs), switch the workflow flag:

   ```python
   !python run_text2image.py \
       --prompt "A futuristic cityscape at sunset" \
       --output outputs/sdxl_base.png \
       --workflow base-only
   ```

### Removing numbered duplicates on Colab

To clean a dataset stored on Google Drive:

1. Mount your Drive inside the notebook:

   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```

2. Change into the repository directory (clone or upload it first) and run the utility against your dataset folder:

   ```bash
   %cd /content/path/to/image-text
   !python remove_numbered_duplicates.py \
       /content/drive/MyDrive/text2imagedataset/text2imagedataset
   ```

   Replace the path with the directory you want to clean. The script prints how many files it deleted, renamed, or kept. You can re-run it safely; once duplicates are gone it will simply report zero deletions.

GPU support dramatically speeds up Stable Diffusion and other diffusion-based models, mirroring the Hugging Face guidance of moving models to CUDA when available.

## Troubleshooting

- Ensure your GPU drivers and CUDA toolkit match the versions required by PyTorch.
- When running locally without a GPU, expect significantly slower inference for Stable Diffusion models.
- For additional options, consult the script-level `--help` flags (e.g., `python run_text2image.py --help`).

## Testing the Setup

After installation, run the inference commands above with sample prompts or images to confirm that everything works correctly. If running in a fresh environment (local or Colab), following these instructions should reproduce the text-to-image and image-captioning outputs.
