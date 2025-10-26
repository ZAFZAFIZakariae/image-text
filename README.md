# Image-Text Inference and Fine-Tuning

This repository contains scripts for text-to-image generation and image captioning using Hugging Face Diffusers and Transformers. It includes inference utilities (`run_text2image.py`, `run_imagecaption.py`) as well as a full fine-tuning pipeline (`prepare_text_image_dataset.py`, `fine_tune_text2image.py`).

## Installation

1. Create and activate a Python 3.9+ virtual environment (recommended).
2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

   The requirements closely follow the setup recommended in the Diffusers documentation. Make sure you have a recent version of `pip`. If you are targeting GPU execution (e.g., CUDA 11.8 on Colab), install PyTorch from the official index first using the command shown in the Colab section below so that `pip install -r requirements.txt` reuses that compatible wheel.

## Running Inference

### Text-to-Image Generation

Use the provided script to generate images from prompts:

```bash
python run_text2image.py \
    --prompt "A futuristic cityscape at sunset" \
    --output outputs/text2image.png \
    --model "Animagine XL 3.0" \
    --workflow base-only
```

The ``--model`` flag accepts aliases for the bundled configurations (``stable-diffusion-xl-1.0`` and ``animagine-xl-3.0``) as well as any Hugging Face model identifier. Leaving the option blank uses Stable Diffusion XL 1.0. Regardless of the model, pipelines are automatically moved to CUDA when available, which is strongly recommended for reasonable generation speed and quality.

Control the diffusion pass with ``--workflow``:

- ``auto`` (default) runs the refiner whenever the selected model includes one.
- ``base-only`` forces a single pass through the base pipeline.
- ``base+refiner`` requires the two-stage SDXL workflow and errors if a refiner is unavailable.

### Image Captioning

Generate captions for images with:

```bash
python run_imagecaption.py \
    --image_path path/to/your/image.jpg
```

Captions are saved alongside the input image unless an output path is specified.

## Fine-Tuning

Follow these steps to adapt Stable Diffusion to your own images:

1. **Generate captions for the dataset** – point the helper at a directory that contains the raw JPEG/PNG files. The script uses the BLIP captioning model to create a `metadata.jsonl` manifest with prompts for every image.

   ```bash
   python prepare_text_image_dataset.py \
       --image-dir /path/to/your/images \
       --output-path /path/to/dataset/metadata.jsonl
   ```

   The manifest stores relative file paths, so placing it next to the image directory keeps the structure portable. Re-run the command with `--refresh` if you need to regenerate captions.

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

   To try a different checkpoint, pass `--model` with another alias or Hugging
   Face model ID. To run SDXL without the refiner (which lowers memory usage on
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
