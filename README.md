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
    --model "Animagine XL 3.0"
```

The ``--model`` flag accepts aliases for the bundled configurations (``stable-diffusion-xl-1.0`` and ``animagine-xl-3.0``) as well as any Hugging Face model identifier. Leaving the option blank uses Stable Diffusion XL 1.0. Regardless of the model, pipelines are automatically moved to CUDA when available, which is strongly recommended for reasonable generation speed and quality.

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

If you prefer to experiment in Google Colab:

1. Open a new notebook and enable a GPU runtime via **Runtime → Change runtime type → Hardware accelerator → GPU**.
2. In a cell, install the dependencies:

   ```python
   !pip install --upgrade pip
   !pip install torch --index-url https://download.pytorch.org/whl/cu118
   !pip install -r requirements.txt
   ```

   The separate `torch` installation ensures that Colab (or any CUDA-enabled runtime) receives a current GPU build before the
   remaining packages are installed. The requirement file now accepts any modern PyTorch release (2.2 or newer), so the command
   above will stay compatible as PyTorch publishes new wheels.

3. Upload or clone this repository into the notebook environment and run the same commands described above (e.g., `!python run_text2image.py --prompt "A futuristic cityscape at sunset"`).

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
