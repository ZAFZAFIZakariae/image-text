# Image-Text Inference and Fine-Tuning

This repository contains scripts for text-to-image generation and image captioning using Hugging Face Diffusers and Transformers. It includes inference utilities (`run_text2image.py`, `run_imagecaption.py`) as well as fine-tuning scripts (`fine_tune_text2image.py`, `fine_tune_image_caption.py`).

## Installation

1. Create and activate a Python 3.9+ virtual environment (recommended).
2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

   The requirement pins were relaxed so that modern Python runtimes (including Colab's Python 3.12 images) can resolve wheels that Hugging Face and PyTorch publish over time. Make sure you have a recent version of `pip`. If you are targeting GPU execution (e.g., CUDA 11.8 on Colab), install PyTorch from the official index first using the command shown in the Colab section below so that `pip install -r requirements.txt` reuses that compatible wheel.

   Only the libraries used by the checked-in scripts remain in `requirements.txt`; optional extras such as SciPy or Accelerate were dropped to avoid version conflicts on newer Python releases.

## Running Inference

### Text-to-Image Generation

Use the provided script to generate images from prompts:

```bash
python run_text2image.py \
    --prompt "A futuristic cityscape at sunset" \
    --output outputs/text2image.png
```

By default, Stable Diffusion models are moved to CUDA when available (as recommended by Hugging Face). A GPU is strongly recommended for reasonable generation speed and quality.

### Image Captioning

Generate captions for images with:

```bash
python run_imagecaption.py path/to/your/image.jpg
```

Captions are saved alongside the input image unless an output path is specified.

## Fine-Tuning

For advanced users, the repository includes starter scripts for fine-tuning both text-to-image and image captioning models:

```bash
python fine_tune_text2image.py --config configs/text2image_config.yaml
python fine_tune_image_caption.py --config configs/image_caption_config.yaml
```

Edit the configuration files to point to your datasets, model checkpoints, and training hyperparameters.

## Running on Google Colab

If you prefer to experiment in Google Colab:

1. Open a new notebook and enable a GPU runtime via **Runtime → Change runtime type → Hardware accelerator → GPU**.
2. In a cell, install the dependencies:

   ```python
   !pip install --upgrade pip
   !pip install torch --index-url https://download.pytorch.org/whl/cu118
   !pip install -r requirements.txt
   ```

   Installing `torch` first ensures that Colab (or any CUDA-enabled runtime) receives a GPU build matched to the available CUDA
   toolkit. Because `requirements.txt` now specifies minimum versions instead of exact pins, the follow-up install step reuses
   that wheel and resolves compatible versions of the Hugging Face libraries even as their upstream projects cut new releases.

3. Upload or clone this repository into the notebook environment and run the same commands described above (e.g., `!python run_text2image.py --prompt "A futuristic cityscape at sunset"`).

GPU support dramatically speeds up Stable Diffusion and other diffusion-based models, mirroring the Hugging Face guidance of moving models to CUDA when available.

## Troubleshooting

- Ensure your GPU drivers and CUDA toolkit match the versions required by PyTorch.
- When running locally without a GPU, expect significantly slower inference for Stable Diffusion models.
- For additional options, consult the script-level `--help` flags (e.g., `python run_text2image.py --help`).

## Testing the Setup

After installation, run the inference commands above with sample prompts or images to confirm that everything works correctly. If running in a fresh environment (local or Colab), following these instructions should reproduce the text-to-image and image-captioning outputs.
