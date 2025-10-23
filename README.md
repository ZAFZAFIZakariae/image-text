# Image-Text Inference and Fine-Tuning

This repository contains scripts for text-to-image generation and image captioning using Hugging Face Diffusers and Transformers. It includes inference utilities (`run_text2image.py`, `run_imagecaption.py`) as well as fine-tuning scripts (`fine_tune_text2image.py`, `fine_tune_image_caption.py`).

## Installation

1. Create and activate a Python 3.9+ virtual environment (recommended).
2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

   The requirements closely follow the setup recommended in the Diffusers documentation, so make sure you have a recent version of `pip` and, if you plan to run Stable Diffusion, the appropriate GPU drivers and CUDA toolkit installed.

## Running Inference

### Text-to-Image Generation

Use the provided script to generate images from prompts:

```bash
python run_text2image.py \
    --prompt "A futuristic cityscape at sunset" \
    --output_dir outputs/text2image
```

By default, Stable Diffusion models are moved to CUDA when available (as recommended by Hugging Face). A GPU is strongly recommended for reasonable generation speed and quality.

### Image Captioning

Generate captions for images with:

```bash
python run_imagecaption.py \
    --image_path path/to/your/image.jpg
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
   !pip install -r requirements.txt
   ```
3. Upload or clone this repository into the notebook environment and run the same commands described above (e.g., `!python run_text2image.py --prompt "A futuristic cityscape at sunset"`).

GPU support dramatically speeds up Stable Diffusion and other diffusion-based models, mirroring the Hugging Face guidance of moving models to CUDA when available.

## Troubleshooting

- Ensure your GPU drivers and CUDA toolkit match the versions required by PyTorch.
- When running locally without a GPU, expect significantly slower inference for Stable Diffusion models.
- For additional options, consult the script-level `--help` flags (e.g., `python run_text2image.py --help`).

## Testing the Setup

After installation, run the inference commands above with sample prompts or images to confirm that everything works correctly. If running in a fresh environment (local or Colab), following these instructions should reproduce the text-to-image and image-captioning outputs.
