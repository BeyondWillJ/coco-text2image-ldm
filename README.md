# coco-text2image-ldm

A **Latent Diffusion Model (LDM)** for text-to-image generation trained on the [MS-COCO](https://cocodataset.org/) dataset.

The model operates entirely in the compressed latent space of a pre-trained KL-regularised VAE and conditions the UNet denoiser on text via a frozen [CLIP](https://openai.com/research/clip) (ViT-L/14) text encoder.

---

## Architecture

```
Text prompt
    │
    ▼
FrozenCLIPEmbedder  ──► cross-attention context (B × 77 × 768)
                                │
Image (256×256 RGB)             │
    │                           │
    ▼                           │
AutoencoderKL (VAE)             │
  Encoder                       │
    │                           │
    ▼                           ▼
  Latent z  (B × 4 × 32 × 32)
    │
    ├─── Forward diffusion: add noise ──► z_t
    │
    └─── UNetModel (denoiser, cross-attn on CLIP embedding)
              │
              ▼
         Predicted noise ε
              │
              ▼
    DDPM / DDIM reverse sampling
              │
              ▼
    Clean latent ẑ  (B × 4 × 32 × 32)
              │
              ▼
AutoencoderKL  Decoder
              │
              ▼
    Generated image (B × 3 × 256 × 256)
```

Key components:

| Module | Description |
|--------|-------------|
| `ldm/models/autoencoder.py` | KL-VAE with ResNet encoder/decoder and self-attention |
| `ldm/models/diffusion/ddpm.py` | DDPM base + LDM wrapper with classifier-free guidance |
| `ldm/modules/diffusionmodules/model.py` | UNet with spatial transformer blocks |
| `ldm/modules/attention.py` | Self-attention, cross-attention, and feed-forward blocks |
| `ldm/modules/encoders/modules.py` | Frozen CLIP text encoder |
| `ldm/modules/ema.py` | Exponential Moving Average for model weights |
| `ldm/data/coco.py` | MS-COCO dataset loader |

---

## Installation

```bash
git clone https://github.com/BeyondWillJ/coco-text2image-ldm.git
cd coco-text2image-ldm
pip install -r requirements.txt
# or in development mode:
pip install -e .
```

Python ≥ 3.9 and PyTorch ≥ 2.0 are required.

---

## Data preparation

Download the 2017 COCO images and captions:

```bash
mkdir -p data/coco/images data/coco/annotations

# Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d data/coco/

# Images (≈18 GB train, ≈1 GB val)
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip -d data/coco/images/
unzip val2017.zip   -d data/coco/images/
```

Expected layout:

```
data/coco/
  annotations/
    captions_train2017.json
    captions_val2017.json
  images/
    train2017/   (118 287 images)
    val2017/     (5 000 images)
```

---

## Training

### Step 1 – Train the VAE

```bash
python scripts/train.py \
    --config configs/autoencoder.yaml \
    --name kl_vae
```

### Step 2 – Train the LDM

```bash
python scripts/train.py \
    --config configs/coco_ldm.yaml \
    --name coco_ldm \
    --first_stage_ckpt checkpoints/autoencoder/autoencoder.ckpt
```

Useful flags:

| Flag | Description |
|------|-------------|
| `--config` | Path to YAML config |
| `--name` | Experiment name suffix |
| `--base_lr` | Override learning rate |
| `--batch_size` | Override batch size |
| `--gpus` | Number of GPUs |
| `--resume` | Resume from checkpoint |
| `--seed` | Random seed (default 42) |

---

## Sampling

```bash
python scripts/sample.py \
    --config configs/coco_ldm.yaml \
    --ckpt   checkpoints/ldm/ldm-coco-best.ckpt \
    --prompt "a red double-decker bus on a rainy city street" \
    --n_samples 4 \
    --ddim_steps 50 \
    --scale 7.5 \
    --outdir outputs/
```

To generate from multiple prompts stored in a file (one per line):

```bash
python scripts/sample.py \
    --config configs/coco_ldm.yaml \
    --ckpt   checkpoints/ldm/ldm-coco-best.ckpt \
    --prompts_file prompts.txt \
    --n_samples 4 \
    --outdir outputs/
```

Sampling flags:

| Flag | Description |
|------|-------------|
| `--prompt` | Single text prompt |
| `--prompts_file` | File with one prompt per line |
| `--n_samples` | Samples per prompt (default 4) |
| `--ddim_steps` | DDIM denoising steps (default 50) |
| `--scale` | Classifier-free guidance scale (default 7.5) |
| `--outdir` | Output directory |
| `--seed` | Random seed |

---

## Configuration

All hyperparameters are controlled via YAML files in `configs/`:

| File | Purpose |
|------|---------|
| `configs/autoencoder.yaml` | KL-VAE training |
| `configs/coco_ldm.yaml` | LDM training |

---

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

15 unit tests cover the VAE, UNet, and dataset loader.

---

## Project structure

```
coco-text2image-ldm/
├── configs/
│   ├── autoencoder.yaml        # VAE training config
│   └── coco_ldm.yaml           # LDM training config
├── ldm/
│   ├── data/
│   │   └── coco.py             # COCO dataset + DataModule
│   ├── models/
│   │   ├── autoencoder.py      # KL-regularised VAE
│   │   └── diffusion/
│   │       └── ddpm.py         # DDPM base + LatentDiffusionModel
│   ├── modules/
│   │   ├── attention.py        # Cross-attention & SpatialTransformer
│   │   ├── diffusionmodules/
│   │   │   ├── model.py        # UNet backbone
│   │   │   └── util.py         # Timestep embeddings, noise schedule utils
│   │   ├── ema.py              # Exponential Moving Average
│   │   └── encoders/
│   │       └── modules.py      # FrozenCLIPEmbedder
│   └── util.py                 # Shared utilities
├── scripts/
│   ├── train.py                # Training entry-point
│   └── sample.py               # Inference / sampling entry-point
├── tests/
│   ├── test_autoencoder.py
│   ├── test_dataset.py
│   └── test_unet.py
├── requirements.txt
└── setup.py
```

---

## References

- Rombach et al., [*High-Resolution Image Synthesis with Latent Diffusion Models*](https://arxiv.org/abs/2112.10752) (CVPR 2022)
- Ho et al., [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239) (NeurIPS 2020)
- Song et al., [*Denoising Diffusion Implicit Models*](https://arxiv.org/abs/2010.02502) (ICLR 2021)
- Radford et al., [*Learning Transferable Visual Models From Natural Language Supervision*](https://arxiv.org/abs/2103.00020) (ICML 2021)
- Lin et al., [*Microsoft COCO: Common Objects in Context*](https://arxiv.org/abs/1405.0312) (ECCV 2014)
