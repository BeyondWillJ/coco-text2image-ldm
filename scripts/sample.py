"""
Sampling / inference script for the COCO LDM.

Generates images from text prompts using a trained LDM checkpoint.

Usage::

    python scripts/sample.py \\
        --config configs/coco_ldm.yaml \\
        --ckpt  checkpoints/ldm/ldm-coco-best.ckpt \\
        --prompt "a red double-decker bus on a city street" \\
        --n_samples 4 \\
        --ddim_steps 50 \\
        --scale 7.5 \\
        --outdir outputs/
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from tqdm import trange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldm.util import instantiate_from_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    return model


def tensor_to_pil(t):
    """Convert a (C, H, W) tensor in [-1, 1] to a PIL image."""
    t = (t.clamp(-1, 1) + 1.0) * 0.5  # [0, 1]
    t = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(t)


def save_grid(images, path, nrow=4):
    """Save a list of PIL images as a grid."""
    import math
    n = len(images)
    ncols = min(n, nrow)
    nrows = math.ceil(n / ncols)
    w, h = images[0].size
    grid = Image.new("RGB", (ncols * w, nrows * h))
    for idx, img in enumerate(images):
        row, col = divmod(idx, ncols)
        grid.paste(img, (col * w, row * h))
    grid.save(path)
    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Sample images from a trained COCO LDM")
    p.add_argument("--config", "-c", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--prompt", "-p", type=str, default="a photo of a cat sitting on a sofa")
    p.add_argument("--prompts_file", type=str, default=None,
                   help="Text file with one prompt per line (overrides --prompt)")
    p.add_argument("--n_samples", type=int, default=4,
                   help="Number of samples per prompt")
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--scale", type=float, default=7.5,
                   help="Classifier-free guidance scale")
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading model from {args.ckpt} …")
    model = load_model(args.config, args.ckpt, args.device)

    # Collect prompts
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [args.prompt]

    with torch.no_grad():
        with model.ema_scope():
            for p_idx, prompt in enumerate(prompts):
                print(f"[{p_idx + 1}/{len(prompts)}] Prompt: {prompt!r}")
                batch_prompts = [prompt] * args.n_samples

                # Encode text
                c = model.get_learned_conditioning(batch_prompts)

                # Sample latents
                samples = model.sample_log(
                    cond=c,
                    batch_size=args.n_samples,
                    ddim=True,
                    ddim_steps=args.ddim_steps,
                    guidance_scale=args.scale,
                )

                # Decode to pixel space
                x_samples = model.decode_first_stage(samples)

                # Save images
                pil_images = [tensor_to_pil(x) for x in x_samples]
                safe_prompt = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:60]
                grid_path = os.path.join(args.outdir, f"{p_idx:04d}_{safe_prompt}.png")
                grid = save_grid(pil_images, grid_path)
                print(f"  Saved grid → {grid_path}")

                # Also save individual images
                for i, img in enumerate(pil_images):
                    img.save(os.path.join(args.outdir, f"{p_idx:04d}_{safe_prompt}_{i}.png"))

    print("Done.")


if __name__ == "__main__":
    main()
