"""
Utility functions for the LDM package.
"""

import importlib
import os
from functools import partial

import numpy as np
import torch
from PIL import Image


def get_obj_from_str(string, reload=False):
    """Instantiate a class or function from a dot-separated string path."""
    module_name, cls_name = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module_name)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module_name), cls_name)


def instantiate_from_config(config):
    """Instantiate an object from an OmegaConf config with a ``target`` key."""
    if "target" not in config:
        raise KeyError("Expected key 'target' to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", {}))


def log_txt_as_img(wh, xc, size=10):
    """Render a list of text strings as a batch of images for logging."""
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        from PIL import ImageDraw

        draw = ImageDraw.Draw(txt)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(
            xc[bi][start : start + nc] for start in range(0, len(xc[bi]), nc)
        )
        draw.text((0, 0), lines, fill="black")
        txt = np.array(txt) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts, dtype=torch.float32).permute(0, 3, 1, 2)
    return txts


def ismap(x):
    """Return True if x is a 4-D segmentation map (C > 3)."""
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    """Return True if x looks like an image tensor."""
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] in (1, 3))


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def mean_flat(tensor):
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """Return a 1-D tensor of betas for the given noise schedule."""
    if schedule == "linear":
        betas = (
            torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )
    elif schedule == "cosine":
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = torch.clamp(betas, min=0, max=0.999)
    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")
    return betas.numpy()
