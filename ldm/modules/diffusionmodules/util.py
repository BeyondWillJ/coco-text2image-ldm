"""
Diffusion module utilities: noise schedules, time-step embeddings, and
checkpoint/memory helpers.
"""

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing
    reduced memory at the expense of extra compute in the backward pass.
    """
    if flag:
        from torch.utils.checkpoint import checkpoint as cp
        return cp(func, *inputs, use_reentrant=False)
    else:
        return func(*inputs)


# ---------------------------------------------------------------------------
# Time-step embeddings
# ---------------------------------------------------------------------------

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: 1-D tensor of N indices, one per batch element.
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    Returns:
        (N, dim) tensor of positional embeddings.
    """
    if repeat_only:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    else:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalization(channels):
    """Group Norm with 32 groups (or fewer if channels < 32)."""
    num_groups = min(32, channels)
    while channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups, channels)


# ---------------------------------------------------------------------------
# Zero-module initialisation
# ---------------------------------------------------------------------------

def zero_module(module):
    """Zero out all parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


# ---------------------------------------------------------------------------
# Noise schedule utilities
# ---------------------------------------------------------------------------

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
            torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )
    elif schedule == "cosine":
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
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


def extract_into_tensor(a, t, x_shape):
    """Index tensor *a* at positions *t* and broadcast to *x_shape*."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    if repeat:
        noise = torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    else:
        noise = torch.randn(shape, device=device)
    return noise
