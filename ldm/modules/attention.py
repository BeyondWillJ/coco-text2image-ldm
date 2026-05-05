"""
Attention mechanisms used by the UNet:
  - Self-attention
  - Cross-attention (for text conditioning)
  - Feed-forward network
  - Transformer block
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GEGLU gating."""

    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x):
        return self.net(x)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention.  When context is None it degrades to
    self-attention.
    """

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        # split heads
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if mask is not None:
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat_interleave(mask, h)
            sim.masked_fill_(~mask[:, None, :], max_neg_value)

        attn = sim.softmax(dim=-1)
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


def repeat_interleave(x, repeats):
    return x.repeat_interleave(repeats, dim=0)


class BasicTransformerBlock(nn.Module):
    """
    One Transformer block: self-attention → cross-attention → feed-forward,
    each with a pre-norm residual connection.
    """

    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )
        self.ff = FeedForward(dim, dropout=dropout)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data (B, C, H, W).
    Flattens spatial dims, applies `depth` transformer blocks, then reshapes.
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(min(32, in_channels), in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for _ in range(depth)
            ]
        )
        self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
