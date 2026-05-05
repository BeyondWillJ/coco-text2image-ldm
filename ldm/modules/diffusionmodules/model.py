"""
UNet backbone for the latent diffusion model.

Architecture:
  Encoder path  → skip connections → Decoder path
  with ResNet blocks, (optional) self/cross-attention, and time conditioning.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    normalization,
    timestep_embedding,
    zero_module,
)


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class Upsample(nn.Module):
    """2× nearest-neighbour upsample followed by an optional convolution."""

    def __init__(self, channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """2× strided convolution (or average-pool) downsample."""

    def __init__(self, channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        if self.with_conv:
            return self.conv(x)
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class ResnetBlock(nn.Module):
    """ResNet block with time-step and optional dropout."""

    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.0, temb_channels=512):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = normalization(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        if in_channels != out_channels:
            if conv_shortcut:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb=None):
        h = self.conv1(F.silu(self.norm1(x)))
        if temb is not None and hasattr(self, "temb_proj"):
            h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    Spatial self-attention block (single-head, efficient implementation).
    Used as a simpler alternative to SpatialTransformer in lower-res layers.
    """

    def __init__(self, channels, num_heads=1, num_head_channels=-1):
        super().__init__()
        self.channels = channels
        if num_head_channels != -1:
            num_heads = channels // num_head_channels
        self.num_heads = num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x).view(b, c, -1)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        # reshape for multi-head
        head_dim = c // self.num_heads
        q = q.reshape(b * self.num_heads, head_dim, h * w)
        k = k.reshape(b * self.num_heads, head_dim, h * w)
        v = v.reshape(b * self.num_heads, head_dim, h * w)
        scale = head_dim ** -0.5
        attn = torch.einsum("b d i, b d j -> b i j", q * scale, k).softmax(dim=-1)
        out = torch.einsum("b i j, b d j -> b d i", attn, v)
        out = out.reshape(b, c, h * w)
        out = self.proj_out(out)
        return (x_in + out.reshape(b, c, h, w))


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

def make_attn(in_channels, attn_type, context_dim=None, num_heads=8, head_dim=32):
    """Factory for attention blocks."""
    if attn_type == "vanilla":
        return AttentionBlock(in_channels)
    elif attn_type == "linear":
        return SpatialTransformer(
            in_channels, num_heads, head_dim, depth=1, context_dim=context_dim
        )
    elif attn_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")


class UNetModel(nn.Module):
    """
    The full UNet model with time-step embedding and optional cross-attention
    for text conditioning.

    Args:
        image_size:        Spatial size of the latent (used for positional info).
        in_channels:       Number of input channels (latent space channels).
        model_channels:    Base channel width.
        out_channels:      Number of output channels (usually == in_channels).
        num_res_blocks:    Number of residual blocks per resolution level.
        attention_resolutions: Downsample factors at which to add attention.
        dropout:           Dropout probability.
        channel_mult:      Channel multipliers for each resolution level.
        num_heads:         Number of attention heads.
        context_dim:       Dimension of the text context vector (for cross-attn).
        use_spatial_transformer: Use SpatialTransformer instead of AttentionBlock.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        num_classes=None,
        use_checkpoint=False,
        num_heads=8,
        num_head_channels=-1,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        attn_type="vanilla",
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.context_dim = context_dim

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        if use_spatial_transformer:
            attn_type = "linear"

        self.input_blocks = nn.ModuleList(
            [nn.ModuleList([nn.Conv2d(in_channels, model_channels, 3, padding=1)])]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResnetBlock(in_channels=ch, out_channels=mult * model_channels,
                                      dropout=dropout, temb_channels=time_embed_dim)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if use_spatial_transformer:
                        layers.append(
                            SpatialTransformer(ch, num_heads, ch // num_heads,
                                               depth=transformer_depth, context_dim=context_dim)
                        )
                    else:
                        layers.append(AttentionBlock(ch, num_heads=num_heads,
                                                      num_head_channels=num_head_channels))
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    nn.ModuleList([Downsample(ch, conv_resample)])
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = nn.ModuleList([
            ResnetBlock(in_channels=ch, out_channels=ch,
                        dropout=dropout, temb_channels=time_embed_dim),
            (SpatialTransformer(ch, num_heads, ch // num_heads,
                                depth=transformer_depth, context_dim=context_dim)
             if use_spatial_transformer
             else AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels)),
            ResnetBlock(in_channels=ch, out_channels=ch,
                        dropout=dropout, temb_channels=time_embed_dim),
        ])

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResnetBlock(in_channels=ch + ich, out_channels=mult * model_channels,
                                      dropout=dropout, temb_channels=time_embed_dim)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if use_spatial_transformer:
                        layers.append(
                            SpatialTransformer(ch, num_heads, ch // num_heads,
                                               depth=transformer_depth, context_dim=context_dim)
                        )
                    else:
                        layers.append(AttentionBlock(ch, num_heads=num_heads,
                                                      num_head_channels=num_head_channels))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.output_blocks.append(nn.ModuleList(layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, out_channels, 3, padding=1)),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_module_list(self, module_list, h, emb, context=None):
        for module in module_list:
            if isinstance(module, (ResnetBlock,)):
                h = module(h, emb)
            elif isinstance(module, (SpatialTransformer,)):
                h = module(h, context)
            else:
                h = module(h)
        return h

    def forward(self, x, timesteps, context=None, y=None):
        """
        Args:
            x:          (B, C, H, W) latent input.
            timesteps:  (B,) integer time steps.
            context:    (B, seq_len, context_dim) text conditioning, or None.
            y:          (B,) class labels (if num_classes is set).
        Returns:
            (B, out_channels, H, W) predicted noise.
        """
        assert (y is not None) == (self.num_classes is not None)

        hs = []
        temb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if y is not None:
            temb = temb + self.label_emb(y)

        h = x
        for module_list in self.input_blocks:
            h = self._forward_module_list(module_list, h, temb, context)
            hs.append(h)

        h = self._forward_module_list(self.middle_block, h, temb, context)

        for module_list in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = self._forward_module_list(module_list, h, temb, context)

        return self.out(h)
