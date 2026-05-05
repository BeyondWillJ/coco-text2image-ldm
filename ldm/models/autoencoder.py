"""
Variational Autoencoder (VAE) for encoding images into a lower-dimensional
latent space and decoding latents back to images.

Architecture mirrors the one used in the original LDM paper (KL-regularised).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import normalization, zero_module


# ---------------------------------------------------------------------------
# Residual & attention blocks (VAE-specific, no time embedding)
# ---------------------------------------------------------------------------

class ResnetBlockVAE(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = normalization(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class AttnBlock(nn.Module):
    """Single-head spatial self-attention for the VAE bottleneck."""

    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalization(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        b, c, H, W = q.shape
        q = q.reshape(b, c, H * W).permute(0, 2, 1)  # (B, HW, C)
        k = k.reshape(b, c, H * W)                    # (B, C, HW)
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = attn.softmax(dim=-1)
        v = v.reshape(b, c, H * W).permute(0, 2, 1)  # (B, HW, C)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, H, W)
        return x + self.proj_out(out)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Convolutional encoder: image → latent mean/log-variance.

    Args:
        in_channels:   Number of image channels (usually 3).
        ch:            Base channel width.
        ch_mult:       Channel multipliers for each resolution level.
        num_res_blocks: Residual blocks per level.
        attn_resolutions: Level indices at which to add self-attention.
        dropout:       Dropout rate.
        z_channels:    Latent channel dimension.
        double_z:      If True, output 2×z_channels (mean + log-var).
    """

    def __init__(
        self,
        in_channels=3,
        ch=128,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        z_channels=4,
        double_z=True,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)
        self.down = nn.ModuleList()
        in_ch = ch
        resolution = 256  # tracked for attention gating
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            block = nn.ModuleList()
            for _ in range(num_res_blocks):
                block.append(ResnetBlockVAE(in_channels=in_ch, out_channels=out_ch, dropout=dropout))
                in_ch = out_ch
                if resolution in attn_resolutions:
                    block.append(AttnBlock(in_ch))
            level = nn.Module()
            level.block = block
            level.downsample = (
                nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
                if i < len(ch_mult) - 1
                else nn.Identity()
            )
            self.down.append(level)
            if i < len(ch_mult) - 1:
                resolution //= 2

        self.mid = nn.ModuleList([
            ResnetBlockVAE(in_channels=in_ch, out_channels=in_ch, dropout=dropout),
            AttnBlock(in_ch),
            ResnetBlockVAE(in_channels=in_ch, out_channels=in_ch, dropout=dropout),
        ])
        self.norm_out = normalization(in_ch)
        out_ch = 2 * z_channels if double_z else z_channels
        self.conv_out = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for level in self.down:
            for module in level.block:
                h = module(h)
            h = level.downsample(h)
        for module in self.mid:
            h = module(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Convolutional decoder: latent → image.
    """

    def __init__(
        self,
        out_channels=3,
        ch=128,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        z_channels=4,
    ):
        super().__init__()
        in_ch = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(z_channels, in_ch, 3, padding=1)

        self.mid = nn.ModuleList([
            ResnetBlockVAE(in_channels=in_ch, out_channels=in_ch, dropout=dropout),
            AttnBlock(in_ch),
            ResnetBlockVAE(in_channels=in_ch, out_channels=in_ch, dropout=dropout),
        ])

        self.up = nn.ModuleList()
        resolution = 16  # bottom resolution
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            block = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlockVAE(in_channels=in_ch, out_channels=out_ch, dropout=dropout))
                in_ch = out_ch
                if resolution in attn_resolutions:
                    block.append(AttnBlock(in_ch))
            level = nn.Module()
            level.block = block
            level.upsample = (
                nn.Sequential(
                    nn.Upsample(scale_factor=2.0, mode="nearest"),
                    nn.Conv2d(in_ch, in_ch, 3, padding=1),
                )
                if i > 0
                else nn.Identity()
            )
            self.up.append(level)
            if i > 0:
                resolution *= 2

        self.norm_out = normalization(in_ch)
        self.conv_out = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        for module in self.mid:
            h = module(h)
        for level in self.up:
            for module in level.block:
                h = module(h)
            h = level.upsample(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


# ---------------------------------------------------------------------------
# KL-regularised diagonal Gaussian distribution
# ---------------------------------------------------------------------------

class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.deterministic = deterministic
        if deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self, other=None):
        if self.deterministic:
            return torch.zeros(1)
        if other is None:
            # KL against standard normal
            return 0.5 * torch.sum(
                self.mean.pow(2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3],
            )
        return 0.5 * torch.sum(
            (self.mean - other.mean).pow(2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=[1, 2, 3],
        )

    def nll(self, sample, dims=(1, 2, 3)):
        if self.deterministic:
            return torch.zeros(1)
        log2pi = 1.8378770664093455
        return 0.5 * torch.sum(
            log2pi + self.logvar + (sample - self.mean).pow(2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


# ---------------------------------------------------------------------------
# AutoencoderKL
# ---------------------------------------------------------------------------

class AutoencoderKL(nn.Module):
    """
    KL-regularised Variational Autoencoder.

    Encodes an RGB image into a 4×(H/f)×(W/f) latent, applies KL
    regularisation, and decodes back to pixel space.

    Args:
        embed_dim:   Dimension of the latent-space channels (z_channels).
        scale_factor: Latent scaling factor (divide by this after encoding).
        encoder_config / decoder_config: dicts forwarded to Encoder / Decoder.
    """

    def __init__(
        self,
        embed_dim=4,
        scale_factor=0.18215,
        in_channels=3,
        ch=128,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.encoder = Encoder(
            in_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            z_channels=embed_dim,
            double_z=True,
        )
        self.decoder = Decoder(
            out_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            z_channels=embed_dim,
        )
        self.quant_conv = nn.Conv2d(2 * embed_dim, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, embed_dim, 1)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        rec = self.decode(z)
        return rec, posterior

    def get_last_layer(self):
        return self.decoder.conv_out.weight
