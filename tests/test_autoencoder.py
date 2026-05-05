"""
Tests for the VAE autoencoder.
"""

import pytest
import torch


@pytest.fixture
def tiny_vae():
    """A very small AutoencoderKL for fast CPU tests."""
    from ldm.models.autoencoder import AutoencoderKL

    return AutoencoderKL(
        embed_dim=4,
        scale_factor=0.18215,
        in_channels=3,
        ch=16,
        ch_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=[],
        dropout=0.0,
    )


def test_encode_shape(tiny_vae):
    x = torch.randn(2, 3, 32, 32)
    posterior = tiny_vae.encode(x)
    z = posterior.sample()
    # With ch_mult=(1,2) the spatial downsampling is 2×, so 32→16
    assert z.shape == (2, 4, 16, 16), f"Unexpected latent shape: {z.shape}"


def test_decode_shape(tiny_vae):
    z = torch.randn(2, 4, 16, 16)
    rec = tiny_vae.decode(z)
    assert rec.shape == (2, 3, 32, 32), f"Unexpected reconstruction shape: {rec.shape}"


def test_forward_shapes(tiny_vae):
    x = torch.randn(2, 3, 32, 32)
    rec, posterior = tiny_vae(x)
    assert rec.shape == x.shape
    assert posterior.mean.shape == (2, 4, 16, 16)


def test_kl_nonnegative(tiny_vae):
    x = torch.randn(2, 3, 32, 32)
    _, posterior = tiny_vae(x)
    kl = posterior.kl()
    assert (kl >= 0).all(), "KL divergence should be non-negative"


def test_reconstruction_range(tiny_vae):
    """After decoding the reconstruction values should be finite."""
    x = torch.randn(1, 3, 32, 32)
    rec, _ = tiny_vae(x)
    assert torch.isfinite(rec).all(), "Reconstruction contains non-finite values"


def test_diagonal_gaussian_deterministic():
    from ldm.models.autoencoder import DiagonalGaussianDistribution

    params = torch.cat([torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 8, 8)], dim=1)
    dist = DiagonalGaussianDistribution(params, deterministic=True)
    kl = dist.kl()
    assert kl.item() == 0.0
