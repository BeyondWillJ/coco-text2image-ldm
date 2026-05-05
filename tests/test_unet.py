"""
Tests for the UNet model.
"""

import pytest
import torch


@pytest.fixture
def tiny_unet():
    """A very small UNet for fast CPU tests."""
    from ldm.modules.diffusionmodules.model import UNetModel

    return UNetModel(
        image_size=8,
        in_channels=4,
        model_channels=16,
        out_channels=4,
        num_res_blocks=1,
        attention_resolutions=[2],
        channel_mult=(1, 2),
        num_heads=2,
        use_spatial_transformer=False,
    )


def test_unet_output_shape(tiny_unet):
    B, C, H, W = 2, 4, 8, 8
    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 100, (B,))
    out = tiny_unet(x, t)
    assert out.shape == (B, C, H, W), f"Unexpected shape: {out.shape}"


def test_unet_output_dtype(tiny_unet):
    x = torch.randn(1, 4, 8, 8)
    t = torch.randint(0, 100, (1,))
    out = tiny_unet(x, t)
    assert out.dtype == torch.float32


def test_unet_with_cross_attention():
    from ldm.modules.diffusionmodules.model import UNetModel

    context_dim = 32
    model = UNetModel(
        image_size=8,
        in_channels=4,
        model_channels=16,
        out_channels=4,
        num_res_blocks=1,
        attention_resolutions=[2],
        channel_mult=(1, 2),
        num_heads=2,
        use_spatial_transformer=True,
        context_dim=context_dim,
    )
    B, C, H, W = 1, 4, 8, 8
    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 100, (B,))
    context = torch.randn(B, 10, context_dim)  # 10 tokens
    out = model(x, t, context=context)
    assert out.shape == (B, C, H, W)


def test_unet_zero_timestep(tiny_unet):
    """UNet should not explode for t=0."""
    x = torch.randn(1, 4, 8, 8)
    t = torch.zeros(1, dtype=torch.long)
    out = tiny_unet(x, t)
    assert not torch.isnan(out).any(), "NaNs in UNet output at t=0"
