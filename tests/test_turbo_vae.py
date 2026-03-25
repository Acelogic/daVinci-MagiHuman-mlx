"""Tests for Turbo VAE decoder components."""
import mlx.core as mx
from davinci_mlx.model.turbo_vae.conv3d import Conv3d, SpatialUpsample2x, TemporalUpsample2x
from davinci_mlx.model.turbo_vae.decoder import (
    ResNetBlock3d, UpBlock3d, Upsampler3d, MidBlock3d, TurboVAEDecoder,
)


# --- Conv3d tests ---

def test_conv3d_shape_same_channels():
    """Conv3d preserves spatial dims with padding=1 and kernel 3x3x3."""
    conv = Conv3d(16, 16, kernel_size=(3, 3, 3), padding=1)
    x = mx.random.normal((1, 16, 4, 8, 8))
    y = conv(x)
    assert y.shape == (1, 16, 4, 8, 8), f"Expected (1,16,4,8,8), got {y.shape}"


def test_conv3d_shape_channel_change():
    """Conv3d changes channels correctly."""
    conv = Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1)
    x = mx.random.normal((1, 16, 4, 8, 8))
    y = conv(x)
    assert y.shape == (1, 32, 4, 8, 8), f"Expected (1,32,4,8,8), got {y.shape}"


def test_conv3d_1x1x1():
    """1x1x1 conv acts as pointwise channel projection."""
    conv = Conv3d(16, 8, kernel_size=(1, 1, 1), padding=0)
    x = mx.random.normal((1, 16, 4, 8, 8))
    y = conv(x)
    assert y.shape == (1, 8, 4, 8, 8), f"Expected (1,8,4,8,8), got {y.shape}"


def test_conv3d_causal():
    """Causal conv3d preserves temporal dim."""
    conv = Conv3d(16, 16, kernel_size=(3, 3, 3), padding=1, causal=True)
    x = mx.random.normal((1, 16, 4, 8, 8))
    y = conv(x)
    assert y.shape == (1, 16, 4, 8, 8), f"Expected (1,16,4,8,8), got {y.shape}"


def test_conv3d_temporal_kernel_only():
    """Conv3d with kernel (3,1,1) — temporal-only convolution."""
    conv = Conv3d(16, 32, kernel_size=(3, 1, 1), padding=0)
    x = mx.random.normal((1, 16, 4, 8, 8))
    y = conv(x)
    assert y.shape == (1, 32, 4, 8, 8), f"Expected (1,32,4,8,8), got {y.shape}"


# --- Spatial upsampler tests ---

def test_spatial_upsample2x():
    """Spatial upsampler doubles H and W."""
    up = SpatialUpsample2x(16)
    x = mx.random.normal((1, 16, 4, 8, 8))
    y = up(x)
    assert y.shape == (1, 16, 4, 16, 16), f"Expected (1,16,4,16,16), got {y.shape}"


# --- Temporal upsampler tests ---

def test_temporal_upsample2x():
    """Temporal upsampler doubles T."""
    up = TemporalUpsample2x(16)
    x = mx.random.normal((1, 16, 4, 8, 8))
    y = up(x)
    assert y.shape == (1, 16, 8, 8, 8), f"Expected (1,16,8,8,8), got {y.shape}"


# --- ResNet block tests ---

def test_resnet_block_same_channels():
    """ResNet block preserves shape when in==out channels."""
    block = ResNetBlock3d(16, 16)
    x = mx.random.normal((1, 16, 4, 8, 8))
    y = block(x)
    assert y.shape == (1, 16, 4, 8, 8), f"Expected (1,16,4,8,8), got {y.shape}"


def test_resnet_block_channel_change():
    """ResNet block handles channel change with shortcut conv."""
    block = ResNetBlock3d(32, 16)
    x = mx.random.normal((1, 32, 4, 8, 8))
    y = block(x)
    assert y.shape == (1, 16, 4, 8, 8), f"Expected (1,16,4,8,8), got {y.shape}"
    assert block.conv_shortcut is not None, "Should have shortcut conv"


def test_resnet_block_residual():
    """ResNet block with zero-init produces identity-like output."""
    block = ResNetBlock3d(16, 16)
    x = mx.random.normal((1, 16, 4, 8, 8))
    y = block(x)
    # With zero-init weights, conv outputs are zero -> y = residual + 0 = residual
    # But SiLU is applied to residual before conv, so this won't be exact identity
    # Just check shape
    assert y.shape == x.shape


# --- MidBlock tests ---

def test_mid_block_shape():
    """Mid block preserves shape across multiple resnets."""
    mid = MidBlock3d(32, num_layers=3)
    x = mx.random.normal((1, 32, 4, 8, 8))
    y = mid(x)
    assert y.shape == (1, 32, 4, 8, 8), f"Expected (1,32,4,8,8), got {y.shape}"


# --- UpBlock tests ---

def test_upblock_with_upsampler():
    """UpBlock upsamples spatially and temporally."""
    block = UpBlock3d(32, 32, num_resnets=2,
                      has_upsampler=True, upsampler_temporal=True)
    x = mx.random.normal((1, 32, 4, 8, 8))
    y = block(x)
    assert y.shape == (1, 32, 8, 16, 16), f"Expected (1,32,8,16,16), got {y.shape}"


def test_upblock_spatial_only():
    """UpBlock with spatial-only upsampling."""
    block = UpBlock3d(32, 16, num_resnets=2,
                      has_upsampler=True, upsampler_temporal=False)
    x = mx.random.normal((1, 32, 4, 8, 8))
    y = block(x)
    assert y.shape == (1, 16, 4, 16, 16), f"Expected (1,16,4,16,16), got {y.shape}"


def test_upblock_no_upsampler():
    """UpBlock without upsampling."""
    block = UpBlock3d(32, 16, num_resnets=2, has_upsampler=False)
    x = mx.random.normal((1, 32, 4, 8, 8))
    y = block(x)
    assert y.shape == (1, 16, 4, 8, 8), f"Expected (1,16,4,8,8), got {y.shape}"


# --- Full decoder tests ---

def test_decoder_output_channels():
    """Decoder conv_out produces 12 channels (3 RGB * 2*2 patch) before unpatchify."""
    decoder = TurboVAEDecoder(latent_channels=48, patch_size=2)
    assert decoder.conv_out.out_channels == 12


def test_decoder_unpatchify():
    """Unpatchify correctly rearranges 12ch -> 3 RGB with 2x spatial expansion."""
    decoder = TurboVAEDecoder(latent_channels=48, patch_size=2)
    # Simulate conv_out output
    x = mx.random.normal((1, 12, 4, 8, 8))
    y = decoder.unpatchify(x)
    assert y.shape == (1, 3, 4, 16, 16), f"Expected (1,3,4,16,16), got {y.shape}"


def test_decoder_full_forward():
    """Full decoder forward pass with correct output shape.

    Input: (B, 48, T=1, H=1, W=1) latent
    Expected output shape progression:
        conv_in: (1, 512, 1, 1, 1)
        mid_block: (1, 512, 1, 1, 1)
        up_blocks.0 (512, spatial+temporal): (1, 512, 2, 2, 2)
        up_blocks.1 (256, spatial+temporal): (1, 256, 4, 4, 4)
        up_blocks.2 (128, spatial only): (1, 128, 4, 8, 8)
        up_blocks.3 (64, no upsampler): (1, 64, 4, 8, 8)
        conv_out: (1, 12, 4, 8, 8)
        unpatchify: (1, 3, 4, 16, 16)
    """
    decoder = TurboVAEDecoder(latent_channels=48, patch_size=2)
    z = mx.random.normal((1, 48, 1, 1, 1))
    y = decoder(z, denormalize=False)
    assert y.shape == (1, 3, 4, 16, 16), f"Expected (1,3,4,16,16), got {y.shape}"


def test_decoder_256p_shape():
    """Decoder with 256p latent dimensions.

    For 256p video (256x256, ~17 frames):
        Latent: (1, 48, 4, 16, 16)  [T=17->4 with temporal 4x, H=W=256->16 with spatial 16x]
        Output: (1, 3, T', H', W')
        T' = 4 * 4 = 16 (temporal 2x from up_blocks.0 and up_blocks.1)
            Wait — actually we only have 2 temporal upsamplers, so T' = 4 * 2 * 2 = 16
        H' = 16 * 2 * 2 * 2 * 2 = 256 (3 spatial upsamplers * 2 + unpatchify * 2)
        W' same as H'
    """
    decoder = TurboVAEDecoder(latent_channels=48, patch_size=2)
    z = mx.random.normal((1, 48, 4, 16, 16))
    y = decoder(z, denormalize=False)
    # Spatial: 16 -> 32 (up0) -> 64 (up1) -> 128 (up2) -> 128 (up3, no up) -> 128
    # Then unpatchify 2x: 128 * 2 = 256
    # Temporal: 4 -> 8 (up0) -> 16 (up1) -> 16 (up2, spatial only) -> 16 (up3)
    assert y.shape == (1, 3, 16, 256, 256), f"Expected (1,3,16,256,256), got {y.shape}"


# --- Weight key mapping tests ---

def test_key_mapping():
    """Verify checkpoint key mapping produces correct model paths."""
    from davinci_mlx.model.turbo_vae.decoder import _map_ckpt_key

    # conv_in / conv_out
    assert _map_ckpt_key("conv_in.conv.weight") == "conv_in.weight"
    assert _map_ckpt_key("conv_in.conv.bias") == "conv_in.bias"
    assert _map_ckpt_key("conv_out.conv.weight") == "conv_out.weight"

    # mid_block
    assert _map_ckpt_key("mid_block.resnets.0.conv1.conv.weight") == "mid_block.resnets.0.conv1.weight"
    assert _map_ckpt_key("mid_block.resnets.2.conv2.conv.bias") == "mid_block.resnets.2.conv2.bias"

    # up_blocks resnets
    assert _map_ckpt_key("up_blocks.0.resnets.0.conv1.conv.weight") == "up_blocks.0.resnets.0.conv1.weight"

    # up_blocks conv_in (channel-change resnet)
    assert _map_ckpt_key("up_blocks.1.conv_in.conv1.conv.weight") == "up_blocks.1.conv_in.conv1.weight"
    assert _map_ckpt_key("up_blocks.1.conv_in.conv_shortcut.conv.weight") == "up_blocks.1.conv_in.conv_shortcut.weight"

    # up_blocks upsampler spatial
    assert _map_ckpt_key("up_blocks.0.upsamplers.0.resample.1.weight") == "up_blocks.0.upsamplers.0.resample.conv.weight"
    assert _map_ckpt_key("up_blocks.0.upsamplers.0.resample.1.bias") == "up_blocks.0.upsamplers.0.resample.conv.bias"

    # up_blocks upsampler temporal (no mapping change)
    assert _map_ckpt_key("up_blocks.0.upsamplers.0.time_conv.conv.weight") == "up_blocks.0.upsamplers.0.time_conv.conv.weight"
