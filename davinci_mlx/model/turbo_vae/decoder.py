"""Turbo VAE Decoder for daVinci-MagiHuman.

Architecture (channels: 512 -> 256 -> 128 -> 64):
  conv_in(48 -> 512, 3x3x3)
  mid_block: 3 ResNet blocks at 512
  up_blocks.0: 3 resnets(512) + upsampler(spatial+temporal)
  up_blocks.1: conv_in resnet(512->256) + 2 resnets(256) + upsampler(spatial+temporal)
  up_blocks.2: conv_in resnet(256->128) + 2 resnets(128) + upsampler(spatial only)
  up_blocks.3: conv_in resnet(128->64) + 2 resnets(64), no upsampler
  conv_out(64 -> 12, 3x3x3)
  unpatchify: 12 -> 3 RGB with patch_size=2

Weight key convention: each 3D conv is wrapped as `name.conv.weight/bias`.
"""

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .conv3d import Conv3d, SpatialUpsample2x, TemporalUpsample2x


class ResNetBlock3d(nn.Module):
    """Residual block with two 3D convolutions and SiLU activation.

    If in_channels != out_channels, a 1x1x1 shortcut conv is used.
    No normalization layers — the Turbo VAE uses pure SiLU + residual.

    Weight keys (relative to block):
        conv1.conv.weight/bias  (out_ch, in_ch, 3, 3, 3)
        conv2.conv.weight/bias  (out_ch, out_ch, 3, 3, 3)
        conv_shortcut.conv.weight/bias  (out_ch, in_ch, 1, 1, 1)  [if in!=out]
    """

    def __init__(self, in_channels: int, out_channels: int, causal: bool = False):
        super().__init__()
        self.conv1 = Conv3d(in_channels, out_channels,
                            kernel_size=(3, 3, 3), padding=1, causal=causal)
        self.conv2 = Conv3d(out_channels, out_channels,
                            kernel_size=(3, 3, 3), padding=1, causal=causal)

        if in_channels != out_channels:
            self.conv_shortcut = Conv3d(in_channels, out_channels,
                                        kernel_size=(1, 1, 1), padding=0,
                                        causal=causal)
        else:
            self.conv_shortcut = None

    def __call__(self, x: mx.array) -> mx.array:
        residual = x

        h = nn.silu(x)
        h = self.conv1(h)
        h = nn.silu(h)
        h = self.conv2(h)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return residual + h


class Upsampler3d(nn.Module):
    """Upsampling block: spatial 2x + optional temporal 2x.

    Weight keys (relative to upsampler):
        resample.1.weight/bias — Conv2d (channels, channels, 3, 3)
        time_conv.conv.weight/bias — Conv3d (2*channels, channels, 3, 1, 1) [if temporal]
    """

    def __init__(self, channels: int, temporal: bool = True):
        super().__init__()
        self.resample = SpatialUpsample2x(channels)
        self.time_conv = TemporalUpsample2x(channels) if temporal else None

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resample(x)
        if self.time_conv is not None:
            x = self.time_conv(x)
        return x


class UpBlock3d(nn.Module):
    """Decoder upsampling block.

    Contains:
    - Optional conv_in ResNetBlock (for channel change, e.g. 512->256)
    - N ResNet blocks at output channels
    - Optional Upsampler (spatial + optional temporal)

    Weight keys (relative to up_block):
        conv_in.conv1.conv.weight/bias
        conv_in.conv2.conv.weight/bias
        conv_in.conv_shortcut.conv.weight/bias
        resnets.{i}.conv1.conv.weight/bias
        resnets.{i}.conv2.conv.weight/bias
        upsamplers.0.resample.1.weight/bias
        upsamplers.0.time_conv.conv.weight/bias
    """

    def __init__(self, in_channels: int, out_channels: int,
                 num_resnets: int, has_upsampler: bool = True,
                 upsampler_temporal: bool = True, causal: bool = False):
        super().__init__()

        # Channel-changing resnet at input (if needed)
        if in_channels != out_channels:
            self.conv_in = ResNetBlock3d(in_channels, out_channels, causal=causal)
        else:
            self.conv_in = None

        # Additional ResNet blocks at output channel size
        self.resnets = [
            ResNetBlock3d(out_channels, out_channels, causal=causal)
            for _ in range(num_resnets)
        ]

        # Upsampler
        if has_upsampler:
            self.upsamplers = [Upsampler3d(out_channels, temporal=upsampler_temporal)]
        else:
            self.upsamplers = []

    def __call__(self, x: mx.array) -> mx.array:
        if self.conv_in is not None:
            x = self.conv_in(x)

        for resnet in self.resnets:
            x = resnet(x)

        for upsampler in self.upsamplers:
            x = upsampler(x)

        return x


class TurboVAEDecoder(nn.Module):
    """Turbo VAE 3D decoder for daVinci-MagiHuman video generation.

    Decodes latent tensor (B, 48, T, H, W) to pixel video (B, 3, T', H', W')
    where T' = T*4, H' = H*16, W' = W*16.

    Architecture from TurboV3-Wan22-TinyShallow_7_7.json:
        - conv_in: 48 -> 512
        - mid_block: 3 ResNet blocks at 512
        - up_blocks: 4 stages with progressive channel reduction and upsampling
        - conv_out: 64 -> 12 (= 3 * 2^2 for unpatchify)
        - unpatchify: rearrange 12ch -> 3 RGB with 2x2 spatial expansion
    """

    def __init__(self, latent_channels: int = 48, causal: bool = False,
                 patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.causal = causal

        # Latent denormalization statistics (48 channels)
        # These are loaded from the config or set during weight loading
        self.latent_mean = mx.zeros((1, latent_channels, 1, 1, 1))
        self.latent_std = mx.ones((1, latent_channels, 1, 1, 1))

        # conv_in: 48 -> 512
        self.conv_in = Conv3d(latent_channels, 512, kernel_size=(3, 3, 3),
                              padding=1, causal=causal)

        # mid_block: 3 ResNet blocks at 512
        self.mid_block = MidBlock3d(512, num_layers=3, causal=causal)

        # up_blocks (4 stages, decoder_layers_per_block = [2,2,2,3,3]):
        # Block layout from weights:
        #   up_blocks.0: 3 resnets(512,512) + upsampler(spatial+temporal)
        #   up_blocks.1: conv_in(512->256) + 2 resnets(256) + upsampler(spatial+temporal)
        #   up_blocks.2: conv_in(256->128) + 2 resnets(128) + upsampler(spatial only)
        #   up_blocks.3: conv_in(128->64) + 2 resnets(64), no upsampler
        self.up_blocks = [
            UpBlock3d(512, 512, num_resnets=3,
                      has_upsampler=True, upsampler_temporal=True, causal=causal),
            UpBlock3d(512, 256, num_resnets=2,
                      has_upsampler=True, upsampler_temporal=True, causal=causal),
            UpBlock3d(256, 128, num_resnets=2,
                      has_upsampler=True, upsampler_temporal=False, causal=causal),
            UpBlock3d(128, 64, num_resnets=2,
                      has_upsampler=False, causal=causal),
        ]

        # conv_out: 64 -> 12 (= 3 RGB * 2*2 patch)
        out_channels = 3 * patch_size * patch_size  # 12
        self.conv_out = Conv3d(64, out_channels, kernel_size=(3, 3, 3),
                               padding=1, causal=causal)

    def denormalize_latent(self, z: mx.array) -> mx.array:
        """Denormalize latent from standard distribution to model space.

        z_denorm = z * std + mean
        """
        return z * self.latent_std + self.latent_mean

    def unpatchify(self, x: mx.array) -> mx.array:
        """Convert patched output to pixel space.

        Input: (B, C_patch, T, H, W) where C_patch = 3 * p * p = 12
        Output: (B, 3, T, H*p, W*p) where p = patch_size = 2

        Rearranges: (B, 3*p*p, T, H, W) -> (B, 3, T, H*p, W*p)
        """
        B, C_patch, T, H, W = x.shape
        p = self.patch_size
        C = C_patch // (p * p)  # 3

        # (B, C*p*p, T, H, W) -> (B, C, p, p, T, H, W) -> (B, C, T, H, p, W, p)
        x = x.reshape(B, C, p, p, T, H, W)
        x = x.transpose(0, 1, 4, 5, 2, 6, 3)  # (B, C, T, H, p, W, p)
        x = x.reshape(B, C, T, H * p, W * p)

        return x

    def __call__(self, z: mx.array, denormalize: bool = True) -> mx.array:
        """Decode latent to video.

        Args:
            z: Latent tensor (B, 48, T, H, W) in latent space.
            denormalize: If True, apply latent denormalization before decoding.

        Returns:
            Decoded video (B, 3, T', H', W') in [-1, 1] range.
            T' = T * temporal_compression (4)
            H' = H * spatial_compression (16)
            W' = W * spatial_compression (16)
        """
        if denormalize:
            z = self.denormalize_latent(z)

        x = self.conv_in(z)

        x = self.mid_block(x)

        for up_block in self.up_blocks:
            x = up_block(x)

        x = nn.silu(x)
        x = self.conv_out(x)

        # Unpatchify: (B, 12, T, H, W) -> (B, 3, T, H*2, W*2)
        x = self.unpatchify(x)

        return x


class MidBlock3d(nn.Module):
    """Mid-block with N ResNet blocks, all at the same channel size."""

    def __init__(self, channels: int, num_layers: int = 3, causal: bool = False):
        super().__init__()
        self.resnets = [
            ResNetBlock3d(channels, channels, causal=causal)
            for _ in range(num_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        return x


def load_turbo_vae_weights(decoder: TurboVAEDecoder, ckpt_path: str,
                           latent_mean: Optional[List[float]] = None,
                           latent_std: Optional[List[float]] = None) -> TurboVAEDecoder:
    """Load weights from a PyTorch .ckpt file into the TurboVAEDecoder.

    The checkpoint stores EMA weights under ckpt['ema_state_dict'].
    Keys have prefix 'module.decoder.' which we strip.
    We skip 'aligned_feature_projection_heads' (distillation-only).

    5D conv weights (out, in, T, H, W) are stored as-is since our Conv3d
    stores weights in PyTorch format and reshapes at runtime.

    Args:
        decoder: TurboVAEDecoder instance to load weights into.
        ckpt_path: Path to the .ckpt file.
        latent_mean: Per-channel mean for latent denormalization (48 values).
        latent_std: Per-channel std for latent denormalization (48 values).

    Returns:
        The decoder with loaded weights.
    """
    import torch
    import numpy as np

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ema_sd = ckpt["ema_state_dict"]

    # Filter to decoder keys only, strip prefix
    decoder_weights = {}
    for key, value in ema_sd.items():
        if key.startswith("module.decoder."):
            clean_key = key[len("module.decoder."):]
            decoder_weights[clean_key] = value.numpy()
        # Skip aligned_feature_projection_heads

    # Helper to set a weight on a nested module path
    def _set_weight(module, key_path: str, np_array):
        """Set weight on module following dot-separated key path."""
        parts = key_path.split(".")
        obj = module
        for part in parts[:-1]:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)

        leaf_name = parts[-1]
        value = mx.array(np_array)

        # For Conv2d weights in MLX format: (out, in, H, W) -> (out, H, W, in)
        # Check if this is a 2D conv weight (the spatial upsampler `resample.1`)
        if leaf_name == "weight" and value.ndim == 4:
            # PyTorch Conv2d: (out, in, H, W) -> MLX: (out, H, W, in)
            value = mx.array(np_array.transpose(0, 2, 3, 1))

        setattr(obj, leaf_name, value)

    # Map checkpoint keys to model structure
    for ckpt_key, np_val in decoder_weights.items():
        # conv_in.conv.weight -> conv_in.weight (strip the inner .conv.)
        # mid_block.resnets.0.conv1.conv.weight -> mid_block.resnets.0.conv1.weight
        # up_blocks.0.upsamplers.0.resample.1.weight -> up_blocks.0.upsamplers.0.resample.conv.weight
        # up_blocks.0.upsamplers.0.time_conv.conv.weight -> up_blocks.0.upsamplers.0.time_conv.conv.weight

        model_key = _map_ckpt_key(ckpt_key)
        if model_key is None:
            continue

        try:
            _set_weight(decoder, model_key, np_val)
        except (AttributeError, IndexError, TypeError) as e:
            print(f"Warning: could not set key '{ckpt_key}' -> '{model_key}': {e}")

    # Set latent statistics if provided
    if latent_mean is not None:
        mean = mx.array(latent_mean, dtype=mx.float32).reshape(1, -1, 1, 1, 1)
        decoder.latent_mean = mean
    if latent_std is not None:
        std = mx.array(latent_std, dtype=mx.float32).reshape(1, -1, 1, 1, 1)
        decoder.latent_std = std

    return decoder


def _map_ckpt_key(ckpt_key: str) -> Optional[str]:
    """Map a checkpoint key to the corresponding model attribute path.

    Checkpoint key patterns:
        conv_in.conv.{weight,bias}
        conv_out.conv.{weight,bias}
        mid_block.resnets.{i}.conv{1,2}.conv.{weight,bias}
        up_blocks.{i}.resnets.{j}.conv{1,2}.conv.{weight,bias}
        up_blocks.{i}.conv_in.conv{1,2}.conv.{weight,bias}
        up_blocks.{i}.conv_in.conv_shortcut.conv.{weight,bias}
        up_blocks.{i}.upsamplers.0.resample.1.{weight,bias}
        up_blocks.{i}.upsamplers.0.time_conv.conv.{weight,bias}

    Model paths follow the same structure but:
        - Strip one level of '.conv.' nesting for 3D conv wrappers
        - Map 'resample.1' to 'resample.conv' (SpatialUpsample2x.conv)
    """
    # Top-level conv_in / conv_out: strip '.conv.' wrapper
    if ckpt_key.startswith("conv_in.conv.") or ckpt_key.startswith("conv_out.conv."):
        # conv_in.conv.weight -> conv_in.weight
        return ckpt_key.replace(".conv.", ".", 1)

    # mid_block resnets
    if ckpt_key.startswith("mid_block.resnets."):
        # mid_block.resnets.0.conv1.conv.weight -> mid_block.resnets.0.conv1.weight
        return ckpt_key.replace(".conv.", ".", 1)

    # up_blocks
    if ckpt_key.startswith("up_blocks."):
        # Upsampler spatial conv: resample.1.weight -> resample.conv.weight
        if ".upsamplers.0.resample.1." in ckpt_key:
            return ckpt_key.replace(".resample.1.", ".resample.conv.")

        # Upsampler temporal conv: time_conv.conv.weight -> time_conv.conv.weight
        if ".upsamplers.0.time_conv.conv." in ckpt_key:
            return ckpt_key

        # ResNet convs: strip '.conv.' nesting
        # up_blocks.0.resnets.0.conv1.conv.weight -> up_blocks.0.resnets.0.conv1.weight
        # up_blocks.1.conv_in.conv1.conv.weight -> up_blocks.1.conv_in.conv1.weight
        # up_blocks.1.conv_in.conv_shortcut.conv.weight -> up_blocks.1.conv_in.conv_shortcut.weight
        parts = ckpt_key.split(".")
        # Find pattern: ...convX.conv.weight -> ...convX.weight (remove one .conv.)
        # The pattern is: the second-to-last segment before weight/bias is 'conv'
        if len(parts) >= 3 and parts[-2] == "conv" and parts[-1] in ("weight", "bias"):
            # Check if this is a conv wrapper (convN.conv.weight pattern)
            new_parts = parts[:-2] + [parts[-1]]  # remove the '.conv.' before weight/bias
            return ".".join(new_parts)

        return ckpt_key

    return None
