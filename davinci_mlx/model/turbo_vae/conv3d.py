"""3D convolution for MLX via 2D decomposition.

MLX has no native Conv3d. We implement 3D conv by:
1. Applying causal (or symmetric) temporal padding
2. Unfolding the temporal kernel dimension into the channel dimension
3. Running a single 2D conv with kernel (Kh, Kw) over (C_in * Kt) channels

For kernel (Kt, Kh, Kw) = (3, 3, 3):
  - Pad temporally: prepend (Kt-1)=2 frames (causal) or (Kt-1)//2 each side
  - For each output frame t, gather input frames [t, t+1, t+2]
  - Stack along channel dim: (B, C*3, H, W)
  - Apply Conv2d with (3, 3) kernel over C*3 input channels
"""

import mlx.core as mx
import mlx.nn as nn


class Conv3d(nn.Module):
    """3D convolution implemented via temporal unfolding + 2D conv.

    Stores weights in PyTorch 5D format: (out_ch, in_ch, Kt, Kh, Kw).
    At runtime, reshapes to 2D conv weights: (out_ch, Kh, Kw, in_ch * Kt).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size as (Kt, Kh, Kw) tuple or int.
        stride: Stride as (St, Sh, Sw) tuple or int. Only spatial stride supported.
        padding: Spatial padding as int (applied symmetrically to H, W).
        bias: Whether to include a bias term.
        causal: If True, use causal temporal padding (pad start only).
                If False, use symmetric temporal padding.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size=(3, 3, 3), stride=(1, 1, 1),
                 padding: int = 1, bias: bool = True,
                 causal: bool = False):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # (Kt, Kh, Kw)
        self.stride = stride  # (St, Sh, Sw)
        self.padding = padding  # spatial padding for H, W
        self.causal = causal

        kt, kh, kw = kernel_size

        # Weight: PyTorch format (out_ch, in_ch, Kt, Kh, Kw)
        # Will be reshaped to MLX Conv2d format at runtime
        self.weight = mx.zeros((out_channels, in_channels, kt, kh, kw))
        if bias:
            self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (B, C, T, H, W) in channels-first format.

        Returns:
            Output tensor (B, out_C, T', H', W').
        """
        B, C, T, H, W = x.shape
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride

        # --- Temporal padding ---
        if kt > 1:
            if self.causal:
                # Causal: replicate first frame (kt-1) times at the start
                pad_frames = mx.repeat(x[:, :, :1, :, :], kt - 1, axis=2)
                x = mx.concatenate([pad_frames, x], axis=2)
            else:
                # Symmetric: replicate edges
                pad_before = (kt - 1) // 2
                pad_after = kt - 1 - pad_before
                if pad_before > 0:
                    front = mx.repeat(x[:, :, :1, :, :], pad_before, axis=2)
                    x = mx.concatenate([front, x], axis=2)
                if pad_after > 0:
                    back = mx.repeat(x[:, :, -1:, :, :], pad_after, axis=2)
                    x = mx.concatenate([x, back], axis=2)

        _, _, T_padded, _, _ = x.shape

        # --- Temporal unfolding ---
        # Gather windows of kt frames and stack along channel dim
        # Output frames: T_out = (T_padded - kt) // st + 1
        T_out = (T_padded - kt) // st + 1

        # Build list of temporally-unfolded slices
        # Each slice: (B, C, H, W), stacked kt times -> (B, C*kt, H, W)
        frames = []
        for t in range(T_out):
            t_start = t * st
            # Gather kt frames: (B, C, kt, H, W) -> reshape to (B, C*kt, H, W)
            window = x[:, :, t_start:t_start + kt, :, :]  # (B, C, kt, H, W)
            window = window.reshape(B, C * kt, H, W)  # (B, C*kt, H, W)
            frames.append(window)

        # Stack all output frames: (T_out, B, C*kt, H, W)
        # Reshape to (B*T_out, C*kt, H, W) for batched 2D conv
        x_unfolded = mx.stack(frames, axis=2)  # (B, C*kt, T_out, H, W)
        x_unfolded = x_unfolded.transpose(0, 2, 1, 3, 4)  # (B, T_out, C*kt, H, W)
        x_unfolded = x_unfolded.reshape(B * T_out, C * kt, H, W)

        # --- Convert to MLX Conv2d format (channels-last) ---
        x_unfolded = x_unfolded.transpose(0, 2, 3, 1)  # (B*T_out, H, W, C*kt)

        # Reshape weight: (out_ch, in_ch, kt, kh, kw) -> (out_ch, kh, kw, in_ch*kt)
        w = self.weight  # (out_ch, in_ch, kt, kh, kw)
        w = w.transpose(0, 3, 4, 1, 2)  # (out_ch, kh, kw, in_ch, kt)
        w = w.reshape(self.out_channels, kh, kw, C * kt)

        # --- 2D convolution ---
        y = mx.conv2d(x_unfolded, w, stride=(sh, sw), padding=self.padding)
        # y: (B*T_out, H', W', out_ch)

        _, H_out, W_out, _ = y.shape

        # Add bias
        if "bias" in self:
            y = y + self.bias

        # --- Reshape back to 5D ---
        y = y.transpose(0, 3, 1, 2)  # (B*T_out, out_ch, H', W')
        y = y.reshape(B, T_out, self.out_channels, H_out, W_out)
        y = y.transpose(0, 2, 1, 3, 4)  # (B, out_ch, T_out, H', W')

        return y


class SpatialUpsample2x(nn.Module):
    """Spatial 2x upsampling via nearest-neighbor interpolation + 2D conv.

    Used in the upsampler blocks of the Turbo VAE decoder.
    The weight key is `resample.1` (a 2D Conv2d with kernel 3x3).
    """

    def __init__(self, channels: int):
        super().__init__()
        # 2D conv applied after nearest-neighbor upsampling
        # MLX Conv2d weight format: (out_ch, kH, kW, in_ch)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """Upsample spatially by 2x.

        Args:
            x: (B, C, T, H, W)

        Returns:
            (B, C, T, H*2, W*2)
        """
        B, C, T, H, W = x.shape

        # Process each frame: reshape to (B*T, C, H, W)
        x = x.transpose(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)

        # Nearest-neighbor 2x upscale
        # (B*T, C, H, W) -> (B*T, C, H*2, W*2)
        x = x.transpose(0, 2, 3, 1)  # (B*T, H, W, C)
        x = mx.repeat(x, 2, axis=1)  # (B*T, H*2, W, C)
        x = mx.repeat(x, 2, axis=2)  # (B*T, H*2, W*2, C)

        # Apply 2D conv (already in channels-last for MLX)
        x = self.conv(x)  # (B*T, H*2, W*2, C)

        # Reshape back to 5D
        _, H2, W2, _ = x.shape
        x = x.transpose(0, 3, 1, 2)  # (B*T, C, H*2, W*2)
        x = x.reshape(B, T, C, H2, W2)
        x = x.transpose(0, 2, 1, 3, 4)  # (B, C, T, H*2, W*2)

        return x


class TemporalUpsample2x(nn.Module):
    """Temporal 2x upsampling via 3D conv + pixel shuffle along time.

    The temporal upsampler uses a 3D conv with kernel (3, 1, 1) that outputs
    2x the channels, then pixel-shuffles along the temporal axis to double
    the number of frames while halving channels back.

    Weight key: `time_conv.conv` with shape (2*C, C, 3, 1, 1).
    """

    def __init__(self, channels: int):
        super().__init__()
        # 3D conv: (2*C, C, 3, 1, 1) — doubles channels, kernel 3 in time
        # We use our Conv3d but with kernel (3, 1, 1) and no spatial padding
        self.conv = Conv3d(channels, channels * 2,
                           kernel_size=(3, 1, 1), stride=(1, 1, 1),
                           padding=0, bias=True, causal=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Upsample temporally by 2x.

        Args:
            x: (B, C, T, H, W)

        Returns:
            (B, C, T*2, H, W)
        """
        # Apply 3D conv: output is (B, 2*C, T, H, W)
        x = self.conv(x)

        B, C2, T, H, W = x.shape
        C = C2 // 2

        # Pixel shuffle along time: (B, 2*C, T, H, W) -> (B, C, T*2, H, W)
        # Reshape: (B, 2, C, T, H, W) then interleave along T
        x = x.reshape(B, 2, C, T, H, W)
        x = x.transpose(0, 2, 3, 1, 4, 5)  # (B, C, T, 2, H, W)
        x = x.reshape(B, C, T * 2, H, W)

        return x
