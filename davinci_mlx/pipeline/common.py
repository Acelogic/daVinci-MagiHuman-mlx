"""Pipeline utilities: validation, latent shape computation, video output."""
import math
import numpy as np
import mlx.core as mx

VAE_TEMPORAL_STRIDE = 4
VAE_SPATIAL_STRIDE = 16
LATENT_CHANNELS = 48


def validate_dimensions(height: int, width: int, num_frames: int):
    spatial_divisor = VAE_SPATIAL_STRIDE * 2  # stride * patch_size = 32
    if height % spatial_divisor != 0 or width % spatial_divisor != 0:
        raise ValueError(
            f"Height ({height}) and width ({width}) must be divisible by {spatial_divisor}"
        )
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")


def compute_latent_shape(height: int, width: int, num_frames: int):
    T = math.ceil(num_frames / VAE_TEMPORAL_STRIDE)
    H = height // VAE_SPATIAL_STRIDE
    W = width // VAE_SPATIAL_STRIDE
    return (LATENT_CHANNELS, T, H, W)


def video_to_numpy(video: mx.array) -> np.ndarray:
    """Convert (B, C, T, H, W) float video to (T, H, W, C) uint8 numpy.

    The VAE outputs values in [-1, 1] range. We rescale to [0, 255].
    """
    v = video[0].transpose(1, 2, 3, 0)  # (C, T, H, W) -> (T, H, W, C)
    v = (v + 1.0) * 0.5  # [-1, 1] -> [0, 1]
    v = mx.clip(v * 255.0, 0, 255).astype(mx.uint8)
    return np.array(v)


def save_video(frames: np.ndarray, path: str, fps: int = 24):
    import imageio
    writer = imageio.get_writer(path, fps=fps, codec="libx264")
    for i in range(frames.shape[0]):
        writer.append_data(frames[i])
    writer.close()
