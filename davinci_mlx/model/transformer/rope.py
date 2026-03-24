"""Rotary position embeddings (SPLIT) and learnable Fourier embeddings."""
import mlx.core as mx
import mlx.nn as nn


def precompute_freqs(dim: int, max_pos: int, theta: float = 10000.0):
    """Precompute cos/sin frequencies for SPLIT RoPE.

    Returns (cos, sin) each of shape (max_pos, dim//2).
    """
    half_dim = dim // 2
    freqs = 1.0 / (theta ** (mx.arange(0, half_dim).astype(mx.float32) / half_dim))
    positions = mx.arange(max_pos).astype(mx.float32)
    angles = mx.outer(positions, freqs)
    return mx.cos(angles), mx.sin(angles)


def apply_rotary_emb(x: mx.array, cos_freqs: mx.array, sin_freqs: mx.array,
                     positions: mx.array) -> mx.array:
    """Apply SPLIT rotary embeddings.

    SPLIT: first half and second half of head dim are the two components.
    x shape: (B, H, T, D). positions shape: (T,).
    """
    cos = cos_freqs[positions][None, None, :, :]  # (1, 1, T, D//2)
    sin = sin_freqs[positions][None, None, :, :]
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


class ElementWiseFourierEmbed(nn.Module):
    """Learnable Fourier positional embedding.

    Converts coordinate mappings to position encodings using learnable bands.
    """
    def __init__(self, dim: int, num_bands: int = 64):
        super().__init__()
        self.num_bands = num_bands
        self.bands = mx.zeros((num_bands,))

    def __call__(self, coords: mx.array) -> mx.array:
        """coords: (B, T, C) -> (B, T, C * num_bands * 2)."""
        B, T, C = coords.shape
        x = coords[:, :, :, None] * self.bands[None, None, None, :]
        features = mx.concatenate([mx.sin(x), mx.cos(x)], axis=-1)
        return features.reshape(B, T, C * self.num_bands * 2)
