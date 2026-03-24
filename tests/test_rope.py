"""Tests for SPLIT RoPE and Fourier embeddings."""
import mlx.core as mx
from davinci_mlx.model.transformer.rope import precompute_freqs, apply_rotary_emb, ElementWiseFourierEmbed


def test_precompute_freqs_shape():
    cos_f, sin_f = precompute_freqs(dim=128, max_pos=256)
    assert cos_f.shape == (256, 64), f"cos shape: {cos_f.shape}"
    assert sin_f.shape == (256, 64), f"sin shape: {sin_f.shape}"


def test_apply_rotary_emb_shape():
    cos_f, sin_f = precompute_freqs(dim=128, max_pos=256)
    x = mx.random.normal((1, 40, 100, 128))  # (B, H, T, D)
    positions = mx.arange(100)
    result = apply_rotary_emb(x, cos_f, sin_f, positions)
    assert result.shape == x.shape, f"Got {result.shape}"


def test_rotary_emb_different_positions():
    """Different positions should produce different results."""
    cos_f, sin_f = precompute_freqs(dim=128, max_pos=256)
    x = mx.ones((1, 1, 2, 128))
    positions = mx.array([0, 100])
    result = apply_rotary_emb(x, cos_f, sin_f, positions)
    # Position 0 and position 100 should give different outputs
    diff = mx.max(mx.abs(result[0, 0, 0] - result[0, 0, 1])).item()
    assert diff > 0.01, f"Positions should differ, got diff={diff}"


def test_fourier_embed_shape():
    embed = ElementWiseFourierEmbed(dim=128, num_bands=64)
    coords = mx.random.normal((1, 100, 9))
    result = embed(coords)
    # Output: (1, 100, 9 * 64 * 2) = (1, 100, 1152)
    assert result.shape == (1, 100, 9 * 64 * 2), f"Got {result.shape}"
