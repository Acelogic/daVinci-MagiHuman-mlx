"""Tests for fused Metal kernels."""
import mlx.core as mx
from davinci_mlx.kernels.fused_ops import silu_mul


def test_silu_mul():
    a = mx.random.normal((4, 128))
    b = mx.random.normal((4, 128))
    result = silu_mul(a, b)
    expected = mx.sigmoid(a) * a * b
    diff = mx.max(mx.abs(result - expected)).item()
    assert diff < 1e-4, f"Max diff: {diff}"


def test_silu_mul_large():
    a = mx.random.normal((1024, 13652))
    b = mx.random.normal((1024, 13652))
    result = silu_mul(a, b)
    expected = mx.sigmoid(a) * a * b
    diff = mx.max(mx.abs(result - expected)).item()
    assert diff < 1e-3, f"Max diff: {diff}"
