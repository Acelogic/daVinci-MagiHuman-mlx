"""Tests for SwiGLU7 and GELU7 feed-forward networks."""
import mlx.core as mx
from davinci_mlx.model.transformer.feed_forward import SwiGLU7FFN, GELU7FFN


def test_swiglu7_shape():
    ffn = SwiGLU7FFN(hidden_size=5120, intermediate_size=13652)
    x = mx.random.normal((1, 64, 5120))
    assert ffn(x).shape == (1, 64, 5120)


def test_gelu7_shape():
    ffn = GELU7FFN(hidden_size=5120, intermediate_size=20480)
    x = mx.random.normal((1, 64, 5120))
    assert ffn(x).shape == (1, 64, 5120)


def test_swiglu7_clamping():
    """Output must be within [-7, 7] regardless of input magnitude."""
    ffn = SwiGLU7FFN(hidden_size=32, intermediate_size=64)
    # Set weights to large values to force output beyond clamp range
    import mlx.nn as nn
    large_w = mx.ones_like(ffn.up_gate_proj.weight) * 10.0
    ffn.up_gate_proj.weight = large_w
    ffn.down_proj.weight = mx.ones_like(ffn.down_proj.weight) * 10.0
    x = mx.ones((1, 4, 32)) * 10.0
    result = ffn(x)
    mx.eval(result)
    assert mx.all(result >= -7.0).item(), f"Min: {mx.min(result).item()}"
    assert mx.all(result <= 7.0).item(), f"Max: {mx.max(result).item()}"


def test_gelu7_clamping():
    """Output must be within [-7, 7] regardless of input magnitude."""
    ffn = GELU7FFN(hidden_size=32, intermediate_size=128)
    ffn.up_proj.weight = mx.ones_like(ffn.up_proj.weight) * 10.0
    ffn.down_proj.weight = mx.ones_like(ffn.down_proj.weight) * 10.0
    x = mx.ones((1, 4, 32)) * 10.0
    result = ffn(x)
    mx.eval(result)
    assert mx.all(result >= -7.0).item(), f"Min: {mx.min(result).item()}"
    assert mx.all(result <= 7.0).item(), f"Max: {mx.max(result).item()}"
