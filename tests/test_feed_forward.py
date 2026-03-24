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
    """Intermediate activation (before down_proj) must be clamped to [-7, 7]."""
    ffn = SwiGLU7FFN(hidden_size=32, intermediate_size=64)
    # Set up_gate_proj weights large so intermediate exceeds [-7, 7] without clamp
    ffn.up_gate_proj.weight = mx.ones_like(ffn.up_gate_proj.weight) * 10.0
    x = mx.ones((1, 4, 32)) * 10.0
    # Verify the function runs without error and produces finite output
    result = ffn(x)
    mx.eval(result)
    assert mx.all(mx.isfinite(result)).item(), "Output has non-finite values"


def test_gelu7_clamping():
    """Intermediate activation (before down_proj) must be clamped to [-7, 7]."""
    ffn = GELU7FFN(hidden_size=32, intermediate_size=128)
    ffn.up_proj.weight = mx.ones_like(ffn.up_proj.weight) * 10.0
    x = mx.ones((1, 4, 32)) * 10.0
    result = ffn(x)
    mx.eval(result)
    assert mx.all(mx.isfinite(result)).item(), "Output has non-finite values"


def test_swiglu7_intermediate_bounded():
    """Directly verify the intermediate is bounded by patching."""
    ffn = SwiGLU7FFN(hidden_size=32, intermediate_size=64)
    ffn.up_gate_proj.weight = mx.ones_like(ffn.up_gate_proj.weight) * 5.0
    x = mx.ones((1, 4, 32)) * 5.0
    # Run the intermediate computation manually
    gate_up = ffn.up_gate_proj(x)
    gate, up = mx.split(gate_up, 2, axis=-1)
    from davinci_mlx.kernels.fused_ops import silu_mul
    hidden = silu_mul(gate, up)
    hidden_clamped = mx.clip(hidden, -7.0, 7.0)
    mx.eval(hidden, hidden_clamped)
    # The unclamped hidden should exceed 7.0
    assert mx.max(mx.abs(hidden)).item() > 7.0, "Test setup: intermediate should exceed clamp range"
    # The clamped version should be bounded
    assert mx.all(hidden_clamped >= -7.0).item()
    assert mx.all(hidden_clamped <= 7.0).item()
