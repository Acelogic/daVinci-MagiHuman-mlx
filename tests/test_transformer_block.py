"""Tests for transformer block."""
import mlx.core as mx
from davinci_mlx.model.transformer.transformer import TransformerBlock


def test_block_shape():
    block = TransformerBlock(hidden_size=256, num_heads_q=4, num_heads_kv=2,
                            head_dim=64, layer_idx=4)
    x = mx.random.normal((1, 64, 256))
    assert block(x).shape == (1, 64, 256)


def test_gelu_layer_uses_gelu():
    from davinci_mlx.model.transformer.feed_forward import GELU7FFN
    block = TransformerBlock(hidden_size=256, num_heads_q=4, num_heads_kv=2,
                            head_dim=64, layer_idx=0)
    assert isinstance(block.mlp.ffn, GELU7FFN)


def test_swiglu_layer_uses_swiglu():
    from davinci_mlx.model.transformer.feed_forward import SwiGLU7FFN
    block = TransformerBlock(hidden_size=256, num_heads_q=4, num_heads_kv=2,
                            head_dim=64, layer_idx=5)
    assert isinstance(block.mlp.ffn, SwiGLU7FFN)


def test_residual_connection():
    """Output should differ from input (residual adds attention+FFN contribution)."""
    block = TransformerBlock(hidden_size=256, num_heads_q=4, num_heads_kv=2,
                            head_dim=64, layer_idx=4)
    x = mx.random.normal((1, 16, 256))
    result = block(x)
    diff = mx.max(mx.abs(result - x)).item()
    assert diff > 0.001, "Residual should add non-trivial contribution"
