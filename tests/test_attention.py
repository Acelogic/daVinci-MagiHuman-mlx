"""Tests for GQA attention with gating."""
import mlx.core as mx
from davinci_mlx.model.transformer.attention import Attention


def test_attention_output_shape():
    attn = Attention(hidden_size=256, num_heads_q=4, num_heads_kv=2, head_dim=64)
    x = mx.random.normal((1, 16, 256))
    result = attn(x)
    assert result.shape == (1, 16, 256), f"Got {result.shape}"


def test_qkvg_projection_shape():
    attn = Attention(hidden_size=5120, num_heads_q=40, num_heads_kv=8, head_dim=128)
    # Q: 40*128=5120, K: 8*128=1024, V: 8*128=1024, G: 40 (one per head) = 7208
    assert attn.linear_qkv.weight.shape == (7208, 5120), f"Got {attn.linear_qkv.weight.shape}"


def test_attention_with_rope():
    from davinci_mlx.model.transformer.rope import precompute_freqs
    attn = Attention(hidden_size=256, num_heads_q=4, num_heads_kv=2, head_dim=64)
    x = mx.random.normal((1, 16, 256))
    cos_f, sin_f = precompute_freqs(dim=64, max_pos=16)
    positions = mx.arange(16)
    result = attn(x, cos_freqs=cos_f, sin_freqs=sin_f, positions=positions)
    assert result.shape == (1, 16, 256), f"Got {result.shape}"
