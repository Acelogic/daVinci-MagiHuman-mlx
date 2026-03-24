"""Tests for weight converter."""
import mlx.core as mx
from davinci_mlx.loader.weight_converter import _convert_key, _extract_video_expert


def test_convert_adapter_keys():
    key, moe, _ = _convert_key("adapter.video_embedder.weight")
    assert key == "video_embedder.weight"
    assert not moe


def test_convert_adapter_bias():
    key, moe, _ = _convert_key("adapter.video_embedder.bias")
    assert key == "video_embedder.bias"
    assert not moe


def test_convert_shared_layer():
    key, moe, idx = _convert_key("block.layers.10.attention.linear_qkv.weight")
    assert key == "blocks.10.attention.attn.linear_qkv.weight"
    assert not moe  # layer 10 is shared
    assert idx == 10


def test_convert_moe_layer():
    key, moe, idx = _convert_key("block.layers.0.attention.linear_qkv.weight")
    assert key == "blocks.0.attention.attn.linear_qkv.weight"
    assert moe  # layer 0 is MoE
    assert idx == 0


def test_convert_moe_layer_36():
    key, moe, idx = _convert_key("block.layers.36.mlp.up_gate_proj.weight")
    assert key == "blocks.36.mlp.ffn.up_gate_proj.weight"
    assert moe
    assert idx == 36


def test_convert_attention_norms():
    key, moe, idx = _convert_key("block.layers.5.attention.q_norm.weight")
    assert key == "blocks.5.attention.attn.q_norm_weight"
    assert not moe
    assert idx == 5


def test_convert_mlp_keys():
    key, _, _ = _convert_key("block.layers.10.mlp.down_proj.weight")
    assert key == "blocks.10.mlp.ffn.down_proj.weight"


def test_convert_gelu_up_proj():
    """GELU layers (0-3) use up_proj instead of up_gate_proj."""
    key, moe, idx = _convert_key("block.layers.2.mlp.up_proj.weight")
    assert key == "blocks.2.mlp.ffn.up_proj.weight"
    assert moe
    assert idx == 2


def test_convert_pre_norm():
    key, _, _ = _convert_key("block.layers.10.attention.pre_norm.weight")
    assert key == "blocks.10.attention.pre_norm.weight"


def test_skip_audio():
    key, _, _ = _convert_key("adapter.audio_embedder.weight")
    assert key is None


def test_skip_audio_bias():
    key, _, _ = _convert_key("adapter.audio_embedder.bias")
    assert key is None


def test_skip_final_audio():
    key, _, _ = _convert_key("final_norm_audio.weight")
    assert key is None


def test_skip_unknown():
    key, _, _ = _convert_key("some_random_key")
    assert key is None


def test_extract_video_expert_2d():
    # Simulate 3-expert weight: (3*5120, 10) -> (5120, 10)
    weight = mx.arange(3 * 5120 * 10).reshape(3 * 5120, 10).astype(mx.float32)
    result = _extract_video_expert(weight)
    assert result.shape == (5120, 10)
    # Should be the first 5120 rows
    assert mx.array_equal(result, weight[:5120, :])


def test_extract_video_expert_1d():
    # Simulate 3-expert norm: (3*128,) -> (128,)
    weight = mx.arange(384).astype(mx.float32)
    result = _extract_video_expert(weight)
    assert result.shape == (128,)
    assert mx.array_equal(result, weight[:128])


def test_extract_video_expert_1d_pre_norm():
    # pre_norm.weight: (3*5120,) -> (5120,)
    weight = mx.arange(15360).astype(mx.float32)
    result = _extract_video_expert(weight)
    assert result.shape == (5120,)
    assert mx.array_equal(result, weight[:5120])


def test_convert_final_keys():
    key, moe, _ = _convert_key("final_norm_video.weight")
    assert key == "final_norm_video.weight"
    assert not moe


def test_convert_final_linear():
    key, moe, _ = _convert_key("final_linear_video.weight")
    assert key == "final_linear_video.weight"
    assert not moe
