"""Tests for full DaVinci DiT model."""
import mlx.core as mx
from davinci_mlx.model.transformer.model import DaVinciModel


def test_model_output_shape():
    model = DaVinciModel(hidden_size=256, num_layers=4, num_heads_q=4,
                         num_heads_kv=2, head_dim=64, video_in_channels=192,
                         text_in_channels=3584)
    video_tokens = mx.random.normal((1, 8, 192))
    text_tokens = mx.random.normal((1, 4, 3584))
    result = model(video_tokens, text_tokens)
    assert result.shape == (1, 8, 192), f"Got {result.shape}"


def test_model_default_40_layers():
    model = DaVinciModel()
    assert len(model.blocks) == 40


def test_model_video_only_output():
    """Model should only output predictions for video tokens, not text."""
    model = DaVinciModel(hidden_size=256, num_layers=2, num_heads_q=4,
                         num_heads_kv=2, head_dim=64, video_in_channels=192,
                         text_in_channels=3584)
    video_tokens = mx.random.normal((1, 16, 192))
    text_tokens = mx.random.normal((1, 8, 3584))
    result = model(video_tokens, text_tokens)
    # Output should have same seq_len as video input, not total
    assert result.shape[1] == 16, f"Expected 16 video tokens, got {result.shape[1]}"
