"""Tests for text encoder interface."""
from davinci_mlx.model.text_encoder.encoder import TextEncoder


def test_encoder_init():
    """TextEncoder should initialize with default model name."""
    enc = TextEncoder()
    assert enc.model_name == "google/t5gemma-9b-9b-ul2"
    assert enc.model is None
    assert enc.tokenizer is None


def test_encoder_custom_model():
    """TextEncoder should accept custom model name."""
    enc = TextEncoder(model_name="google/t5gemma-2b-2b-ul2")
    assert enc.model_name == "google/t5gemma-2b-2b-ul2"


def test_encoder_unload_when_not_loaded():
    """Unload should be safe even when model isn't loaded."""
    enc = TextEncoder()
    enc.unload()  # Should not raise
    assert enc.model is None
