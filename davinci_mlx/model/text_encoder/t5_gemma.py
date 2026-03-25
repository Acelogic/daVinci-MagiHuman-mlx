"""T5-Gemma text encoder module.

The actual implementation uses HuggingFace transformers.
See encoder.py for the TextEncoder class.
"""
from davinci_mlx.model.text_encoder.encoder import TextEncoder

__all__ = ["TextEncoder"]
