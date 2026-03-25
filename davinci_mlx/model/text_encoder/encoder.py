"""T5-Gemma text encoder: produces (1, seq_len, 3584) embeddings.

Uses HuggingFace transformers with CPU-only PyTorch for encoding,
then converts output to MLX array. The encoder is loaded once,
used to encode the prompt, then unloaded to free memory.
"""
import gc

import mlx.core as mx


class TextEncoder:
    """Encodes text prompts to embeddings for the DiT transformer."""

    def __init__(self, model_name: str = "google/t5gemma-9b-9b-ul2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load T5-Gemma encoder and tokenizer."""
        import torch
        from transformers import AutoTokenizer, AutoModel

        print(f"  Loading text encoder: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            is_encoder_decoder=False,
        )
        self.model.eval()
        print("  Text encoder loaded.")

    def unload(self):
        """Free model from memory."""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()

    def encode(self, prompt: str, max_length: int = 512) -> mx.array:
        """Encode a text prompt to embeddings.

        Args:
            prompt: Text prompt string.
            max_length: Maximum token length.

        Returns:
            MLX array of shape (1, seq_len, 3584).
        """
        if self.model is None:
            self.load()

        import torch

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )

        with torch.inference_mode():
            outputs = self.model(**inputs)

        # Extract last hidden state and convert to MLX
        hidden_state = outputs.last_hidden_state  # (1, seq_len, 3584)
        embeddings = mx.array(hidden_state.numpy()).astype(mx.float16)

        return embeddings
