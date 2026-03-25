"""T5-Gemma text encoder: produces (1, seq_len, 3584) embeddings.

Uses HuggingFace transformers with CPU-only PyTorch for encoding,
then converts output to MLX array. The encoder is loaded once,
used to encode the prompt, then unloaded to free memory.

Only the encoder half of T5-Gemma is used (not the decoder).
"""
import gc

import mlx.core as mx


class TextEncoder:
    """Encodes text prompts to embeddings for the DiT transformer."""

    def __init__(self, model_name: str = "google/t5gemma-9b-9b-ul2"):
        self.model_name = model_name
        self.encoder = None
        self.tokenizer = None

    def load(self):
        """Load T5-Gemma encoder (only) and tokenizer."""
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        print(f"  Loading text encoder: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load full model, extract encoder, free decoder to save memory
        full_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            dtype=torch.float32,
            device_map="cpu",
        )
        # Encoder is at model.model.encoder (not model.encoder)
        self.encoder = full_model.model.encoder
        self.encoder.eval()

        # Free decoder + lm_head weights
        del full_model.model.decoder
        del full_model.lm_head
        del full_model
        gc.collect()

        print("  Text encoder loaded (encoder only).")

    def unload(self):
        """Free model from memory."""
        self.encoder = None
        self.tokenizer = None
        gc.collect()

    def encode(self, prompt: str, max_length: int = 512) -> mx.array:
        """Encode a text prompt to embeddings.

        Returns: MLX array of shape (1, seq_len, 3584).
        """
        if self.encoder is None:
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
            outputs = self.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        hidden_state = outputs.last_hidden_state  # (1, seq_len, 3584)
        embeddings = mx.array(hidden_state.numpy()).astype(mx.float16)
        return embeddings
