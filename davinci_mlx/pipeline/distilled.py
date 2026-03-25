"""Distilled 8-step generation pipeline.

Flow:
1. Text encode (load -> encode -> unload)
2. Initialize noise
3. Denoise (load transformer -> 8 Euler steps -> unload)
4. VAE decode (load -> decode -> unload)
5. Save video
"""
import gc
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

from davinci_mlx.model.transformer.model import DaVinciModel
from davinci_mlx.loader.weight_converter import convert_and_load
from davinci_mlx.model.turbo_vae.decoder import TurboVAEDecoder, load_turbo_vae_weights
from davinci_mlx.components.scheduler import FlowMatchingScheduler
from davinci_mlx.components.patchifier import VideoLatentPatchifier
from davinci_mlx.pipeline.common import (
    validate_dimensions,
    compute_latent_shape,
    video_to_numpy,
    save_video as _save_video,
)


class DistilledPipeline:
    def __init__(self, weights_dir: str = "weights/original", precision: str = "float16"):
        self.weights_dir = Path(weights_dir)
        self.dtype = mx.float16 if precision == "float16" else mx.float32
        self.scheduler = FlowMatchingScheduler()
        self.patchifier = VideoLatentPatchifier(patch_size=2)

    def _load_text_encoder(self):
        """Load and return a TextEncoder instance. Raises helpful error if deps missing."""
        try:
            from davinci_mlx.model.text_encoder.encoder import TextEncoder
        except ImportError:
            raise ImportError(
                "Text encoder requires 'transformers' and 'torch'. "
                "Install them with:\n"
                "  pip install transformers torch\n"
            )
        return TextEncoder()

    def generate(
        self,
        prompt: str,
        height: int = 256,
        width: int = 256,
        num_frames: int = 65,
        steps: int = 8,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate video from text prompt. Returns (T, H, W, C) uint8 numpy."""
        validate_dimensions(height, width, num_frames)
        mx.random.seed(seed)

        # 1. Text encoding
        print("Loading text encoder...")
        try:
            encoder = self._load_text_encoder()
            encoder.load()
            text_embeddings = encoder.encode(prompt)
            print(f"  Text embeddings: {text_embeddings.shape}")
            encoder.unload()
            del encoder
            gc.collect()
        except ImportError as e:
            print(f"  Warning: {e}")
            print("  Falling back to zero embeddings (output will be meaningless).")
            text_embeddings = mx.zeros((1, 64, 3584), dtype=self.dtype)

        # 2. Initialize noise latent
        latent_shape = compute_latent_shape(height, width, num_frames)
        latent = mx.random.normal((1, *latent_shape)).astype(self.dtype)

        # 3. Denoise
        print("Loading transformer...")
        model = DaVinciModel()
        convert_and_load(model, self.weights_dir / "distill", target_dtype=self.dtype)

        sigmas = self.scheduler.get_sigmas(steps)

        for i in range(steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            print(f"  Step {i+1}/{steps} (sigma={float(sigma):.4f})")

            # Patchify: (1, 48, T, H, W) -> (1, N, 192)
            video_tokens = self.patchifier.patchify(latent)

            # Forward pass: predict velocity
            velocity_tokens = model(video_tokens, text_embeddings)

            # Unpatchify: (1, N, 192) -> (1, 48, T, H, W)
            velocity = self.patchifier.unpatchify(
                velocity_tokens,
                num_frames=latent_shape[1],
                height=latent_shape[2],
                width=latent_shape[3],
            )

            # Compute denoised prediction
            denoised = latent - sigma * velocity

            # Euler step
            latent = self.scheduler.step(latent, denoised, sigma, sigma_next)

            # CRITICAL: Force evaluation to prevent graph accumulation
            mx.eval(latent)

        del model
        gc.collect()

        # 4. VAE decode
        print("Loading VAE decoder...")
        vae = TurboVAEDecoder()
        load_turbo_vae_weights(
            vae,
            str(self.weights_dir / "turbo_vae" / "checkpoint-340000.ckpt"),
        )

        video = vae(latent)
        mx.eval(video)
        print(f"  VAE output: {video.shape}")

        del vae
        gc.collect()

        # 5. Convert to numpy and return
        print("Done.")
        return video_to_numpy(video)

    def save_video(self, frames: np.ndarray, path: str, fps: int = 24):
        _save_video(frames, path, fps)
