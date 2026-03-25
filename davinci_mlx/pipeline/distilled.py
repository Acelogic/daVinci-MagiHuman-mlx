"""Distilled 8-step generation pipeline.

Flow:
1. Text encode (load -> encode -> unload)
2. Initialize noise
3. Denoise (load transformer -> 8 Euler steps -> unload)
4. VAE decode (load -> decode -> unload)
5. Save video
"""
import gc
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from tqdm import tqdm

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
        """Load and return a TextEncoder instance."""
        try:
            from davinci_mlx.model.text_encoder.encoder import TextEncoder
        except ImportError:
            raise ImportError(
                "Text encoder requires 'transformers' and 'torch'. "
                "Install with: pip install transformers torch"
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
        total_start = time.time()

        # ── 1. Text encoding ──────────────────────────────────────────
        try:
            encoder = self._load_text_encoder()
            with tqdm(total=3, desc="Text encoder", bar_format="{desc}: {bar} {n_fmt}/{total_fmt}") as pbar:
                pbar.set_postfix_str("loading model...")
                encoder.load()
                pbar.update(1)

                pbar.set_postfix_str("encoding prompt...")
                text_embeddings = encoder.encode(prompt)
                pbar.update(1)

                pbar.set_postfix_str("unloading...")
                encoder.unload()
                del encoder
                gc.collect()
                pbar.update(1)
                pbar.set_postfix_str(f"done ({text_embeddings.shape[1]} tokens)")
        except ImportError:
            print("  Text encoder not available (install transformers + torch)")
            print("  Using zero embeddings — output will be noise")
            text_embeddings = mx.zeros((1, 64, 3584), dtype=self.dtype)

        # ── 2. Initialize noise latent ────────────────────────────────
        latent_shape = compute_latent_shape(height, width, num_frames)
        latent = mx.random.normal((1, *latent_shape)).astype(self.dtype)

        # ── 3. Load transformer ───────────────────────────────────────
        model = DaVinciModel()
        convert_and_load(model, self.weights_dir / "distill", target_dtype=self.dtype)

        # ── 4. Denoise ────────────────────────────────────────────────
        sigmas = self.scheduler.get_sigmas(steps)

        denoise_pbar = tqdm(
            range(steps),
            desc="Denoising",
            bar_format="{desc}: {bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        for i in denoise_pbar:
            step_start = time.time()
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            video_tokens = self.patchifier.patchify(latent)
            velocity_tokens = model(video_tokens, text_embeddings)
            velocity = self.patchifier.unpatchify(
                velocity_tokens,
                num_frames=latent_shape[1],
                height=latent_shape[2],
                width=latent_shape[3],
            )

            denoised = latent - sigma * velocity
            latent = self.scheduler.step(latent, denoised, sigma, sigma_next)
            mx.eval(latent)

            step_time = time.time() - step_start
            denoise_pbar.set_postfix_str(f"sigma={float(sigma):.3f}, {step_time:.1f}s/step")
        denoise_pbar.close()

        del model
        gc.collect()

        # ── 5. VAE decode ─────────────────────────────────────────────
        with tqdm(total=3, desc="VAE decode", bar_format="{desc}: {bar} {n_fmt}/{total_fmt}") as pbar:
            pbar.set_postfix_str("loading weights...")
            vae = TurboVAEDecoder()
            load_turbo_vae_weights(
                vae,
                str(self.weights_dir / "turbo_vae" / "checkpoint-340000.ckpt"),
            )
            pbar.update(1)

            pbar.set_postfix_str("decoding latents...")
            video = vae(latent)
            mx.eval(video)
            pbar.update(1)

            pbar.set_postfix_str("cleanup...")
            del vae
            gc.collect()
            pbar.update(1)

            T_out = video.shape[2]
            pbar.set_postfix_str(f"done ({T_out} frames)")

        total_time = time.time() - total_start
        print(f"\nGeneration complete: {height}x{width}, {T_out} frames in {total_time:.1f}s")

        return video_to_numpy(video)

    def save_video(self, frames: np.ndarray, path: str, fps: int = 24):
        _save_video(frames, path, fps)
