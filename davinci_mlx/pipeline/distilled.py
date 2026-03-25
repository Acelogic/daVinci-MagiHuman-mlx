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


# Hardcoded latent normalization constants from daVinci source (48 channels)
LATENT_MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]
LATENT_STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]


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
                latent_mean=LATENT_MEAN,
                latent_std=LATENT_STD,
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
