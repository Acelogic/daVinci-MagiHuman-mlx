# daVinci-MagiHuman MLX

> **WIP — This project is under active development and does not produce usable video output yet.** The full pipeline runs end-to-end (text encoding, 10.9B transformer denoising, VAE decode) but output quality is not correct — likely due to scheduler tuning, positional encoding, or VAE denormalization issues still being debugged. Contributions and investigation welcome.

Native Apple Silicon port of [daVinci-MagiHuman](https://github.com/GAIR-NLP/daVinci-MagiHuman) — a 15B parameter video generation model — running entirely on MLX.

Generate talking-head and general video from text prompts on your Mac, no GPU server required.

## Features

- **Native MLX** — No PyTorch dependency for inference (only for text encoding)
- **10.9B parameters** loaded from the distilled model (video-only slice of the 15B model)
- **8-step distilled generation** — flow-matching Euler sampling, no classifier-free guidance needed
- **Sequential model loading** — text encoder → transformer → VAE, one at a time to fit in memory
- **INT4 quantization** — reduce transformer from ~22GB to ~5.5GB via `mlx.nn.quantize`
- **Custom Metal kernel** — fused `silu_mul` for SwiGLU acceleration
- **256p + 540p** resolution support (540p via super-resolution, coming soon)

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- **128GB+ unified memory** recommended (for FP16 weights)
- macOS 14+ with MLX 0.25+
- Python 3.12+
- ~60GB disk space for model weights

## Quick Start

```bash
# Clone and install
git clone https://github.com/Acelogic/daVinci-MagiHuman-mlx.git
cd daVinci-MagiHuman-mlx
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Install text encoder dependencies (optional but recommended)
pip install transformers torch accelerate

# Download model weights (~59GB)
python scripts/download_weights.py --components distill turbo_vae

# Generate a video
python scripts/generate.py \
    --prompt "A golden retriever running through a meadow" \
    --height 256 --width 256 --frames 17 --steps 8 \
    --output output.mp4
```

## Python API

```python
from davinci_mlx.pipeline.distilled import DistilledPipeline

pipe = DistilledPipeline(weights_dir="weights/original")
video = pipe.generate(
    prompt="A woman walking through a garden",
    height=256, width=256,
    num_frames=65, steps=8, seed=42,
)
pipe.save_video(video, "output.mp4")
```

## Architecture

daVinci-MagiHuman uses a single-stream DiT (Diffusion Transformer) where video and text tokens are concatenated and processed together via self-attention:

```
Text Prompt → T5-Gemma 9B Encoder → text embeddings (3584-dim)
                                          ↓
Random Noise → Patchify → [video tokens ∥ text tokens]
                                          ↓
                          40-layer Transformer (10.9B params)
                          ├─ Layers 0-3: modality-specific (GELU7)
                          ├─ Layers 4-35: shared (SwiGLU7)
                          └─ Layers 36-39: modality-specific (SwiGLU7)
                                          ↓
                          8 flow-matching Euler steps
                                          ↓
                          Turbo VAE Decoder → video frames
                                          ↓
                                       MP4 output
```

Key specs:
- Hidden dim: 5120, head dim: 128
- GQA: 40 query heads, 8 KV heads (5:1 ratio)
- Attention gating: `output * sigmoid(gate)` with 40 gate scalars
- Clamped activations: SwiGLU7/GELU7 clip intermediate to [-7, 7]
- VAE: 48 latent channels, 16x spatial / 4x temporal compression

## Performance

Tested on M3 Max (128GB) at 256x256, 17 input frames, 8 denoise steps:

| Stage | Time |
|-------|------|
| Text encoding (T5-Gemma 9B) | ~35s |
| Transformer weight loading | ~30s |
| Denoising (8 steps) | ~25s |
| VAE decode | ~15s |
| **Total** | **~5 min** |

With INT4 quantization, transformer loading drops significantly. Weight loading time depends on storage speed (SSD vs external drive).

## INT4 Quantization

```bash
# Quantize transformer weights (reduces ~22GB → ~5.5GB)
python scripts/quantize_weights.py \
    --input weights/original/distill \
    --output weights/int4/transformer \
    --bits 4
```

## Project Structure

```
davinci_mlx/
├── model/
│   ├── transformer/     # 40-layer DiT (attention, FFN, RoPE)
│   ├── turbo_vae/       # Causal 3D conv VAE decoder
│   └── text_encoder/    # T5-Gemma wrapper
├── loader/              # Weight conversion + quantization
├── pipeline/            # Distilled generation pipeline
├── components/          # Scheduler, patchifier
└── kernels/             # Metal fused ops
```

## Status

- [x] Distilled 8-step video generation
- [x] 256p resolution
- [x] INT4 quantization support
- [x] Real weight loading (10.9B params)
- [x] T5-Gemma text encoding
- [x] Turbo VAE decoding
- [ ] 540p super-resolution
- [ ] Audio generation
- [ ] Image-to-video conditioning

## Credits

- [GAIR-NLP/daVinci-MagiHuman](https://github.com/GAIR-NLP/daVinci-MagiHuman) — Original model
- [LTX-2-MLX](https://github.com/Acelogic/LTX-2-MLX) — Reference MLX video model port
- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework

## License

Apache 2.0 (same as the original model)
