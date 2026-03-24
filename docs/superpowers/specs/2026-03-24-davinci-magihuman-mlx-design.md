# daVinci-MagiHuman MLX Port — Design Spec

**Date**: 2026-03-24
**Status**: Approved
**Reference project**: ~/Developer/LTX-2-MLX/

## Scope

Port the daVinci-MagiHuman 15B video generation model from PyTorch/CUDA to native Apple Silicon via MLX.

### In scope
- Distilled model (8-step, no CFG)
- Video-only generation (no audio)
- 256p native generation
- 540p via latent-space super-resolution
- FP16 weights with INT4 quantization path
- Sequential model loading for M3 Max 128GB
- Python API + CLI wrapper

### Out of scope
- Audio generation
- Base model (32-step CFG pipeline)
- 1080p super-resolution
- Training / fine-tuning
- LoRA support

## Target Hardware

- Apple M3 Max, 128GB unified memory
- Sequential model loading: only one major component in memory at a time
- Peak memory budget: ~30GB for transformer (FP16) + activations, ~7.5GB (INT4)

## Architecture

### Source Model (PyTorch)

15B unified transformer with sandwich architecture:
- Layers 0-3: Modality-specific input projection (video)
- Layers 4-35: Shared transformer blocks
- Layers 36-39: Modality-specific output projection (video)

Per-block structure:
- RMSNorm -> Self-Attention (RoPE) -> residual
- RMSNorm -> Cross-Attention (to text) -> residual
- RMSNorm -> SwiGLU FFN -> residual
- AdaLN modulation from timestep embedding

Key dimensions:
- Hidden: 5120
- Attention heads: 128 (40-dim per head)
- Video input channels: 192
- Text input channels: 3584
- Grouped-query attention

### External Dependencies
- Text encoder: T5-Gemma 9B (native MLX implementation, no PyTorch)
- VAE: Turbo VAE decoder (distilled from Wan2.2)
- SR: 540p latent-space super-resolution model

## Project Structure

```
~/Developer/daVinci-MagiHuman-mlx/
├── davinci_mlx/
│   ├── model/
│   │   ├── transformer/
│   │   │   ├── model.py            # DaVinciModel - unified forward pass
│   │   │   ├── transformer.py      # TransformerBlock with AdaLN
│   │   │   ├── attention.py        # Multi-head attention with RoPE
│   │   │   ├── feed_forward.py     # SwiGLU feed-forward
│   │   │   ├── rope.py             # Rotary position embeddings
│   │   │   └── modality.py         # Modality-specific layer routing
│   │   ├── vae/
│   │   │   ├── decoder.py          # Wan2.2 VAE decoder (latent -> video)
│   │   │   └── conv3d.py           # Causal 3D convolutions
│   │   ├── turbo_vae/
│   │   │   └── decoder.py          # Distilled fast VAE decoder
│   │   ├── text_encoder/
│   │   │   ├── encoder.py          # Encoding pipeline
│   │   │   └── t5_gemma.py         # Native MLX T5-Gemma implementation
│   │   └── super_resolution/
│   │       └── model.py            # 540p latent-space upscaler
│   ├── loader/
│   │   ├── weight_converter.py     # PyTorch safetensors -> MLX
│   │   └── quantize.py             # FP16 -> INT4 conversion
│   ├── pipeline/
│   │   ├── distilled.py            # 8-step distilled generation
│   │   └── common.py               # Shared utilities
│   ├── components/
│   │   ├── scheduler.py            # UniPC flow-matching scheduler
│   │   ├── diffusion_steps.py      # Euler stepping
│   │   └── patchifier.py           # Latent <-> sequence conversion
│   └── __init__.py
├── scripts/
│   ├── generate.py                 # CLI entry point
│   ├── download_weights.py         # HF download -> SSD
│   └── quantize_weights.py         # FP16 -> INT4 offline
├── weights -> /Volumes/Untitled/ai-models/davinci-magihuman-mlx/weights
├── pyproject.toml
└── README.md
```

## Weight Management

### Download and Storage

All weights stored on external SSD:
```
/Volumes/Untitled/ai-models/davinci-magihuman-mlx/
├── weights/
│   ├── original/          # Raw HF download (sharded safetensors)
│   ├── fp16/              # Converted MLX weights
│   │   ├── transformer/
│   │   ├── text_encoder/
│   │   ├── turbo_vae/
│   │   └── sr_540p/
│   └── int4/              # Quantized MLX weights
│       ├── transformer/
│       ├── text_encoder/
│       ├── turbo_vae/
│       └── sr_540p/
```

Symlinked from project: `weights -> /Volumes/Untitled/ai-models/davinci-magihuman-mlx/weights`

### Conversion Pipeline

```
HuggingFace (sharded safetensors)
  -> download_weights.py (fetch to SSD original/)
  -> weight_converter.py (remap keys, cast FP16, save as MLX safetensors to fp16/)
  -> quantize_weights.py (FP16 -> INT4 via mlx.nn.QuantizedLinear to int4/)
```

### Key Mapping (PyTorch -> MLX)

- Linear weights: same storage format, no transpose
- RMSNorm weights: direct copy
- Modality-specific layers: route to `modality_blocks_in.{i}` / `modality_blocks_out.{i}`
- Shared layers: route to `blocks.{i}`

### Streaming Load

- Load one safetensor shard at a time
- Convert and assign weights immediately
- GC every ~100 weights
- Build nested dict for `model.update()`

## Inference Pipeline

### Distilled Pipeline (8-step, no CFG)

```
1. Load text_encoder -> encode prompt -> unload     (~18GB peak)
2. Initialize noise latent
3. Load transformer -> 8 Euler denoise steps -> unload  (~30GB FP16 / ~7.5GB INT4)
4. Load turbo_vae -> decode latents to video -> unload   (~small)
5. (Optional) Load sr_540p -> upscale -> unload
6. Save MP4 via imageio
```

### Memory Management

Sequential load/unload pattern:
- Load weights from safetensors into model via `model.update()`
- Materialize in unified memory with `mx.eval(model.parameters())`
- After use, replace weights with empty arrays and call `gc.collect()`

### Scheduler

UniPC flow-matching scheduler with fixed sigma schedule for distilled model.
Ported from daVinci's `scheduler_unipc.py`.

### Compiled Kernels

Following LTX-2-MLX patterns:
- `@mx.compile` for fused AdaLN (RMSNorm + scale/shift)
- `@mx.compile` for fused attention core
- `mx.fast.rms_norm()` and `mx.fast.scaled_dot_product_attention()`

## CLI Interface

```bash
# 256p generation
python scripts/generate.py --prompt "A woman walking through a garden" \
    --height 256 --width 256 --frames 65 --steps 8

# 540p with super-resolution
python scripts/generate.py --prompt "..." --height 540 --width 960 --steps 8 --sr

# INT4 quantization
python scripts/generate.py --prompt "..." --precision int4

# Download weights
python scripts/download_weights.py \
    --output /Volumes/Untitled/ai-models/davinci-magihuman-mlx/weights/original

# Quantize
python scripts/quantize_weights.py --input weights/fp16 --output weights/int4 --bits 4
```

## Python API

```python
from davinci_mlx.pipeline.distilled import DistilledPipeline

pipe = DistilledPipeline(
    weights_dir="weights/fp16",
    precision="float16",
)

video = pipe.generate(
    prompt="A woman walking through a garden",
    height=256, width=256,
    num_frames=65, steps=8, seed=42,
)
pipe.save_video(video, "output.mp4", fps=24)
```

## Dependencies

```toml
[project]
name = "davinci-magihuman-mlx"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "mlx>=0.25.0",
    "safetensors",
    "numpy",
    "imageio[ffmpeg]",
    "huggingface-hub",
    "sentencepiece",
    "tqdm",
]
```

No PyTorch dependency. All components implemented natively in MLX.
