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
- VAE encoder (text-to-video only, no image-to-video conditioning)

## Target Hardware

- Apple M3 Max, 128GB unified memory
- Sequential model loading: only one major component in memory at a time

### Memory Budget (FP16)

| Stage | Weights | Activations (256p) | Peak |
|-------|---------|-------------------|------|
| Text encoder (9B) | ~18 GB | ~2 GB | ~20 GB |
| Transformer (15B) | ~30 GB | ~5-10 GB | ~35-40 GB |
| Turbo VAE decode | ~2 GB | ~3 GB | ~5 GB |
| SR 540p | TBD | TBD | TBD |

With INT4: transformer weights drop to ~7.5 GB, text encoder to ~4.5 GB.

Activation memory assumes `mx.fast.scaled_dot_product_attention` (memory-efficient).
Without fused attention, per-layer attention maps would be ~256 MB/layer (40 layers = ~10 GB).

### I/O Budget

Samsung T7 Shield SSD via USB 3.2 Gen 2: ~1000 MB/s practical.
Loading 30 GB transformer from SSD: ~30 seconds.
Full pipeline I/O (text encoder + transformer + VAE + SR): ~1-2 minutes of weight loading.

## Architecture

### Source Model (PyTorch)

15B unified transformer with sandwich architecture:
- Layers 0-3: Modality-specific input projection (GELU7 FFN, non-gated)
- Layers 4-35: Shared transformer blocks (SwiGLU7 FFN, gated)
- Layers 36-39: Modality-specific output projection (SwiGLU7 FFN, gated)

Per-block structure:
- RMSNorm -> Self-Attention (RoPE, SPLIT variant) -> residual
- RMSNorm -> Cross-Attention (to text) -> residual
- RMSNorm -> FFN (GELU7 or SwiGLU7 depending on layer) -> residual
- AdaLN modulation from timestep embedding

Key dimensions:
- Hidden: 5120
- Head dim: 128
- Query heads: 40 (5120 / 128)
- KV heads: 8 (grouped-query attention, 5:1 ratio)
- Video input channels: 192 (= 48 VAE latent channels x 4 from 2x2 spatial patch)
- Text input channels: 3584
- FFN intermediate (SwiGLU7): 13652 (= int(5120 * 4 * 2/3) // 4 * 4, gated)
- FFN intermediate (GELU7): 20480 (= 5120 * 4, non-gated)

### VAE Latent Space
- VAE latent channels: 48 (z_dim)
- VAE stride: (4, 16, 16) -- temporal 4x, spatial 16x16
- Patch size: (1, 2, 2) -- temporal 1, spatial 2x2
- Effective input channels: 48 * 1 * 2 * 2 = 192

### External Dependencies
- Text encoder: T5-Gemma 9B (native MLX implementation, no PyTorch)
- VAE: Turbo VAE decoder (TurboV3-Wan22-TinyShallow, decode-only)
- SR: 540p latent-space super-resolution model

## Project Structure

```
~/Developer/daVinci-MagiHuman-mlx/
├── davinci_mlx/
│   ├── model/
│   │   ├── transformer/
│   │   │   ├── model.py                # DaVinciModel - unified forward pass
│   │   │   ├── transformer.py          # TransformerBlock with AdaLN
│   │   │   ├── attention.py            # GQA attention (40Q/8KV) with RoPE
│   │   │   ├── feed_forward.py         # SwiGLU7 (gated) and GELU7 (non-gated)
│   │   │   ├── rope.py                 # Rotary position embeddings (SPLIT)
│   │   │   ├── timestep_embedding.py   # Timestep embedding + AdaLN modulation
│   │   │   └── modality.py             # Modality-specific layer routing
│   │   ├── turbo_vae/
│   │   │   ├── decoder.py              # Distilled fast VAE decoder
│   │   │   ├── conv3d.py               # Causal 3D convolutions (via 2D reshape)
│   │   │   └── tiling.py               # Tiled decoding for 540p memory management
│   │   ├── text_encoder/
│   │   │   ├── encoder.py              # Encoding pipeline + text projection
│   │   │   └── t5_gemma.py             # Native MLX T5-Gemma implementation
│   │   └── super_resolution/
│   │       └── model.py                # 540p latent-space upscaler
│   ├── loader/
│   │   ├── weight_converter.py         # PyTorch safetensors -> MLX key remapping
│   │   └── quantize.py                 # FP16 -> INT4 conversion
│   ├── pipeline/
│   │   ├── distilled.py                # 8-step distilled generation
│   │   └── common.py                   # Shared utilities, input validation
│   ├── components/
│   │   ├── scheduler.py                # Flow-matching scheduler (DDIM step)
│   │   ├── patchifier.py               # Latent <-> sequence conversion
│   │   └── data_proxy.py               # Token sequence assembly
│   ├── kernels/
│   │   └── fused_ops.py                # Metal kernels: silu_mul for SwiGLU7
│   └── __init__.py
├── scripts/
│   ├── generate.py                     # CLI entry point
│   ├── download_weights.py             # HF download -> SSD
│   └── quantize_weights.py             # FP16 -> INT4 offline
├── tests/
│   └── test_weight_conversion.py       # Weight loading validation
├── weights -> /Volumes/Untitled/ai-models/davinci-magihuman-mlx/weights
├── pyproject.toml
├── .gitignore
└── README.md
```

Key differences from LTX-2-MLX:
- **GQA attention**: 40 query heads / 8 KV heads (5:1 ratio), vs LTX-2's standard MHA
- **Hybrid FFN**: Layers 0-3 use GELU7 (non-gated), layers 4-39 use SwiGLU7 (gated)
- **Sandwich routing**: Modality-specific first/last 4 layers, shared middle 32
- **No regular VAE**: Only turbo VAE (decode-only), no encoder needed for text-to-video
- **Causal 3D conv risk**: MLX has no native Conv3d; must implement via 2D reshape (same as LTX-2-MLX)

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

- Standard Linear weights: same storage format [out, in], no transpose needed
- GQA projections: Q is [5120, 5120], K/V are [1024, 5120] (8 KV heads x 128 dim)
- SwiGLU7 gated FFN: 3 weight matrices (gate, up, down) vs GELU7: 2 (in, out)
- RMSNorm weights: direct copy
- Modality-specific layers 0-3: route to `modality_blocks_in.{i}`
- Modality-specific layers 36-39: route to `modality_blocks_out.{i}`
- Shared layers 4-35: route to `blocks.{i}`
- Verify any fused QKV or gate/up projections and split if needed

### Streaming Load

- Load one safetensor shard at a time
- Convert and assign weights immediately
- GC every ~100 weights
- Build nested dict for `model.update()`

## Inference Pipeline

### Distilled Pipeline (8-step, no CFG)

```
1. Load text_encoder -> encode prompt -> unload        (~20 GB peak)
2. Initialize noise latent (48 channels, VAE stride 4/16/16)
3. Load transformer -> 8 DDIM/Euler denoise steps -> unload  (~35-40 GB FP16 / ~12 GB INT4)
4. Load turbo_vae -> decode latents to video -> unload  (~5 GB peak)
5. (Optional) Load sr_540p -> upscale -> unload
6. Save MP4 via imageio at 24fps
```

### Memory Management

Sequential load/unload pattern:
- Load weights from safetensors into model via `model.update()`
- Materialize in unified memory
- After use, replace weights with empty arrays and call `gc.collect()`

### Scheduler

FlowUniPCMultistepScheduler using `step_ddim` method for the distilled path.
This is effectively a flow-matching Euler/DDIM step: `prev_state = prev_t * noise + (1 - prev_t) * predicted_clean`.
Fixed sigma schedule for 8 steps. Ported from daVinci's `scheduler_unipc.py`.

### Compiled Kernels and Fused Ops

Following LTX-2-MLX patterns:
- `@mx.compile` for fused AdaLN: `_compiled_adaln_forward` (RMSNorm + scale/shift)
- `@mx.compile` for fused attention core: `_compiled_attention_core`
- `@mx.compile` for fused residual gate: `_compiled_residual_gate`
- `mx.fast.rms_norm()` for normalization
- `mx.fast.scaled_dot_product_attention()` for memory-efficient attention
- Custom Metal kernel `silu_mul` for SwiGLU7 fusion (in `kernels/fused_ops.py`)

## Validation

### Weight Conversion Validation
- Count total parameters after loading; compare against expected 15B
- Check for NaN/Inf in loaded weights
- Verify all expected keys present (no silent misses)

### Numerical Validation
- Compare single transformer block output: MLX vs PyTorch reference for same input
- Compare full pipeline output for same (seed, prompt, noise) inputs
- Acceptable tolerance: max absolute diff < 0.01 for FP16

### Input Validation
- Frame count constraints (must be compatible with VAE temporal stride 4)
- Resolution divisibility (must be compatible with VAE spatial stride 16 and patch size 2)
- Prompt length limits (T5-Gemma max tokens)

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

## Known Risks

- **Causal 3D convolution**: MLX has no native `nn.Conv3d`. Must implement via manual reshaping and 2D convolutions, same approach as LTX-2-MLX. This is a complexity hotspot.
- **SSD I/O latency**: Weight loading from external USB SSD adds ~30s per model swap. Total pipeline I/O overhead: ~1-2 minutes.
- **540p activation memory**: Tiled VAE decoding may be required at 540p to keep peak memory under 128 GB. The `tiling.py` module handles this.
- **SwiGLU7 clamping**: The source uses a clamped variant (output clamped to [-7, 7]). Must preserve this behavior exactly for numerical parity.
