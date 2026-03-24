# daVinci-MagiHuman MLX Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the daVinci-MagiHuman 15B distilled video generation model to native MLX for Apple Silicon (M3 Max 128GB).

**Architecture:** Single-stream 40-layer DiT where video + text tokens are concatenated and processed via self-attention (no cross-attention). Sandwich layers: modality-specific (0-3, 36-39) + shared (4-35). Sequential model loading (text encoder -> transformer -> VAE).

**Tech Stack:** MLX, safetensors, imageio, huggingface-hub, sentencepiece

**Spec:** `docs/superpowers/specs/2026-03-24-davinci-magihuman-mlx-design.md`

**Reference:** `~/Developer/LTX-2-MLX/` (proven MLX video diffusion port)

---

## Spec Corrections (Discovered During Source Analysis)

The following corrections to the spec were discovered by reading the actual daVinci source code. The plan implements the ACTUAL architecture:

1. **No cross-attention.** daVinci is single-stream: video + text tokens are concatenated into one sequence and attend to each other via self-attention. The spec incorrectly listed cross-attention.

2. **No AdaLN / timestep embedding.** The model is "timestep-free" -- it infers noise level from the noisy input directly. Remove `timestep_embedding.py` from project. No AdaLN modulation.

3. **Attention gating.** The QKV projection is actually QKV+G (gate). Output = `attn_output * sigmoid(gate)`. Not present in the spec.

4. **Fused QKV+Gate projection.** A single `linear_qkv` produces Q, K, V, and G concatenated. Must split after projection.

5. **ElementWiseFourierEmbed for positional encoding.** Not standard sinusoidal -- uses learnable Fourier bands.

6. **NativeMoELinear on modality-specific layers.** Layers 0-3 and 36-39 may use per-modality expert weights. Shared layers 4-35 use single-expert weights.

7. **MultiModalityRMSNorm.** RMSNorm with per-modality learned scale (1 or 3 expert sets).

8. **Actual weight key names** differ from spec assumptions. See Task 11 for exact mapping.

9. **NativeMoELinear handling.** Layers 0-3 and 36-39 may store per-modality expert weights (shape `[num_experts * out, in]`). For video-only distilled inference, we only need the video expert slice. Task 11 Step 1 MUST verify the actual weight shapes to determine if expert extraction is needed. If weights have standard shapes (no expert dimension), use plain `nn.Linear`. If expert-indexed, the weight converter must extract the video expert slice during conversion.

10. **MultiModalityRMSNorm simplification.** If the distilled video-only weights use single-expert norms (1 weight vector per layer), the plan's plain `RMSNorm` implementation is correct. If some layers have 3 expert norms, the converter must extract the video norm. Verify in Task 11 Step 1.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: all `__init__.py` files for package structure

- [ ] **Step 1: Create pyproject.toml**

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

[project.scripts]
davinci-generate = "scripts.generate:main"

[build-system]
requires = ["setuptools>=75.0"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
*.egg-info/
dist/
build/
.DS_Store
weights/
*.mp4
*.safetensors
```

- [ ] **Step 3: Create all `__init__.py` files**

Empty `__init__.py` for: `davinci_mlx/`, `davinci_mlx/model/`, `davinci_mlx/model/transformer/`, `davinci_mlx/model/turbo_vae/`, `davinci_mlx/model/text_encoder/`, `davinci_mlx/loader/`, `davinci_mlx/pipeline/`, `davinci_mlx/components/`, `davinci_mlx/kernels/`, `tests/`.

- [ ] **Step 4: Create SSD directory structure and symlink**

```bash
mkdir -p /Volumes/Untitled/ai-models/davinci-magihuman-mlx/weights/{original,fp16/{transformer,text_encoder,turbo_vae,sr_540p},int4/{transformer,text_encoder,turbo_vae,sr_540p}}
ln -s /Volumes/Untitled/ai-models/davinci-magihuman-mlx/weights weights
```

- [ ] **Step 5: Install package in dev mode**

```bash
cd ~/Developer/daVinci-MagiHuman-mlx
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore davinci_mlx/ tests/
git commit -m "feat: project scaffolding with package structure and SSD symlink"
```

---

### Task 2: Weight Download Script

**Files:**
- Create: `scripts/download_weights.py`

- [ ] **Step 1: Write download script**

```python
#!/usr/bin/env python3
"""Download daVinci-MagiHuman weights from HuggingFace to SSD."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download daVinci-MagiHuman weights")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Volumes/Untitled/ai-models/davinci-magihuman-mlx/weights/original"),
    )
    parser.add_argument("--repo-id", default="GAIR/daVinci-MagiHuman")
    parser.add_argument(
        "--components",
        nargs="+",
        default=["distill", "turbo_vae"],
        choices=["base", "distill", "turbo_vae", "540p_sr", "1080p_sr"],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    allow_patterns = []
    for component in args.components:
        allow_patterns.append(f"{component}/**")
    allow_patterns.extend(["*.json", "*.md"])

    print(f"Downloading {args.components} to {args.output}")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(args.output),
        allow_patterns=allow_patterns,
    )
    print("Download complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test help output**

Run: `python scripts/download_weights.py --help`
Expected: Shows help text.

- [ ] **Step 3: Commit**

```bash
git add scripts/download_weights.py
git commit -m "feat: add weight download script for HuggingFace"
```

- [ ] **Step 4: Download distill + turbo_vae weights to SSD**

Run: `python scripts/download_weights.py --components distill turbo_vae`
Expected: Downloads to SSD. Run in background, continue with Tasks 3-9 while downloading.

---

### Task 3: Core Components -- Patchifier

**Files:**
- Create: `davinci_mlx/components/patchifier.py`
- Create: `tests/test_patchifier.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for video latent patchifier."""
import mlx.core as mx
from davinci_mlx.components.patchifier import VideoLatentPatchifier


def test_patchify_shape():
    patchifier = VideoLatentPatchifier(patch_size=2)
    latent = mx.random.normal((1, 48, 4, 16, 16))
    result = patchifier.patchify(latent)
    # N = 4 * 8 * 8 = 256, D = 48 * 2 * 2 = 192
    assert result.shape == (1, 256, 192)


def test_unpatchify_roundtrip():
    patchifier = VideoLatentPatchifier(patch_size=2)
    latent = mx.random.normal((1, 48, 4, 16, 16))
    patchified = patchifier.patchify(latent)
    restored = patchifier.unpatchify(patchified, num_frames=4, height=16, width=16)
    assert restored.shape == latent.shape
    assert mx.allclose(restored, latent, atol=1e-5)


def test_patchify_256p():
    patchifier = VideoLatentPatchifier(patch_size=2)
    latent = mx.random.normal((1, 48, 16, 16, 16))
    result = patchifier.patchify(latent)
    assert result.shape == (1, 1024, 192)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_patchifier.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement patchifier**

```python
"""Video latent patchifier: spatial (B,C,T,H,W) <-> sequence (B,N,D)."""
import mlx.core as mx


class VideoLatentPatchifier:
    def __init__(self, patch_size: int = 2):
        self.p = patch_size

    def patchify(self, latent: mx.array) -> mx.array:
        B, C, T, H, W = latent.shape
        p = self.p
        Hp, Wp = H // p, W // p
        x = latent.reshape(B, C, T, Hp, p, Wp, p)
        x = x.transpose(0, 2, 3, 5, 1, 4, 6)
        x = x.reshape(B, T * Hp * Wp, C * p * p)
        return x

    def unpatchify(self, x: mx.array, num_frames: int, height: int, width: int) -> mx.array:
        B, N, D = x.shape
        p = self.p
        Hp, Wp = height // p, width // p
        C = D // (p * p)
        x = x.reshape(B, num_frames, Hp, Wp, C, p, p)
        x = x.transpose(0, 4, 1, 2, 5, 3, 6)
        x = x.reshape(B, C, num_frames, height, width)
        return x
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_patchifier.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_mlx/components/patchifier.py tests/test_patchifier.py
git commit -m "feat: add video latent patchifier with roundtrip tests"
```

---

### Task 4: Core Components -- Scheduler

**Files:**
- Create: `davinci_mlx/components/scheduler.py`
- Create: `tests/test_scheduler.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for flow-matching scheduler."""
import mlx.core as mx
from davinci_mlx.components.scheduler import FlowMatchingScheduler


def test_get_sigmas_8_steps():
    scheduler = FlowMatchingScheduler()
    sigmas = scheduler.get_sigmas(num_steps=8)
    assert len(sigmas) == 9
    assert float(sigmas[0]) == 1.0
    assert float(sigmas[-1]) == 0.0


def test_euler_step():
    scheduler = FlowMatchingScheduler()
    sample = mx.ones((1, 4, 4))
    denoised = mx.zeros((1, 4, 4))
    sigma = mx.array(1.0)
    sigma_next = mx.array(0.5)
    result = scheduler.step(sample, denoised, sigma, sigma_next)
    expected = mx.full((1, 4, 4), 0.5)
    assert mx.allclose(result, expected, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scheduler.py -v`

- [ ] **Step 3: Implement scheduler**

```python
"""Flow-matching scheduler for distilled inference."""
import mlx.core as mx


class FlowMatchingScheduler:
    def get_sigmas(self, num_steps: int = 8) -> mx.array:
        return mx.linspace(1.0, 0.0, num_steps + 1)

    def step(self, sample: mx.array, denoised: mx.array,
             sigma: mx.array, sigma_next: mx.array) -> mx.array:
        velocity = (sample - denoised) / sigma
        return sample + velocity * (sigma_next - sigma)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scheduler.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_mlx/components/scheduler.py tests/test_scheduler.py
git commit -m "feat: add flow-matching scheduler with Euler step"
```

---

### Task 5: RoPE and Fourier Embeddings

**Files:**
- Create: `davinci_mlx/model/transformer/rope.py`
- Create: `tests/test_rope.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for SPLIT RoPE and Fourier embeddings."""
import mlx.core as mx
from davinci_mlx.model.transformer.rope import precompute_freqs, apply_rotary_emb


def test_precompute_freqs_shape():
    cos_f, sin_f = precompute_freqs(dim=128, max_pos=256)
    assert cos_f.shape == (256, 64)
    assert sin_f.shape == (256, 64)


def test_apply_rotary_emb_shape():
    cos_f, sin_f = precompute_freqs(dim=128, max_pos=256)
    x = mx.random.normal((1, 40, 100, 128))
    positions = mx.arange(100)
    result = apply_rotary_emb(x, cos_f, sin_f, positions)
    assert result.shape == x.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rope.py -v`

- [ ] **Step 3: Implement RoPE (SPLIT) and ElementWiseFourierEmbed**

```python
"""Rotary position embeddings (SPLIT) and learnable Fourier embeddings."""
import mlx.core as mx
import mlx.nn as nn


def precompute_freqs(dim: int, max_pos: int, theta: float = 10000.0):
    half_dim = dim // 2
    freqs = 1.0 / (theta ** (mx.arange(0, half_dim).astype(mx.float32) / half_dim))
    positions = mx.arange(max_pos).astype(mx.float32)
    angles = mx.outer(positions, freqs)
    return mx.cos(angles), mx.sin(angles)


def apply_rotary_emb(x, cos_freqs, sin_freqs, positions):
    cos = cos_freqs[positions][None, None, :, :]
    sin = sin_freqs[positions][None, None, :, :]
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


class ElementWiseFourierEmbed(nn.Module):
    def __init__(self, dim: int, num_bands: int = 64):
        super().__init__()
        self.num_bands = num_bands
        self.bands = mx.zeros((num_bands,))

    def __call__(self, coords: mx.array) -> mx.array:
        B, T, C = coords.shape
        x = coords[:, :, :, None] * self.bands[None, None, None, :]
        features = mx.concatenate([mx.sin(x), mx.cos(x)], axis=-1)
        return features.reshape(B, T, C * self.num_bands * 2)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_rope.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_mlx/model/transformer/rope.py tests/test_rope.py
git commit -m "feat: add SPLIT RoPE and ElementWiseFourierEmbed"
```

---

### Task 6: Fused silu_mul Metal Kernel

**Files:**
- Create: `davinci_mlx/kernels/fused_ops.py`
- Create: `tests/test_kernels.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for fused Metal kernels."""
import mlx.core as mx
from davinci_mlx.kernels.fused_ops import silu_mul


def test_silu_mul():
    a = mx.random.normal((4, 128))
    b = mx.random.normal((4, 128))
    result = silu_mul(a, b)
    expected = mx.sigmoid(a) * a * b
    assert mx.allclose(result, expected, atol=1e-4)


def test_silu_mul_large():
    a = mx.random.normal((1024, 13652))
    b = mx.random.normal((1024, 13652))
    result = silu_mul(a, b)
    expected = mx.sigmoid(a) * a * b
    assert mx.allclose(result, expected, atol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_kernels.py -v`

- [ ] **Step 3: Implement fused silu_mul**

Reference: `~/Developer/LTX-2-MLX/LTX_2_MLX/kernels/fused_ops.py`

```python
"""Fused Metal kernels for performance-critical operations."""
import mlx.core as mx

_SILU_MUL_SOURCE = """
    uint elem = thread_position_in_grid.x;
    if (elem < a.size()) {
        T x = a[elem];
        T sigmoid_x = T(1) / (T(1) + exp(-x));
        out[elem] = x * sigmoid_x * b[elem];
    }
"""

_silu_mul_kernel = None


def _get_kernel():
    global _silu_mul_kernel
    if _silu_mul_kernel is None:
        _silu_mul_kernel = mx.fast.metal_kernel(
            name="silu_mul",
            input_names=["a", "b"],
            output_names=["out"],
            source=_SILU_MUL_SOURCE,
        )
    return _silu_mul_kernel


def silu_mul(a: mx.array, b: mx.array) -> mx.array:
    kernel = _get_kernel()
    # Ensure contiguous memory layout for Metal kernel
    a = mx.contiguous(a) if not a.flags["row_contiguous"] else a
    b = mx.contiguous(b) if not b.flags["row_contiguous"] else b
    return kernel(
        inputs=[a, b],
        template=[("T", a.dtype)],
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
        grid=(a.size, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_kernels.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_mlx/kernels/fused_ops.py tests/test_kernels.py
git commit -m "feat: add fused silu_mul Metal kernel for SwiGLU"
```

---

### Task 7: Attention Module (GQA + Gating)

**Files:**
- Create: `davinci_mlx/model/transformer/attention.py`
- Create: `tests/test_attention.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for GQA attention with gating."""
import mlx.core as mx
from davinci_mlx.model.transformer.attention import Attention


def test_attention_output_shape():
    attn = Attention(hidden_size=256, num_heads_q=4, num_heads_kv=2, head_dim=64)
    x = mx.random.normal((1, 16, 256))
    result = attn(x)
    assert result.shape == (1, 16, 256)


def test_qkvg_projection_shape():
    attn = Attention(hidden_size=5120, num_heads_q=40, num_heads_kv=8, head_dim=128)
    # Q: 5120, K: 1024, V: 1024, G: 5120 = 12288
    assert attn.linear_qkv.weight.shape == (12288, 5120)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_attention.py -v`

- [ ] **Step 3: Implement GQA attention with gating**

```python
"""GQA attention with sigmoid gating for daVinci single-stream DiT.

Fused QKV+G projection -> split -> QK norm -> RoPE -> flash attention -> gate.
"""
import math
import mlx.core as mx
import mlx.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size=5120, num_heads_q=40, num_heads_kv=8, head_dim=128):
        super().__init__()
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.gqa_ratio = num_heads_q // num_heads_kv

        q_dim = num_heads_q * head_dim
        kv_dim = num_heads_kv * head_dim
        qkvg_dim = q_dim + 2 * kv_dim + q_dim  # Q + K + V + Gate

        self.linear_qkv = nn.Linear(hidden_size, qkvg_dim, bias=True)
        self.q_norm_weight = mx.ones((head_dim,))
        self.k_norm_weight = mx.ones((head_dim,))
        self.linear_proj = nn.Linear(q_dim, hidden_size, bias=True)

    def __call__(self, x, cos_freqs=None, sin_freqs=None, positions=None, mask=None):
        B, T, _ = x.shape
        qkvg = self.linear_qkv(x)

        q_dim = self.num_heads_q * self.head_dim
        kv_dim = self.num_heads_kv * self.head_dim
        q, k, v, gate = (
            qkvg[:, :, :q_dim],
            qkvg[:, :, q_dim:q_dim + kv_dim],
            qkvg[:, :, q_dim + kv_dim:q_dim + 2 * kv_dim],
            qkvg[:, :, q_dim + 2 * kv_dim:],
        )

        q = q.reshape(B, T, self.num_heads_q, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads_kv, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads_kv, self.head_dim).transpose(0, 2, 1, 3)

        q = mx.fast.rms_norm(q, self.q_norm_weight, eps=1e-6)
        k = mx.fast.rms_norm(k, self.k_norm_weight, eps=1e-6)

        if cos_freqs is not None:
            from davinci_mlx.model.transformer.rope import apply_rotary_emb
            q = apply_rotary_emb(q, cos_freqs, sin_freqs, positions)
            k = apply_rotary_emb(k, cos_freqs, sin_freqs, positions)

        # NOTE: Do NOT repeat K/V for GQA. MLX's scaled_dot_product_attention
        # natively handles GQA when K/V have fewer heads than Q.
        attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        attn_out = self.linear_proj(attn_out)
        attn_out = attn_out * mx.sigmoid(gate)
        return attn_out
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_attention.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_mlx/model/transformer/attention.py tests/test_attention.py
git commit -m "feat: add GQA attention with sigmoid gating and QK norm"
```

---

### Task 8: Feed-Forward (SwiGLU7 + GELU7)

**Files:**
- Create: `davinci_mlx/model/transformer/feed_forward.py`
- Create: `tests/test_feed_forward.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for SwiGLU7 and GELU7 feed-forward networks."""
import mlx.core as mx
from davinci_mlx.model.transformer.feed_forward import SwiGLU7FFN, GELU7FFN


def test_swiglu7_shape():
    ffn = SwiGLU7FFN(hidden_size=5120, intermediate_size=13652)
    x = mx.random.normal((1, 64, 5120))
    assert ffn(x).shape == (1, 64, 5120)


def test_gelu7_shape():
    ffn = GELU7FFN(hidden_size=5120, intermediate_size=20480)
    x = mx.random.normal((1, 64, 5120))
    assert ffn(x).shape == (1, 64, 5120)


def test_swiglu7_clamping():
    ffn = SwiGLU7FFN(hidden_size=32, intermediate_size=64)
    x = mx.ones((1, 4, 32)) * 100.0
    result = ffn(x)
    assert mx.all(result >= -7.0) and mx.all(result <= 7.0)


def test_gelu7_clamping():
    ffn = GELU7FFN(hidden_size=32, intermediate_size=128)
    x = mx.ones((1, 4, 32)) * 100.0
    result = ffn(x)
    assert mx.all(result >= -7.0) and mx.all(result <= 7.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_feed_forward.py -v`

- [ ] **Step 3: Implement SwiGLU7 and GELU7**

```python
"""Feed-forward networks with clamped activations [-7, 7].

SwiGLU7: gated (3 matrices, layers 4-39). GELU7: non-gated (2 matrices, layers 0-3).
"""
import mlx.core as mx
import mlx.nn as nn
from davinci_mlx.kernels.fused_ops import silu_mul


class SwiGLU7FFN(nn.Module):
    def __init__(self, hidden_size=5120, intermediate_size=13652):
        super().__init__()
        self.up_gate_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

    def __call__(self, x):
        gate_up = self.up_gate_proj(x)
        gate, up = mx.split(gate_up, 2, axis=-1)
        hidden = silu_mul(gate, up)
        hidden = mx.clip(hidden, -7.0, 7.0)
        return self.down_proj(hidden)


class GELU7FFN(nn.Module):
    def __init__(self, hidden_size=5120, intermediate_size=20480):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

    def __call__(self, x):
        hidden = nn.gelu_approx(self.up_proj(x))
        hidden = mx.clip(hidden, -7.0, 7.0)
        return self.down_proj(hidden)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_feed_forward.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_mlx/model/transformer/feed_forward.py tests/test_feed_forward.py
git commit -m "feat: add SwiGLU7 and GELU7 FFN with [-7,7] clamping"
```

---

### Task 9: Transformer Block

**Files:**
- Create: `davinci_mlx/model/transformer/transformer.py`
- Create: `tests/test_transformer_block.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for transformer block."""
import mlx.core as mx
from davinci_mlx.model.transformer.transformer import TransformerBlock
from davinci_mlx.model.transformer.feed_forward import GELU7FFN


def test_block_shape():
    block = TransformerBlock(hidden_size=256, num_heads_q=4, num_heads_kv=2,
                            head_dim=64, layer_idx=4)
    x = mx.random.normal((1, 64, 256))
    assert block(x).shape == (1, 64, 256)


def test_gelu_layer_uses_gelu():
    block = TransformerBlock(hidden_size=256, num_heads_q=4, num_heads_kv=2,
                            head_dim=64, layer_idx=0)
    assert isinstance(block.mlp.ffn, GELU7FFN)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_transformer_block.py -v`

- [ ] **Step 3: Implement transformer block**

```python
"""Transformer block: RMSNorm -> Attention -> residual -> RMSNorm -> FFN -> residual.

No AdaLN. daVinci is timestep-free.
"""
import mlx.core as mx
import mlx.nn as nn
from davinci_mlx.model.transformer.attention import Attention
from davinci_mlx.model.transformer.feed_forward import SwiGLU7FFN, GELU7FFN

GELU_LAYERS = {0, 1, 2, 3}


class MultiModalityRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class AttentionWithNorm(nn.Module):
    def __init__(self, hidden_size, num_heads_q, num_heads_kv, head_dim):
        super().__init__()
        self.pre_norm = MultiModalityRMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads_q, num_heads_kv, head_dim)

    def __call__(self, x, cos_freqs=None, sin_freqs=None, positions=None, mask=None):
        return self.attn(self.pre_norm(x), cos_freqs, sin_freqs, positions, mask)


class MLPWithNorm(nn.Module):
    def __init__(self, hidden_size, layer_idx,
                 swiglu_intermediate=13652, gelu_intermediate=20480):
        super().__init__()
        self.pre_norm = MultiModalityRMSNorm(hidden_size)
        if layer_idx in GELU_LAYERS:
            self.ffn = GELU7FFN(hidden_size, gelu_intermediate)
        else:
            self.ffn = SwiGLU7FFN(hidden_size, swiglu_intermediate)

    def __call__(self, x):
        return self.ffn(self.pre_norm(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=5120, num_heads_q=40, num_heads_kv=8,
                 head_dim=128, layer_idx=0):
        super().__init__()
        self.attention = AttentionWithNorm(hidden_size, num_heads_q, num_heads_kv, head_dim)
        self.mlp = MLPWithNorm(hidden_size, layer_idx)

    def __call__(self, x, cos_freqs=None, sin_freqs=None, positions=None, mask=None):
        x = x + self.attention(x, cos_freqs, sin_freqs, positions, mask)
        x = x + self.mlp(x)
        return x
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_transformer_block.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_mlx/model/transformer/transformer.py tests/test_transformer_block.py
git commit -m "feat: add transformer block with pre-norm, GQA, hybrid FFN"
```

---

### Task 10: DaVinci Model (Full 40-Layer)

**Files:**
- Create: `davinci_mlx/model/transformer/model.py`
- Create: `tests/test_davinci_model.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for full DaVinci DiT model."""
import mlx.core as mx
from davinci_mlx.model.transformer.model import DaVinciModel


def test_model_output_shape():
    model = DaVinciModel(hidden_size=256, num_layers=4, num_heads_q=4,
                         num_heads_kv=2, head_dim=64, video_in_channels=192,
                         text_in_channels=3584)
    video_tokens = mx.random.normal((1, 8, 192))
    text_tokens = mx.random.normal((1, 4, 3584))
    result = model(video_tokens, text_tokens)
    assert result.shape == (1, 8, 192)


def test_model_40_layers():
    model = DaVinciModel()
    assert len(model.blocks) == 40
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_davinci_model.py -v`

- [ ] **Step 3: Implement DaVinci model**

```python
"""DaVinci DiT: 15B single-stream transformer.

Sandwich: adapter -> 40 blocks -> final projection.
Video + text concatenated, self-attention only, timestep-free.
"""
import mlx.core as mx
import mlx.nn as nn
from davinci_mlx.model.transformer.transformer import TransformerBlock, MultiModalityRMSNorm
from davinci_mlx.model.transformer.rope import ElementWiseFourierEmbed, precompute_freqs


class DaVinciModel(nn.Module):
    def __init__(self, hidden_size=5120, num_layers=40, num_heads_q=40,
                 num_heads_kv=8, head_dim=128, video_in_channels=192,
                 text_in_channels=3584, rope_num_bands=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.video_in_channels = video_in_channels

        self.video_embedder = nn.Linear(video_in_channels, hidden_size, bias=True)
        self.text_embedder = nn.Linear(text_in_channels, hidden_size, bias=True)
        self.rope = ElementWiseFourierEmbed(dim=head_dim, num_bands=rope_num_bands)

        self.blocks = [
            TransformerBlock(hidden_size, num_heads_q, num_heads_kv, head_dim, i)
            for i in range(num_layers)
        ]

        self.final_norm_video = MultiModalityRMSNorm(hidden_size)
        self.final_linear_video = nn.Linear(hidden_size, video_in_channels, bias=True)

    def __call__(self, video_tokens, text_tokens, coords_mapping=None):
        B = video_tokens.shape[0]
        num_video = video_tokens.shape[1]

        video_hidden = self.video_embedder(video_tokens)
        text_hidden = self.text_embedder(text_tokens)
        x = mx.concatenate([video_hidden, text_hidden], axis=1)

        total_len = x.shape[1]
        cos_freqs, sin_freqs = precompute_freqs(
            dim=self.blocks[0].attention.attn.head_dim, max_pos=total_len)
        positions = mx.arange(total_len)

        for block in self.blocks:
            x = block(x, cos_freqs, sin_freqs, positions)

        video_out = x[:, :num_video, :]
        video_out = self.final_norm_video(video_out)
        return self.final_linear_video(video_out)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_davinci_model.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_mlx/model/transformer/model.py tests/test_davinci_model.py
git commit -m "feat: add DaVinci 40-layer single-stream DiT model"
```

---

### Task 11: Weight Converter

**Files:**
- Create: `davinci_mlx/loader/weight_converter.py`
- Create: `tests/test_weight_conversion.py`

**Prerequisite:** Weights downloaded (Task 2 Step 4).

- [ ] **Step 1: Explore actual weight key names from safetensors**

```bash
python -c "
from safetensors import safe_open
import glob, json
index_path = glob.glob('weights/original/distill/*index*.json')
if index_path:
    with open(index_path[0]) as f:
        index = json.load(f)
    for k in sorted(index['weight_map'].keys())[:50]:
        print(k)
    print(f'Total: {len(index[\"weight_map\"])} keys')
"
```

This reveals exact key names. Update `convert_key()` mapping accordingly.

- [ ] **Step 2: Write failing test**

```python
"""Tests for weight converter."""
from davinci_mlx.loader.weight_converter import convert_key


def test_convert_adapter_keys():
    assert convert_key("adapter.video_embedder.weight") == "video_embedder.weight"
    assert convert_key("adapter.text_embedder.weight") == "text_embedder.weight"


def test_convert_block_keys():
    assert convert_key("block.layers.4.attention.linear_qkv.weight") is not None
    assert convert_key("block.layers.4.mlp.up_gate_proj.weight") is not None


def test_convert_final_keys():
    assert convert_key("final_norm_video.weight") == "final_norm_video.weight"
    assert convert_key("final_linear_video.weight") == "final_linear_video.weight"


def test_skip_audio_keys():
    assert convert_key("block.layers.0.attention.linear_qkv.audio_weight") is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_weight_conversion.py -v`

- [ ] **Step 4: Implement weight converter**

Key mapping must be refined based on Step 1 findings. Initial structure:

```python
"""Convert PyTorch daVinci weights to MLX format."""
import gc
import json
from pathlib import Path
from typing import Optional
import mlx.core as mx
from safetensors import safe_open


def convert_key(pytorch_key: str) -> Optional[str]:
    if "audio" in pytorch_key.lower():
        return None
    key = pytorch_key
    if key.startswith("adapter."):
        return key[len("adapter."):]
    if key.startswith("block.layers."):
        key = key.replace("block.layers.", "blocks.")
        # Map attention subkeys to our nesting
        key = key.replace(".attention.linear_qkv.", ".attention.attn.linear_qkv.")
        key = key.replace(".attention.linear_proj.", ".attention.attn.linear_proj.")
        key = key.replace(".attention.q_norm.weight", ".attention.attn.q_norm_weight")
        key = key.replace(".attention.k_norm.weight", ".attention.attn.k_norm_weight")
        key = key.replace(".mlp.up_gate_proj.", ".mlp.ffn.up_gate_proj.")
        key = key.replace(".mlp.down_proj.", ".mlp.ffn.down_proj.")
        return key
    if key.startswith("final_"):
        return key
    return None


def load_weights(model, weights_dir, target_dtype=mx.float16):
    weights_dir = Path(weights_dir)
    index_files = list(weights_dir.glob("*.index.json"))
    if index_files:
        with open(index_files[0]) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = [f.name for f in sorted(weights_dir.glob("*.safetensors"))]

    converted = {}
    count = 0
    for shard_file in shard_files:
        shard_path = weights_dir / shard_file
        print(f"Loading {shard_path.name}...")
        with safe_open(str(shard_path), framework="numpy") as sf:
            for pytorch_key in sf.keys():
                mlx_key = convert_key(pytorch_key)
                if mlx_key is None:
                    continue
                tensor = sf.get_tensor(pytorch_key)
                arr = mx.array(tensor).astype(target_dtype)
                _set_nested(converted, mlx_key, arr)
                count += 1
                if count % 100 == 0:
                    gc.collect()
    model.update(converted)
    print(f"Loaded {count} weight tensors.")


def _set_nested(d, key, value):
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in d:
            d[part] = {}
        d = d[part]
    d[parts[-1]] = value
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_weight_conversion.py -v`
Expected: 4 tests PASS.

- [ ] **Step 6: Integration test -- load real weights**

```bash
python -c "
from davinci_mlx.model.transformer.model import DaVinciModel
from davinci_mlx.loader.weight_converter import load_weights
model = DaVinciModel()
load_weights(model, 'weights/original/distill')
print('Success')
"
```

Iterate on key mapping until all weights load without errors.

- [ ] **Step 7: Commit**

```bash
git add davinci_mlx/loader/weight_converter.py tests/test_weight_conversion.py
git commit -m "feat: add weight converter for PyTorch safetensors -> MLX"
```

---

### Task 12: Turbo VAE Decoder

**Files:**
- Create: `davinci_mlx/model/turbo_vae/conv3d.py`
- Create: `davinci_mlx/model/turbo_vae/decoder.py`
- Create: `tests/test_turbo_vae.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for Turbo VAE decoder."""
import mlx.core as mx
from davinci_mlx.model.turbo_vae.decoder import TurboVAEDecoder


def test_decoder_output_ndim():
    decoder = TurboVAEDecoder(latent_channels=48, out_channels=3)
    latent = mx.random.normal((1, 48, 2, 4, 4))
    result = decoder(latent)
    assert result.ndim == 5
    assert result.shape[1] == 3  # RGB
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_turbo_vae.py -v`

- [ ] **Step 3: Implement CausalConv3d**

Port `TurboVAEDCausalConv3d` from PyTorch using 2D conv reshape approach.
Reference: daVinci `turbo_vaed_module.py` and LTX-2-MLX `convolution.py`.

- [ ] **Step 4: Implement TurboVAEDecoder**

Port the decoder architecture: mid blocks -> up blocks -> final conv.
This is a large module (~400 lines). Key components:
- `TurboVAEDResnetBlock3d`
- `TurboVAEDUpBlock3d`
- `TurboVAEDDecoder3d`
- Normalization constants (hardcoded mean/std)

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_turbo_vae.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add davinci_mlx/model/turbo_vae/ tests/test_turbo_vae.py
git commit -m "feat: add Turbo VAE decoder with causal 3D conv"
```

---

### Task 13: Text Encoder (T5-Gemma)

**Files:**
- Create: `davinci_mlx/model/text_encoder/t5_gemma.py`
- Create: `davinci_mlx/model/text_encoder/encoder.py`

- [ ] **Step 1: Investigate if mlx-lm or existing MLX implementation works**

```bash
pip install mlx-lm
python -c "from mlx_lm import load; help(load)"
```

Check if T5-Gemma-9B can be loaded. If not, implement natively using LTX-2-MLX's Gemma 3 as reference.

- [ ] **Step 2: Implement encoder wrapper**

```python
"""T5-Gemma text encoder: produces (B, seq_len, 3584) embeddings."""
from pathlib import Path
import gc
import mlx.core as mx


class TextEncoder:
    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir)
        self.model = None
        self.tokenizer = None

    def load(self):
        # Load via mlx-lm or native implementation
        raise NotImplementedError("Implement after Step 1")

    def unload(self):
        self.model = None
        self.tokenizer = None
        gc.collect()

    def encode(self, prompt, max_length=512):
        if self.model is None:
            self.load()
        # Tokenize and forward
        raise NotImplementedError("Implement after Step 1")
```

- [ ] **Step 3: Commit**

```bash
git add davinci_mlx/model/text_encoder/
git commit -m "feat: add T5-Gemma text encoder wrapper (implementation TBD)"
```

---

### Task 14: Distilled Pipeline + CLI

**Files:**
- Create: `davinci_mlx/pipeline/common.py`
- Create: `davinci_mlx/pipeline/distilled.py`
- Create: `scripts/generate.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test for common utilities**

```python
from davinci_mlx.pipeline.common import validate_dimensions, compute_latent_shape

def test_validate_256p():
    validate_dimensions(256, 256, 65)

def test_validate_bad_resolution():
    import pytest
    with pytest.raises(ValueError):
        validate_dimensions(100, 100, 65)

def test_compute_latent_shape():
    shape = compute_latent_shape(256, 256, 65)
    assert shape == (48, 17, 16, 16)
```

- [ ] **Step 2: Implement common.py and distilled.py**

Pipeline orchestrates: text encode -> denoise -> VAE decode with sequential loading.

- [ ] **Step 3: Implement generate.py CLI**

```python
#!/usr/bin/env python3
"""Generate video from text prompt."""
import argparse, time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--frames", type=int, default=65)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", default="float16")
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--weights-dir", default="weights/fp16")
    args = parser.parse_args()

    from davinci_mlx.pipeline.distilled import DistilledPipeline
    pipe = DistilledPipeline(args.weights_dir, args.precision)
    start = time.time()
    video = pipe.generate(args.prompt, args.height, args.width,
                          args.frames, args.steps, args.seed)
    pipe.save_video(video, args.output)
    print(f"Saved to {args.output} ({time.time()-start:.1f}s)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_pipeline.py -v`

- [ ] **Step 5: Commit**

```bash
git add davinci_mlx/pipeline/ scripts/generate.py tests/test_pipeline.py
git commit -m "feat: add distilled pipeline and CLI generate script"
```

---

### Task 15: Quantization Script

**Files:**
- Create: `davinci_mlx/loader/quantize.py`
- Create: `scripts/quantize_weights.py`

- [ ] **Step 1: Implement quantization module and CLI**

Uses `mlx.nn.quantize()` to convert Linear -> QuantizedLinear (INT4).

- [ ] **Step 2: Commit**

```bash
git add davinci_mlx/loader/quantize.py scripts/quantize_weights.py
git commit -m "feat: add INT4 quantization script"
```

---

### Task 16: SSD Manifest Update

- [ ] **Step 1: Add entry id 22 to `/Volumes/Untitled/ai-models/manifest.json`**
- [ ] **Step 2: Update `~/Developer/.ai-models-manifest.md`**
- [ ] **Step 3: Commit**

---

### Task 17: End-to-End Integration Test

**Prerequisite:** All weights downloaded and converted.

- [ ] **Step 1: Generate 256p test video**

```bash
python scripts/generate.py --prompt "A golden retriever running through a meadow" \
    --height 256 --width 256 --frames 65 --steps 8 --output test_256p.mp4
```

- [ ] **Step 2: Verify output is valid MP4**
- [ ] **Step 3: Record timing for each pipeline stage**
- [ ] **Step 4: Test INT4 quantization end-to-end**
- [ ] **Step 5: Commit any fixes**
