"""Convert PyTorch daVinci weights to MLX format.

Handles:
- Key remapping from PyTorch checkpoint layout to MLX model structure
- MoE layer video-expert extraction (layers 0-3, 36-39 have 3 experts concatenated)
- Streaming load from sharded safetensors with periodic GC
"""
import gc
import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
from safetensors import safe_open

MOE_LAYERS = set(range(4)) | set(range(36, 40))  # layers 0-3, 36-39
NUM_EXPERTS = 3  # video=0, audio=1, text=2
VIDEO_EXPERT = 0


def convert_and_load(model, weights_dir: str, target_dtype=mx.float16):
    """Load sharded safetensors, convert keys, extract video expert for MoE layers.

    Args:
        model: MLX DaVinciModel instance to update in-place.
        weights_dir: Path to directory containing safetensors shards and index.json.
        target_dtype: Target dtype for all weights (default float16).
    """
    weights_dir = Path(weights_dir)

    # Find index file
    index_files = list(weights_dir.glob("*.index.json"))
    if not index_files:
        raise FileNotFoundError(f"No index.json found in {weights_dir}")

    with open(index_files[0]) as f:
        index = json.load(f)

    shard_files = sorted(set(index["weight_map"].values()))
    converted = {}
    count = 0
    skipped = 0

    for shard_file in shard_files:
        shard_path = weights_dir / shard_file
        if not shard_path.exists():
            print(f"  Skipping missing shard: {shard_file}")
            continue

        print(f"  Loading {shard_file}...")
        with safe_open(str(shard_path), framework="numpy") as sf:
            for pt_key in sf.keys():
                mlx_key, needs_expert_slice, layer_idx = _convert_key(pt_key)
                if mlx_key is None:
                    skipped += 1
                    continue

                tensor = sf.get_tensor(pt_key)
                arr = mx.array(tensor)

                # Extract video expert for MoE layers
                if needs_expert_slice:
                    arr = _extract_video_expert(arr, pt_key, layer_idx)

                arr = arr.astype(target_dtype)
                _set_nested(converted, mlx_key, arr)
                count += 1

                if count % 100 == 0:
                    gc.collect()

    model.update(converted)
    print(f"  Loaded {count} tensors, skipped {skipped}")


def _convert_key(pt_key: str):
    """Convert PyTorch key to MLX key.

    Returns:
        Tuple of (mlx_key, needs_expert_slice, layer_idx).
        mlx_key is None if the key should be skipped.
    """
    # Skip audio-only keys
    if pt_key in ("adapter.audio_embedder.weight", "adapter.audio_embedder.bias",
                   "final_norm_audio.weight", "final_linear_audio.weight"):
        return None, False, None

    # Adapter keys: strip "adapter." prefix
    if pt_key.startswith("adapter."):
        mlx_key = pt_key[len("adapter."):]
        return mlx_key, False, None

    # Block layer keys
    if pt_key.startswith("block.layers."):
        parts = pt_key.split(".")
        layer_idx = int(parts[2])
        is_moe = layer_idx in MOE_LAYERS

        # Build MLX key: block.layers.N -> blocks.N
        mlx_key = pt_key.replace("block.layers.", "blocks.")

        # Nest attention subkeys under AttentionWithNorm.attn
        mlx_key = mlx_key.replace(".attention.linear_qkv.", ".attention.attn.linear_qkv.")
        mlx_key = mlx_key.replace(".attention.linear_proj.", ".attention.attn.linear_proj.")
        mlx_key = mlx_key.replace(".attention.q_norm.weight", ".attention.attn.q_norm_weight")
        mlx_key = mlx_key.replace(".attention.k_norm.weight", ".attention.attn.k_norm_weight")

        # Nest MLP subkeys under MLPWithNorm.ffn
        mlx_key = mlx_key.replace(".mlp.up_gate_proj.", ".mlp.ffn.up_gate_proj.")
        mlx_key = mlx_key.replace(".mlp.down_proj.", ".mlp.ffn.down_proj.")
        # GELU layers use up_proj instead of up_gate_proj
        mlx_key = mlx_key.replace(".mlp.up_proj.", ".mlp.ffn.up_proj.")

        return mlx_key, is_moe, layer_idx

    # Final projection keys (final_norm_video, final_linear_video)
    if pt_key.startswith("final_"):
        return pt_key, False, None

    return None, False, None


def _extract_video_expert(arr: mx.array, pt_key: str, layer_idx: int) -> mx.array:
    """Extract video expert (expert 0) from MoE weight.

    MoE weights concatenate experts along the output dim:
    - 2D: (num_experts * out_per_expert, in) -> slice first 1/3 for video
    - 1D: (num_experts * D,) -> slice first 1/3 for video
    """
    if arr.ndim == 1:
        expert_size = arr.shape[0] // NUM_EXPERTS
        return arr[VIDEO_EXPERT * expert_size:(VIDEO_EXPERT + 1) * expert_size]
    elif arr.ndim == 2:
        expert_size = arr.shape[0] // NUM_EXPERTS
        return arr[VIDEO_EXPERT * expert_size:(VIDEO_EXPERT + 1) * expert_size, :]
    return arr


def _set_nested(d: dict, key: str, value):
    """Set a nested dict value from a dotted key path."""
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in d:
            d[part] = {}
        d = d[part]
    d[parts[-1]] = value
