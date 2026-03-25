"""Quantize FP16 model weights to INT4 using MLX."""
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def quantize_model(model: nn.Module, bits: int = 4, group_size: int = 64):
    """Quantize all Linear layers in the model to INT4/INT8.

    Uses mlx.nn.quantize to replace nn.Linear with nn.QuantizedLinear.
    Modifies the model in-place.

    Args:
        model: MLX Module whose Linear layers will be quantized.
        bits: Bit width for quantized weights (4 or 8).
        group_size: Number of weights per quantization group (default 64).
    """
    nn.quantize(model, bits=bits, group_size=group_size)


def save_quantized_weights(model: nn.Module, output_dir: Path):
    """Save quantized model weights as a single safetensors file.

    Flattens all model parameters (including QuantizedLinear's weight,
    scales, and biases arrays) into a flat dict and writes them out.

    Args:
        model: Quantized MLX Module.
        output_dir: Directory where ``model.safetensors`` will be written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(output_dir / "model.safetensors"), weights)
    print(f"Saved quantized weights to {output_dir}")
