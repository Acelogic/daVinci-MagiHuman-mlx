"""Tests for quantization utilities."""
import mlx.core as mx
import mlx.nn as nn
from davinci_mlx.loader.quantize import quantize_model, save_quantized_weights


def test_quantize_reduces_size():
    """Quantized model should use less memory than FP32/FP16."""

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(256, 128)

    model = TinyModel()
    original_size = model.linear.weight.nbytes  # FP32: 256*128*4 = 131072 bytes

    quantize_model(model, bits=4, group_size=64)

    # Linear should have been replaced with QuantizedLinear
    assert isinstance(model.linear, nn.QuantizedLinear), (
        f"Expected QuantizedLinear, got {type(model.linear)}"
    )

    # INT4 weight tensor is packed into uint32; shape shrinks from (128,256) to (128,32)
    quantized_weight = model.linear.weight
    assert quantized_weight.dtype == mx.uint32, (
        f"Expected uint32 packed weight, got {quantized_weight.dtype}"
    )
    assert quantized_weight.nbytes < original_size, (
        f"Quantized size {quantized_weight.nbytes} not smaller than original {original_size}"
    )


def test_quantize_int8():
    """INT8 quantization should also produce a QuantizedLinear."""

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(128, 64)

    model = TinyModel()
    quantize_model(model, bits=8, group_size=64)
    assert isinstance(model.fc, nn.QuantizedLinear)


def test_save_quantized_weights(tmp_path):
    """save_quantized_weights should write a safetensors file."""
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 32)

    model = TinyModel()
    quantize_model(model, bits=4, group_size=64)
    save_quantized_weights(model, tmp_path)

    out_file = tmp_path / "model.safetensors"
    assert out_file.exists(), "model.safetensors was not created"
    assert out_file.stat().st_size > 0, "model.safetensors is empty"
