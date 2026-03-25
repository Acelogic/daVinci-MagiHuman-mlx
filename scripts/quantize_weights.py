#!/usr/bin/env python3
"""Quantize FP16 transformer weights to INT4."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Quantize weights to INT4")
    parser.add_argument("--input", type=Path, required=True, help="FP16 weights dir (safetensors)")
    parser.add_argument("--output", type=Path, required=True, help="INT4 output dir")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    import mlx.core as mx
    from davinci_mlx.model.transformer.model import DaVinciModel
    from davinci_mlx.loader.weight_converter import convert_and_load
    from davinci_mlx.loader.quantize import quantize_model, save_quantized_weights

    print(f"Quantizing transformer to {args.bits}-bit...")
    model = DaVinciModel()
    convert_and_load(model, args.input, target_dtype=mx.float16)

    print(f"Quantizing with {args.bits}-bit, group_size={args.group_size}...")
    quantize_model(model, bits=args.bits, group_size=args.group_size)

    save_quantized_weights(model, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
