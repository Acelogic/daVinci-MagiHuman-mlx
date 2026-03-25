#!/usr/bin/env python3
"""Generate video from text prompt using daVinci-MagiHuman MLX."""
import argparse
import time


def main():
    parser = argparse.ArgumentParser(description="daVinci-MagiHuman MLX video generation")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--frames", type=int, default=65)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["float16", "int4"], default="float16")
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--weights-dir", type=str, default="weights/original")
    parser.add_argument("--sr", action="store_true", help="Apply 540p super-resolution")
    args = parser.parse_args()

    from davinci_mlx.pipeline.distilled import DistilledPipeline

    pipe = DistilledPipeline(weights_dir=args.weights_dir, precision=args.precision)

    print(f"Generating {args.height}x{args.width} video, {args.frames} frames, {args.steps} steps")
    start = time.time()

    video = pipe.generate(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        steps=args.steps,
        seed=args.seed,
    )

    pipe.save_video(video, args.output)
    elapsed = time.time() - start
    print(f"Saved to {args.output} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
