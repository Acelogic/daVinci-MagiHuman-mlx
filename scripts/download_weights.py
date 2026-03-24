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
