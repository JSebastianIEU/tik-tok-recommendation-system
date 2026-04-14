#!/usr/bin/env python3
"""Download trained model artifacts from HuggingFace Hub."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download recommender artifacts from HuggingFace.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="JSebastianIEU/tiktok-repo",
        help="HuggingFace repo ID.",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=".",
        help="Local directory to download into (default: current directory).",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        return 1

    print(f"Downloading artifacts from {args.repo_id}...")
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=args.local_dir,
        allow_patterns=["artifacts/**"],
    )
    print(f"Artifacts downloaded to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
