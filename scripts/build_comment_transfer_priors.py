#!/usr/bin/env python3
"""Build comment transfer priors from comment-intelligence snapshot manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.recommendation import build_comment_transfer_priors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build comment transfer priors from snapshot manifest."
    )
    parser.add_argument(
        "snapshot_manifest",
        type=Path,
        help="Path to comment snapshot manifest.json (or folder containing it).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/comment_intelligence/priors"),
    )
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--shrinkage-alpha", type=float, default=8.0)
    args = parser.parse_args()

    payload = build_comment_transfer_priors(
        snapshot_manifest=args.snapshot_manifest,
        output_root=args.output_root,
        min_support=max(1, int(args.min_support)),
        shrinkage_alpha=max(0.0, float(args.shrinkage_alpha)),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
