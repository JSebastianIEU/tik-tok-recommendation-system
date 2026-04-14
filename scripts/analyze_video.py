#!/usr/bin/env python3
"""Standalone video analyzer script — runs in its own process to avoid OOM.

Usage: python scripts/analyze_video.py <input_video> <output_json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root is on path
repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: analyze_video.py <input_video> <output_json>", file=sys.stderr)
        return 1

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    from src.recommendation.video.analyzer import VideoAnalyzer

    va = VideoAnalyzer(max_workers=4)
    result = va.analyze(input_path)
    data = result.model_dump(mode="python")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
