#!/usr/bin/env python3
"""Evaluate trained recommender artifacts against datamart.v1 JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.recommendation.learning.artifacts import ArtifactRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description="Print recommender evaluation summary.")
    parser.add_argument(
        "bundle_dir",
        type=Path,
        help="Path to trained recommender bundle.",
    )
    parser.add_argument(
        "--show-manifest",
        action="store_true",
        help="Print bundle manifest as well.",
    )
    args = parser.parse_args()

    bundle_dir = args.bundle_dir
    if bundle_dir.is_file():
        resolved = bundle_dir.read_text(encoding="utf-8").strip()
        bundle_dir = Path(resolved)

    registry = ArtifactRegistry(bundle_dir.parent)
    manifest = registry.load_manifest(bundle_dir)
    metrics_path = bundle_dir / "metrics" / "objective_metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else {}
    )

    output = {"bundle_dir": str(bundle_dir), "metrics": metrics}
    if args.show_manifest:
        output["manifest"] = manifest
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
