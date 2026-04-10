#!/usr/bin/env python3
"""Build trajectory.v2 artifacts from datamart rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.recommendation.learning.trajectory import (
    TrajectoryBuildConfig,
    build_trajectory_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build trajectory.v2 artifact from datamart JSON."
    )
    parser.add_argument("datamart_json", type=Path, help="Path to datamart JSON file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/trajectory/latest"),
        help="Output directory for trajectory artifact files.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=16,
        help="Trajectory embedding dimension.",
    )
    parser.add_argument(
        "--feature-version",
        type=str,
        default="trajectory_features.v2",
        help="Trajectory feature schema version tag.",
    )
    parser.add_argument(
        "--encoder-mode",
        type=str,
        default="feature_only",
        choices=["feature_only", "sequence_encoder_shadow"],
        help="Trajectory encoder mode.",
    )
    args = parser.parse_args()

    datamart = json.loads(args.datamart_json.read_text(encoding="utf-8"))
    rows = list(datamart.get("rows") or [])
    if not rows:
        raise ValueError("Datamart has no rows.")
    bundle = build_trajectory_bundle(
        rows=rows,
        as_of_time=datamart.get("generated_at"),
        run_cutoff_time=datamart.get("generated_at"),
        config=TrajectoryBuildConfig(
            embedding_dim=max(4, int(args.embedding_dim)),
            feature_version=str(args.feature_version),
            encoder_mode=str(args.encoder_mode),
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = bundle.save(args.output_dir)
    payload = {
        "output_dir": str(args.output_dir),
        "manifest_path": str(manifest_path),
        "trajectory_manifest_id": bundle.trajectory_manifest_id,
        "trajectory_schema_hash": bundle.trajectory_schema_hash,
        "profile_count": len(bundle.profiles),
        "embedding_dim": bundle.config.embedding_dim,
        "version": bundle.version,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
