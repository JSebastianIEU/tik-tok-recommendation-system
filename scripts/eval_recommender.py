#!/usr/bin/env python3
"""Evaluate trained recommender artifacts against datamart.v1 JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    parser.add_argument(
        "--show-ablation",
        action="store_true",
        help="Include retriever ablation diagnostics when available.",
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
    if args.show_ablation:
        ablation: dict[str, dict[str, object]] = {}
        reports = manifest.get("objective_ablation_reports")
        if isinstance(reports, dict):
            for objective, meta in reports.items():
                if not isinstance(meta, dict):
                    continue
                rel_path = str(meta.get("path") or "").strip()
                if not rel_path:
                    continue
                report_path = bundle_dir / rel_path
                if not report_path.exists():
                    continue
                try:
                    payload = json.loads(report_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    ablation[str(objective)] = payload
        if not ablation and isinstance(metrics, dict):
            for objective, payload in metrics.items():
                if not isinstance(payload, dict):
                    continue
                summary = payload.get("retriever_ablation")
                if isinstance(summary, dict):
                    ablation[str(objective)] = {
                        "objective": str(objective),
                        "retriever_ablation": summary,
                    }
        output["ablation"] = ablation
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
