#!/usr/bin/env python3
"""Bootstrap an EDA extraction run from a YAML plan.

Example:
  python3 scripts/eda_bootstrap.py --plan eda/configs/plan.example.yaml
  python3 scripts/eda_bootstrap.py --plan eda/configs/plan.example.yaml --profile dev --build-silver
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eda.pipeline import run_plan  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run config-driven EDA data extraction.")
    parser.add_argument("--plan", default="eda/configs/plan.example.yaml", help="Path to EDA plan YAML.")
    parser.add_argument("--output-root", default="eda/extracts/bronze", help="Base output directory.")
    parser.add_argument("--db-url", default=None, help="Optional DB URL override.")
    parser.add_argument("--profile", default=None, help="Optional EDA profile name from eda/config/profiles.yaml.")
    parser.add_argument("--profiles-path", default="eda/config/profiles.yaml", help="Path to EDA profiles YAML.")
    parser.add_argument("--registry-path", default="eda/metadata/runs.jsonl", help="Path to append run registry JSONL.")
    parser.add_argument("--build-silver", action="store_true", help="Also build curated Silver outputs.")
    parser.add_argument(
        "--silver-output-root",
        default="eda/extracts/silver",
        help="Base output directory for Silver artifacts.",
    )
    parser.add_argument("--run-id", default=None, help="Optional explicit run id.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = run_plan(
        plan_path=args.plan,
        output_root=args.output_root,
        db_url=args.db_url,
        run_id=args.run_id,
        profile=args.profile,
        profiles_path=args.profiles_path,
        registry_path=args.registry_path,
        build_silver=args.build_silver,
        silver_output_root=args.silver_output_root,
    )
    print(f"EDA extraction complete: run_id={manifest['run_id']} output_dir={manifest['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
