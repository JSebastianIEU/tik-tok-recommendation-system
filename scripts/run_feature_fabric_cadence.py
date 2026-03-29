#!/usr/bin/env python3
"""Run feature fabric extraction cadence jobs (hourly incremental / daily full)."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cadence jobs for fabric feature extraction.")
    parser.add_argument("input_jsonl", type=Path, help="Raw JSONL path.")
    parser.add_argument(
        "--mode",
        choices=["hourly_incremental", "daily_full"],
        default="hourly_incremental",
    )
    parser.add_argument(
        "--contracts-root",
        type=Path,
        default=Path("artifacts/contracts"),
    )
    parser.add_argument(
        "--features-root",
        type=Path,
        default=Path("artifacts/features"),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
    )
    args = parser.parse_args()

    cadence_mode = "incremental" if args.mode == "hourly_incremental" else "full"
    _run(
        [
            "python3",
            "scripts/build_feature_fabric_snapshot.py",
            str(args.input_jsonl),
            "--as-of-time",
            _utc_now_iso(),
            "--mode",
            cadence_mode,
            "--contract-manifest-root",
            str(args.contracts_root),
            "--output-root",
            str(args.features_root),
        ],
        cwd=args.repo_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
