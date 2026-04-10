#!/usr/bin/env python3
"""Run comment-intelligence cadence jobs (hourly incremental / daily full)."""

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


def _latest_manifest_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"No feature manifest root found: {root}")
    manifest_dirs = [item for item in root.iterdir() if item.is_dir() and (item / "manifest.json").exists()]
    if not manifest_dirs:
        raise FileNotFoundError(f"No manifests found in: {root}")
    manifest_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return manifest_dirs[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run cadence jobs for comment-intelligence extraction."
    )
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
        default=Path("artifacts/comment_intelligence/features"),
    )
    parser.add_argument(
        "--priors-root",
        type=Path,
        default=Path("artifacts/comment_intelligence/priors"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    cadence_mode = "incremental" if args.mode == "hourly_incremental" else "full"
    _run(
        [
            "python3",
            "scripts/build_comment_intelligence_snapshot.py",
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
    latest_manifest = _latest_manifest_dir(args.features_root)
    _run(
        [
            "python3",
            "scripts/build_comment_transfer_priors.py",
            str(latest_manifest / "manifest.json"),
            "--output-root",
            str(args.priors_root),
        ],
        cwd=args.repo_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
