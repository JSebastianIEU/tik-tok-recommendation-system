#!/usr/bin/env python3
"""Build comment-intelligence snapshot manifests from raw JSONL."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.recommendation import (
    CommentIntelligenceConfig,
    build_comment_intelligence_snapshot_manifest,
    build_contract_manifest,
    validate_raw_dataset_jsonl_against_contract,
)


def _parse_iso_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build comment-intelligence.v2 snapshot manifests."
    )
    parser.add_argument("input_jsonl", type=Path, help="Path to source JSONL records.")
    parser.add_argument(
        "--as-of-time",
        type=str,
        default=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        help="As-of timestamp in ISO-8601.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "incremental"],
    )
    parser.add_argument(
        "--contract-manifest-root",
        type=Path,
        default=None,
        help="Optional folder to persist contract.v2 manifest first.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/comment_intelligence/features"),
    )
    parser.add_argument("--early-window-hours", type=int, default=24)
    parser.add_argument("--late-window-hours", type=int, default=96)
    parser.add_argument("--min-comments-for-stable", type=int, default=3)
    args = parser.parse_args()

    as_of = _parse_iso_datetime(args.as_of_time)
    raw = args.input_jsonl.read_text(encoding="utf-8")
    validated = validate_raw_dataset_jsonl_against_contract(
        raw_jsonl=raw,
        as_of_time=as_of,
        source="script_build_comment_intelligence_snapshot",
    )
    if not validated.ok or validated.bundle is None:
        raise RuntimeError(
            f"Raw dataset failed contract validation with {len(validated.errors)} error(s)."
        )

    bundle = validated.bundle
    if args.contract_manifest_root is not None:
        manifest_payload = build_contract_manifest(
            bundle=bundle,
            manifest_root=args.contract_manifest_root,
            source_file_hashes={"input_jsonl": str(args.input_jsonl.resolve())},
            as_of_time=as_of,
        )
        bundle = bundle.model_copy(update={"manifest_id": manifest_payload["manifest_id"]})

    payload = build_comment_intelligence_snapshot_manifest(
        bundle=bundle,
        as_of_time=as_of,
        output_root=args.output_root,
        mode=args.mode,
        config=CommentIntelligenceConfig(
            early_window_hours=max(1, int(args.early_window_hours)),
            late_window_hours=max(2, int(args.late_window_hours)),
            min_comments_for_stable=max(1, int(args.min_comments_for_stable)),
        ),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
