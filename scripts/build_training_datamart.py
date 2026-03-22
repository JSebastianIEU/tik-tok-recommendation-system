#!/usr/bin/env python3
"""Build a recommendation training data mart from JSONL input."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.recommendation import (
    BuildTrainingDataMartConfig,
    build_training_data_mart_from_jsonl,
)


def _parse_iso_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {key: _to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(inner) for inner in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Build datamart.v1 from raw JSONL.")
    parser.add_argument(
        "input_jsonl",
        type=Path,
        help="Path to input JSONL dataset.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/mock/training_datamart.json"),
        help="Where to write the resulting data mart JSON.",
    )
    parser.add_argument(
        "--as-of-time",
        type=str,
        default=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        help="As-of timestamp (ISO-8601, UTC recommended).",
    )
    parser.add_argument(
        "--track",
        choices=["pre_publication", "post_publication"],
        default="pre_publication",
    )
    parser.add_argument("--min-history-hours", type=int, default=24)
    parser.add_argument("--label-window-hours", type=int, default=72)
    parser.add_argument(
        "--strict-timestamps",
        action="store_true",
        help="Fail if any snapshot timestamp must fallback from source data.",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Fail if contract normalization produces warnings.",
    )
    args = parser.parse_args()

    raw_jsonl = args.input_jsonl.read_text(encoding="utf-8")
    mart = build_training_data_mart_from_jsonl(
        raw_jsonl=raw_jsonl,
        as_of_time=_parse_iso_datetime(args.as_of_time),
        source="script_build_training_datamart",
        config=BuildTrainingDataMartConfig(
            track=args.track,
            min_history_hours=args.min_history_hours,
            label_window_hours=args.label_window_hours,
        ),
        strict_timestamps=args.strict_timestamps,
        fail_on_warnings=args.fail_on_warnings,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(_to_jsonable(mart), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote datamart to {args.output_json}")
    print(f"Rows: {mart['stats']['rows_total']}, pairs: {mart['stats']['pair_rows_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
