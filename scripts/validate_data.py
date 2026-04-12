#!/usr/bin/env python3
"""CLI to validate JSONL data files against the mock TikTok schema and rules.

Usage:
    python scripts/validate_data.py data/mock/tiktok_posts_mock.jsonl
    python scripts/validate_data.py data/mock/tiktok_posts_mock.jsonl --max-failures 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.common.validation import (
    format_failure_details,
    format_validation_summary,
    validate_jsonl_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate TikTok mock JSONL data (schema + business rules per line)."
    )
    parser.add_argument("path", help="Path to JSONL file to validate")
    parser.add_argument(
        "--max-failures",
        type=int,
        default=None,
        metavar="N",
        help="Print at most N failing lines in full (default: all).",
    )
    args = parser.parse_args(argv)

    jsonl_path = Path(args.path)
    if not jsonl_path.is_file():
        print(f"Error: file not found: {jsonl_path}", file=sys.stderr)
        return 1

    total, failed, failures = validate_jsonl_file(jsonl_path)
    print(format_validation_summary(total, failed))

    if failed:
        print("Failure details:")
        print(
            format_failure_details(
                failures,
                max_failures=args.max_failures,
            )
        )
        return 2

    print("All records valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
