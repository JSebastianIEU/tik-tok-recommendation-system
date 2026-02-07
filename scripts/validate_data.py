#!/usr/bin/env python3
"""CLI to validate JSONL data files against the canonical schema.

Usage: python scripts/validate_data.py data/mock/tiktok_posts_mock.jsonl
"""

import os
import sys
import argparse

# Ensure repository root is on PYTHONPATH so `src` package is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.common.validation import load_jsonl, validate_stream


def main(argv=None):
    p = argparse.ArgumentParser(description="Validate TikTok mock JSONL data")
    p.add_argument("path", help="Path to JSONL file to validate")
    args = p.parse_args(argv)

    records = load_jsonl(args.path)
    total, failed, failures = validate_stream(records)

    print(f"Validated {total} records; failures: {failed}")
    if failed:
        print("Failure details (record_index -> errors):")
        for idx, errs in failures:
            print(f" - {idx}: ")
            for e in errs:
                print(f"    - {e}")
        sys.exit(2)

    print("All records valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
