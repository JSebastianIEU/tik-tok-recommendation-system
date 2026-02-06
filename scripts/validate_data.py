#!/usr/bin/env python3
"""CLI to validate JSONL data files against the canonical schema.

Usage: python scripts/validate_data.py data/mock/tiktok_posts_mock.jsonl
"""

import os
import sys
import argparse
import argparse
import os
from pathlib import Path

from src.common.validation import validate_file
from src.common.constants import MOCK_DATA_PATH

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
#!/usr/bin/env python3
"""CLI to validate JSONL data files against the canonical schema.

Usage: python scripts/validate_data.py data/mock/tiktok_posts_mock.jsonl
"""

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

def main() -> int:
    jsonl_path = Path(sys.argv[1]) if len(sys.argv) > 1 else MOCK_DATA_PATH
    count, errors = validate_file(jsonl_path)
    print(f"Validated {count} records from {jsonl_path}")
    if errors:
        print("Errors:")
        for err in errors:
            print(f"- {err}")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

# TODO: add optional schema auto-fix suggestions and output as JSON.
