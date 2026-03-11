"""Retrieve pre-joined datasets from scraper Postgres/Supabase.

Examples:
  python scripts/data_requests.py --dataset full --limit 1000 --out /tmp/full.jsonl
  python scripts/data_requests.py --dataset comments --all --format csv --out /tmp/comments.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scraper.data_requests import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
