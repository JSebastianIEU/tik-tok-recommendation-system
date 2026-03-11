from __future__ import annotations

from collections import Counter
from typing import Any


def _null_rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    nulls = sum(1 for row in rows if row.get(key) is None)
    return nulls / len(rows)


def _duplicate_rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    values = [row.get(key) for row in rows]
    counts = Counter(values)
    dupes = sum(count - 1 for count in counts.values() if count > 1)
    return dupes / len(rows)


def build_quality_scorecard(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if dataset == "videos":
        primary_key = "video_id"
    elif dataset == "comments":
        primary_key = "comment_id"
    elif dataset == "authors":
        primary_key = "author_id"
    else:
        primary_key = "video_id"

    keys = set()
    for row in rows[:100]:
        keys.update(row.keys())

    null_rates = {k: _null_rate(rows, k) for k in sorted(keys)}

    return {
        "dataset": dataset,
        "rows": len(rows),
        "primary_key": primary_key,
        "duplicate_rate": _duplicate_rate(rows, primary_key),
        "null_rates": null_rates,
    }
