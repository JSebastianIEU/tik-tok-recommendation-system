#!/usr/bin/env python3
"""Local CPU smoke benchmark for recommender runtime latency and memory."""

from __future__ import annotations

import argparse
import json
try:
    import resource
except ImportError:
    resource = None  # Windows compatibility
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from src.recommendation.learning import RecommenderRuntime


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    weight = index - lower
    return float((1.0 - weight) * ordered[lower] + weight * ordered[upper])


def _maxrss_mb() -> float:
    if resource is None:
        # Windows: use psutil if available, otherwise return 0
        try:
            import psutil
            return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
        except ImportError:
            return 0.0
    # On macOS ru_maxrss is bytes, on Linux it's KB.
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if value > 10_000_000:
        return round(value / (1024 * 1024), 2)
    return round(value / 1024.0, 2)


def _query_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query_id": row.get("row_id"),
        "text": f"{row.get('topic_key', 'general')} {row.get('row_id')}",
        "topic_key": row.get("topic_key"),
        "author_id": row.get("author_id"),
        "as_of_time": row.get("as_of_time"),
    }


def _candidate_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": row.get("row_id"),
        "text": f"{row.get('topic_key', 'general')} {row.get('row_id')}",
        "topic_key": row.get("topic_key"),
        "author_id": row.get("author_id"),
        "as_of_time": row.get("as_of_time"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark recommender runtime latency.")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="Artifact bundle directory (or latest symlink).",
    )
    parser.add_argument(
        "--datamart-json",
        type=Path,
        required=True,
        help="Datamart JSON used to construct query/candidate benchmark workload.",
    )
    parser.add_argument("--objective", type=str, default="engagement")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--retrieve-k", type=int, default=200)
    args = parser.parse_args()

    runtime = RecommenderRuntime(bundle_dir=args.bundle_dir)
    datamart = json.loads(args.datamart_json.read_text(encoding="utf-8"))
    rows = list(datamart.get("rows") or [])
    if not rows:
        raise ValueError("Datamart has no rows for benchmark workload.")

    test_rows = [row for row in rows if str(row.get("split")) in {"validation", "test"}]
    workload = test_rows or rows

    latencies_ms: List[float] = []
    item_counts: List[int] = []
    runs = max(1, int(args.runs))
    for idx in range(runs):
        query_row = workload[idx % len(workload)]
        query_as_of = query_row.get("as_of_time")
        candidates = [
            _candidate_payload(row)
            for row in rows
            if row.get("row_id") != query_row.get("row_id")
            and row.get("as_of_time")
            and query_as_of
            and row.get("as_of_time") < query_as_of
        ]
        if not candidates:
            continue

        start = time.perf_counter()
        response = runtime.recommend(
            objective=args.objective,
            as_of_time=query_as_of,
            query=_query_payload(query_row),
            candidates=candidates,
            top_k=max(1, int(args.top_k)),
            retrieve_k=max(1, int(args.retrieve_k)),
            debug=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)
        item_counts.append(len(response.get("items") or []))

    result = {
        "bundle_dir": str(args.bundle_dir),
        "objective": args.objective,
        "runs_requested": runs,
        "runs_executed": len(latencies_ms),
        "top_k": max(1, int(args.top_k)),
        "retrieve_k": max(1, int(args.retrieve_k)),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies_ms), 3) if latencies_ms else 0.0,
            "p50": round(_percentile(latencies_ms, 0.50), 3),
            "p95": round(_percentile(latencies_ms, 0.95), 3),
        },
        "memory_mb": {"maxrss": _maxrss_mb()},
        "avg_items_returned": (
            round(statistics.fmean(item_counts), 3) if item_counts else 0.0
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
