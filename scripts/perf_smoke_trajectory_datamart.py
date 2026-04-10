#!/usr/bin/env python3
"""Performance smoke benchmark for trajectory-enabled datamart builds."""

from __future__ import annotations

import argparse
import json
import os
import time
import tempfile
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import List

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from src.recommendation.contracts import CanonicalDatasetBundle
from src.recommendation.datamart import (
    BuildTrainingDataMartConfig,
    build_training_data_mart,
)


def _to_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _synthetic_bundle(video_count: int) -> CanonicalDatasetBundle:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    generated_at = base + timedelta(hours=(6 * (video_count + 32)))
    authors = [{"author_id": "author-main", "followers_count": 50_000}]
    videos = []
    snapshots = []
    for idx in range(video_count):
        video_id = f"vid-{idx + 1}"
        posted_at = base + timedelta(hours=6 * idx)
        videos.append(
            {
                "video_id": video_id,
                "author_id": "author-main",
                "caption": f"synthetic benchmark video {idx + 1}",
                "hashtags": ["#benchmark", f"#topic{idx % 20}"],
                "keywords": ["benchmark", "topic"],
                "search_query": "benchmark",
                "posted_at": _to_z(posted_at),
                "duration_seconds": 30 + (idx % 20),
                "language": "en",
            }
        )
        snapshots.append(
            {
                "video_snapshot_id": f"{video_id}-s1",
                "video_id": video_id,
                "scraped_at": _to_z(posted_at + timedelta(hours=24)),
                "event_time": _to_z(posted_at + timedelta(hours=24)),
                "ingested_at": _to_z(posted_at + timedelta(hours=24, minutes=2)),
                "source_watermark_time": _to_z(posted_at + timedelta(hours=24)),
                "views": 100 + (idx * 3),
                "likes": 10 + (idx % 30),
                "comments_count": 4 + (idx % 12),
                "shares": 1 + (idx % 8),
            }
        )
        snapshots.append(
            {
                "video_snapshot_id": f"{video_id}-s2",
                "video_id": video_id,
                "scraped_at": _to_z(posted_at + timedelta(hours=96)),
                "event_time": _to_z(posted_at + timedelta(hours=96)),
                "ingested_at": _to_z(posted_at + timedelta(hours=96, minutes=2)),
                "source_watermark_time": _to_z(posted_at + timedelta(hours=96)),
                "views": 900 + (idx * 12),
                "likes": 80 + (idx % 120),
                "comments_count": 14 + (idx % 40),
                "shares": 7 + (idx % 16),
            }
        )
    return CanonicalDatasetBundle(
        version="contract.v2",
        generated_at=generated_at,
        authors=authors,
        videos=videos,
        video_snapshots=snapshots,
        comments=[],
        comment_snapshots=[],
    )


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round(0.95 * (len(ordered) - 1)))
    return float(ordered[max(0, min(idx, len(ordered) - 1))])


def _benchmark(bundle: CanonicalDatasetBundle, enable_trajectory: bool, runs: int) -> List[float]:
    durations_ms: List[float] = []
    config = BuildTrainingDataMartConfig(
        track="pre_publication",
        min_history_hours=24,
        label_window_hours=72,
        include_pair_rows=True,
        pair_candidates_per_query=8,
        enable_trajectory_labels=enable_trajectory,
    )
    for _ in range(runs):
        t0 = time.perf_counter()
        build_training_data_mart(bundle=bundle, config=config)
        durations_ms.append((time.perf_counter() - t0) * 1000.0)
    return durations_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Trajectory datamart performance smoke benchmark.")
    parser.add_argument("--video-count", type=int, default=2000)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    video_count = max(50, int(args.video_count))
    runs = max(2, int(args.runs))
    bundle = _synthetic_bundle(video_count)

    base_ms = _benchmark(bundle=bundle, enable_trajectory=False, runs=runs)
    traj_ms = _benchmark(bundle=bundle, enable_trajectory=True, runs=runs)

    base_p50 = float(median(base_ms))
    traj_p50 = float(median(traj_ms))
    base_p95 = _p95(base_ms)
    traj_p95 = _p95(traj_ms)
    overhead_p50 = 0.0 if base_p50 == 0.0 else ((traj_p50 - base_p50) / base_p50)
    overhead_p95 = 0.0 if base_p95 == 0.0 else ((traj_p95 - base_p95) / base_p95)

    payload = {
        "video_count": video_count,
        "runs": runs,
        "baseline_ms": {
            "p50": round(base_p50, 3),
            "p95": round(base_p95, 3),
            "samples": [round(v, 3) for v in base_ms],
        },
        "trajectory_ms": {
            "p50": round(traj_p50, 3),
            "p95": round(traj_p95, 3),
            "samples": [round(v, 3) for v in traj_ms],
        },
        "overhead_ratio": {
            "p50": round(overhead_p50, 6),
            "p95": round(overhead_p95, 6),
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
