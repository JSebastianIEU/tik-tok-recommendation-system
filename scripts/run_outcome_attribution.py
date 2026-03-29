#!/usr/bin/env python3
"""Populate rec_outcome_events from served recommendation outputs and video snapshots."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from scraper.db.client import get_connection
from src.recommendation.control_plane import build_outcome_event


def _to_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _fetch_request_heads(conn, since: datetime) -> List[Dict[str, Any]]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
SELECT
  r.request_id::text AS request_id,
  r.objective_effective,
  r.served_at,
  so.candidate_id AS video_id
FROM rec_request_events r
JOIN LATERAL (
  SELECT candidate_id, rank
  FROM rec_served_outputs so
  WHERE so.request_id = r.request_id
  ORDER BY so.rank ASC
  LIMIT 1
) so ON TRUE
WHERE r.served_at >= %s
""",
            (since,),
        )
        rows = cursor.fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "request_id": str(row[0]),
                "objective_effective": str(row[1]),
                "served_at": row[2],
                "video_id": str(row[3]),
            }
        )
    return out


def _fetch_snapshots(conn, table_name: str, video_id: str, upper_bound: datetime) -> List[Dict[str, Any]]:
    sql = f"""
SELECT
  video_id,
  scraped_at AS event_time,
  scraped_at AS ingested_at,
  plays,
  likes,
  comments_count,
  shares
FROM {table_name}
WHERE video_id = %s AND scraped_at <= %s
ORDER BY scraped_at ASC
"""
    with conn.cursor() as cursor:
        cursor.execute(sql, (video_id, upper_bound))
        rows = cursor.fetchall()
    return [
        {
            "video_id": str(row[0]),
            "event_time": row[1],
            "ingested_at": row[2],
            "plays": row[3],
            "likes": row[4],
            "comments_count": row[5],
            "shares": row[6],
        }
        for row in rows
    ]


def _upsert_outcome(conn, payload: Dict[str, Any]) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
INSERT INTO rec_outcome_events (
  request_id,
  objective_effective,
  window_hours,
  matured,
  censorship_reason,
  reach_value,
  engagement_value,
  conversion_value
) VALUES (
  %s::uuid, %s, %s, %s, %s, %s, %s, %s
)
ON CONFLICT (request_id, window_hours) DO UPDATE SET
  objective_effective = EXCLUDED.objective_effective,
  matured = EXCLUDED.matured,
  censorship_reason = EXCLUDED.censorship_reason,
  reach_value = EXCLUDED.reach_value,
  engagement_value = EXCLUDED.engagement_value,
  conversion_value = EXCLUDED.conversion_value,
  computed_at = NOW()
""",
            (
                payload["request_id"],
                payload["objective_effective"],
                int(payload["window_hours"]),
                bool(payload["matured"]),
                payload.get("censorship_reason"),
                float(payload.get("reach_value") or 0.0),
                float(payload.get("engagement_value") or 0.0),
                float(payload.get("conversion_value") or 0.0),
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build rec_outcome_events using snapshot lineage.")
    parser.add_argument("--db-url", type=str, default=None, help="Optional Postgres URL override.")
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=240,
        help="How far back to scan request serve events.",
    )
    parser.add_argument(
        "--window-hours",
        type=str,
        default="24,96",
        help="Comma-separated maturity windows.",
    )
    parser.add_argument(
        "--video-snapshots-table",
        type=str,
        default="video_snapshots",
        help="Table name for video snapshots.",
    )
    parser.add_argument(
        "--as-of-run-time",
        type=str,
        default=None,
        help="Optional as-of cutoff (ISO-8601). Defaults to now UTC.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/control_plane/outcome_attribution_report.json"),
        help="Report output path.",
    )
    args = parser.parse_args()

    as_of_run_time = _parse_dt(args.as_of_run_time) if args.as_of_run_time else _to_utc_now()
    since = as_of_run_time - timedelta(hours=max(1, int(args.lookback_hours)))
    windows = [int(item.strip()) for item in str(args.window_hours).split(",") if item.strip()]
    windows = [item for item in windows if item > 0]
    if not windows:
        raise ValueError("window-hours must contain at least one positive integer.")

    report: Dict[str, Any] = {
        "as_of_run_time": as_of_run_time.isoformat().replace("+00:00", "Z"),
        "lookback_hours": int(args.lookback_hours),
        "windows": windows,
        "rows_scanned": 0,
        "rows_written": 0,
        "censored": 0,
    }
    with get_connection(args.db_url) as conn:
        heads = _fetch_request_heads(conn, since=since)
        report["rows_scanned"] = len(heads)
        for head in heads:
            served_at = head["served_at"]
            if not isinstance(served_at, datetime):
                continue
            upper_bound = served_at + timedelta(hours=max(windows))
            snapshots = _fetch_snapshots(
                conn,
                table_name=str(args.video_snapshots_table),
                video_id=head["video_id"],
                upper_bound=upper_bound,
            )
            for window_hours in windows:
                payload = build_outcome_event(
                    request_id=head["request_id"],
                    objective_effective=head["objective_effective"],
                    served_at=served_at.astimezone(timezone.utc),
                    as_of_run_time=as_of_run_time,
                    window_hours=window_hours,
                    snapshots=snapshots,
                )
                _upsert_outcome(conn, payload)
                report["rows_written"] += 1
                if payload.get("censorship_reason"):
                    report["censored"] += 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

