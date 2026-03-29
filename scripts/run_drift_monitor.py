#!/usr/bin/env python3
"""Compute daily drift report and persist rec_drift_daily."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from scraper.db.client import get_connection
from src.recommendation.control_plane import DriftThresholds, summarize_drift


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def _fetch_objectives(conn, since: datetime, until: datetime) -> List[str]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
SELECT DISTINCT objective_effective
FROM rec_request_events
WHERE served_at >= %s AND served_at < %s
ORDER BY objective_effective ASC
""",
            (since, until),
        )
        rows = cursor.fetchall()
    return [str(row[0]) for row in rows if row and row[0]]


def _fetch_label_values(conn, objective: str, window_hours: int, since: datetime, until: datetime) -> List[float]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
SELECT
  CASE
    WHEN %s = 'reach' THEN reach_value
    WHEN %s = 'conversion' THEN conversion_value
    ELSE engagement_value
  END AS target_value
FROM rec_outcome_events
WHERE objective_effective = %s
  AND window_hours = %s
  AND matured = TRUE
  AND computed_at >= %s
  AND computed_at < %s
""",
            (objective, objective, objective, int(window_hours), since, until),
        )
        rows = cursor.fetchall()
    out: List[float] = []
    for row in rows:
        try:
            out.append(float(row[0]))
        except (TypeError, ValueError):
            continue
    return out


def _fetch_feature_values(conn, objective: str, since: datetime, until: datetime) -> Dict[str, List[float]]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
SELECT
  ce.score_calibrated,
  ce.policy_adjusted_score,
  ce.retrieval_branch_scores->>'fused',
  ce.retrieval_branch_scores->>'multimodal'
FROM rec_candidate_events ce
JOIN rec_request_events re ON re.request_id = ce.request_id
WHERE re.objective_effective = %s
  AND ce.stage = 'ranked_universe'
  AND re.served_at >= %s
  AND re.served_at < %s
""",
            (objective, since, until),
        )
        rows = cursor.fetchall()
    out: Dict[str, List[float]] = {
        "mandatory_score_calibrated": [],
        "mandatory_retrieval_fused": [],
        "optional_policy_adjusted": [],
        "optional_multimodal": [],
    }
    for row in rows:
        values = [row[0], row[1], row[2], row[3]]
        keys = [
            "mandatory_score_calibrated",
            "optional_policy_adjusted",
            "mandatory_retrieval_fused",
            "optional_multimodal",
        ]
        for key, raw in zip(keys, values):
            try:
                out[key].append(float(raw))
            except (TypeError, ValueError):
                continue
    return out


def _fetch_policy_rates(conn, objective: str, since: datetime, until: datetime) -> Dict[str, float]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
SELECT
  COUNT(*) AS total,
  SUM(CASE WHEN fallback_mode THEN 1 ELSE 0 END) AS fallback_total,
  SUM(CASE WHEN (latency_breakdown_ms->>'constraint_tier_used') = '3' THEN 1 ELSE 0 END) AS tier3_total
FROM rec_request_events
WHERE objective_effective = %s
  AND served_at >= %s
  AND served_at < %s
""",
            (objective, since, until),
        )
        row = cursor.fetchone()
    total = float(row[0] or 0.0) if row else 0.0
    fallback_total = float(row[1] or 0.0) if row else 0.0
    tier3_total = float(row[2] or 0.0) if row else 0.0
    return {
        "fallback_rate": fallback_total / max(1.0, total),
        "constraint_tier_3_rate": tier3_total / max(1.0, total),
        "author_cap_drop_rate": 0.0,
        "strict_language_drop_rate": 0.0,
    }


def _upsert_drift_row(conn, payload: Dict[str, Any]) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
INSERT INTO rec_drift_daily (
  drift_date,
  objective_effective,
  segment_id,
  feature_drift,
  label_drift,
  policy_drift,
  breach_severity,
  trigger_recommendation
) VALUES (
  %s::date, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s
)
ON CONFLICT (drift_date, objective_effective, segment_id) DO UPDATE SET
  feature_drift = EXCLUDED.feature_drift,
  label_drift = EXCLUDED.label_drift,
  policy_drift = EXCLUDED.policy_drift,
  breach_severity = EXCLUDED.breach_severity,
  trigger_recommendation = EXCLUDED.trigger_recommendation,
  created_at = NOW()
""",
            (
                payload["drift_date"],
                payload["objective_effective"],
                payload["segment_id"],
                json.dumps(payload["feature_drift"], ensure_ascii=False),
                json.dumps(payload["label_drift"], ensure_ascii=False),
                json.dumps(payload["policy_drift"], ensure_ascii=False),
                payload["breach_severity"],
                payload["trigger_recommendation"],
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute drift report and persist rec_drift_daily.")
    parser.add_argument("--db-url", type=str, default=None, help="Optional Postgres URL override.")
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Optional as-of date (YYYY-MM-DD). Defaults to UTC today.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/control_plane/drift_report.json"),
        help="Output drift report artifact path.",
    )
    args = parser.parse_args()

    as_of = (
        datetime.fromisoformat(f"{args.as_of_date}T00:00:00+00:00")
        if args.as_of_date
        else datetime.now(timezone.utc)
    )
    as_of = _to_utc(as_of)
    current_start = as_of - timedelta(days=7)
    baseline_start = as_of - timedelta(days=14)
    baseline_end = current_start

    report: Dict[str, Any] = {
        "as_of_date": as_of.date().isoformat(),
        "windows": {
            "baseline_start": baseline_start.isoformat().replace("+00:00", "Z"),
            "baseline_end": baseline_end.isoformat().replace("+00:00", "Z"),
            "current_start": current_start.isoformat().replace("+00:00", "Z"),
            "current_end": as_of.isoformat().replace("+00:00", "Z"),
        },
        "objectives": {},
    }

    with get_connection(args.db_url) as conn:
        objectives = _fetch_objectives(conn, since=baseline_start, until=as_of)
        for objective in objectives:
            feature_expected = _fetch_feature_values(
                conn, objective=objective, since=baseline_start, until=baseline_end
            )
            feature_actual = _fetch_feature_values(
                conn, objective=objective, since=current_start, until=as_of
            )
            label_expected = {
                "primary_24h": _fetch_label_values(
                    conn, objective=objective, window_hours=24, since=baseline_start, until=baseline_end
                ),
                "stability_96h": _fetch_label_values(
                    conn, objective=objective, window_hours=96, since=baseline_start, until=baseline_end
                ),
            }
            label_actual = {
                "primary_24h": _fetch_label_values(
                    conn, objective=objective, window_hours=24, since=current_start, until=as_of
                ),
                "stability_96h": _fetch_label_values(
                    conn, objective=objective, window_hours=96, since=current_start, until=as_of
                ),
            }
            policy_baseline = _fetch_policy_rates(
                conn, objective=objective, since=baseline_start, until=baseline_end
            )
            policy_current = _fetch_policy_rates(
                conn, objective=objective, since=current_start, until=as_of
            )
            summary = summarize_drift(
                feature_expected=feature_expected,
                feature_actual=feature_actual,
                label_expected=label_expected,
                label_actual=label_actual,
                policy_baseline=policy_baseline,
                policy_current=policy_current,
                thresholds=DriftThresholds(),
            )
            report["objectives"][objective] = summary
            _upsert_drift_row(
                conn,
                {
                    "drift_date": as_of.date().isoformat(),
                    "objective_effective": objective,
                    "segment_id": "global",
                    "feature_drift": summary["feature_drift"],
                    "label_drift": summary["label_drift"],
                    "policy_drift": summary["policy_drift"],
                    "breach_severity": summary["severity"],
                    "trigger_recommendation": summary["trigger_recommendation"],
                },
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

