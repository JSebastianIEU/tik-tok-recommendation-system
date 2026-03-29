#!/usr/bin/env python3
"""Build objective-level experiment report with 24h primary and 96h stability KPIs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from scraper.db.client import get_connection


def _avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


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


def _fetch_variant_rows(conn, objective: str, since: datetime, until: datetime) -> List[Dict[str, Any]]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
SELECT
  re.variant,
  re.fallback_mode,
  (re.latency_breakdown_ms->>'total')::double precision AS latency_total_ms,
  o24.engagement_value AS engagement_24h,
  o96.engagement_value AS engagement_96h,
  o24.reach_value AS reach_24h,
  o96.reach_value AS reach_96h,
  o24.conversion_value AS conversion_24h,
  o96.conversion_value AS conversion_96h
FROM rec_request_events re
LEFT JOIN rec_outcome_events o24
  ON o24.request_id = re.request_id AND o24.window_hours = 24 AND o24.matured = TRUE
LEFT JOIN rec_outcome_events o96
  ON o96.request_id = re.request_id AND o96.window_hours = 96 AND o96.matured = TRUE
WHERE re.objective_effective = %s
  AND re.served_at >= %s
  AND re.served_at < %s
  AND re.variant IN ('control', 'treatment')
""",
            (objective, since, until),
        )
        rows = cursor.fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "variant": str(row[0]),
                "fallback_mode": bool(row[1]),
                "latency_total_ms": float(row[2] or 0.0),
                "engagement_24h": float(row[3] or 0.0),
                "engagement_96h": float(row[4] or 0.0),
                "reach_24h": float(row[5] or 0.0),
                "reach_96h": float(row[6] or 0.0),
                "conversion_24h": float(row[7] or 0.0),
                "conversion_96h": float(row[8] or 0.0),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run experiment KPI analysis.")
    parser.add_argument("--db-url", type=str, default=None, help="Optional Postgres URL override.")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=14,
        help="Lookback window for exposures and outcomes.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/control_plane/experiment_report.json"),
        help="Output report path.",
    )
    args = parser.parse_args()

    until = datetime.now(timezone.utc)
    since = until - timedelta(days=max(1, int(args.lookback_days)))
    report: Dict[str, Any] = {
        "generated_at": until.isoformat().replace("+00:00", "Z"),
        "lookback_days": int(args.lookback_days),
        "objectives": {},
    }
    with get_connection(args.db_url) as conn:
        for objective in _fetch_objectives(conn, since=since, until=until):
            rows = _fetch_variant_rows(conn, objective=objective, since=since, until=until)
            grouped: Dict[str, List[Dict[str, Any]]] = {"control": [], "treatment": []}
            for row in rows:
                grouped.setdefault(str(row["variant"]), []).append(row)
            objective_payload: Dict[str, Any] = {}
            for variant, values in grouped.items():
                if not values:
                    objective_payload[variant] = {
                        "samples": 0,
                        "primary_kpi_24h": 0.0,
                        "stability_kpi_96h": 0.0,
                        "fallback_rate": 0.0,
                        "latency_p95_ms": 0.0,
                        "policy_violation_rate": 0.0,
                    }
                    continue
                primary_values = [item[f"{objective}_24h"] for item in values if f"{objective}_24h" in item]
                stability_values = [item[f"{objective}_96h"] for item in values if f"{objective}_96h" in item]
                latencies = sorted(item["latency_total_ms"] for item in values)
                p95_index = int(round((len(latencies) - 1) * 0.95))
                objective_payload[variant] = {
                    "samples": len(values),
                    "primary_kpi_24h": round(_avg(primary_values), 6),
                    "stability_kpi_96h": round(_avg(stability_values), 6),
                    "fallback_rate": round(
                        sum(1 for item in values if item["fallback_mode"]) / max(1, len(values)),
                        6,
                    ),
                    "latency_p95_ms": round(latencies[p95_index] if latencies else 0.0, 6),
                    "policy_violation_rate": 0.0,
                }
            report["objectives"][objective] = objective_payload

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

