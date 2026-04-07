#!/usr/bin/env python3
"""Compute daily drift report and persist rec_drift_daily."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from scraper.db.client import get_connection
from src.recommendation.control_plane import DriftThresholds, summarize_drift


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def _ensure_request_event_flags(conn) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            "ALTER TABLE IF EXISTS rec_request_events ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT FALSE;"
        )
        cursor.execute(
            "ALTER TABLE IF EXISTS rec_request_events ADD COLUMN IF NOT EXISTS injected_failure BOOLEAN NOT NULL DEFAULT FALSE;"
        )
        cursor.execute(
            "ALTER TABLE IF EXISTS rec_request_events ADD COLUMN IF NOT EXISTS traffic_class TEXT NOT NULL DEFAULT 'production';"
        )
        cursor.execute(
            "ALTER TABLE IF EXISTS rec_request_events ADD COLUMN IF NOT EXISTS policy_metadata JSONB NOT NULL DEFAULT '{}'::jsonb;"
        )
        cursor.execute(
            "ALTER TABLE IF EXISTS rec_request_events ADD COLUMN IF NOT EXISTS request_context JSONB NOT NULL DEFAULT '{}'::jsonb;"
        )
        cursor.execute(
            """
CREATE TABLE IF NOT EXISTS rec_ui_feedback_events (
  event_id BIGSERIAL PRIMARY KEY,
  request_id UUID NOT NULL,
  event_name TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  entity_id TEXT,
  section TEXT NOT NULL,
  rank INTEGER,
  objective_effective TEXT NOT NULL,
  experiment_id TEXT,
  variant TEXT,
  signal_strength TEXT NOT NULL,
  label_direction TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""
        )


def _fetch_objectives(
    conn,
    since: datetime,
    until: datetime,
    *,
    include_synthetic: bool,
    include_injected_failures: bool,
    request_ids: Sequence[str] | None = None,
) -> List[str]:
    scoped_request_ids = list(request_ids or [])
    with conn.cursor() as cursor:
        cursor.execute(
            """
SELECT DISTINCT objective_effective
FROM rec_request_events
WHERE served_at >= %s AND served_at < %s
  AND (%s OR COALESCE(is_synthetic, FALSE) = FALSE)
  AND (%s OR COALESCE(injected_failure, FALSE) = FALSE)
  AND (
    COALESCE(array_length(%s::uuid[], 1), 0) = 0
    OR request_id = ANY(%s::uuid[])
  )
ORDER BY objective_effective ASC
""",
            (
                since,
                until,
                bool(include_synthetic),
                bool(include_injected_failures),
                scoped_request_ids,
                scoped_request_ids,
            ),
        )
        rows = cursor.fetchall()
    return [str(row[0]) for row in rows if row and row[0]]


def _fetch_label_values(
    conn,
    objective: str,
    window_hours: int,
    since: datetime,
    until: datetime,
    *,
    include_synthetic: bool,
    include_injected_failures: bool,
    request_ids: Sequence[str] | None = None,
) -> List[float]:
    scoped_request_ids = list(request_ids or [])
    with conn.cursor() as cursor:
        cursor.execute(
            """
SELECT
  CASE
    WHEN event_name IN ('comparable_marked_relevant', 'comparable_saved', 'recommendation_marked_useful', 'recommendation_saved') THEN 1.0
    WHEN event_name IN ('comparable_marked_not_relevant', 'comparable_no_good_options', 'recommendation_marked_not_useful') THEN -1.0
    ELSE 0.0
  END AS label_value
FROM rec_ui_feedback_events uife
JOIN rec_request_events re ON re.request_id = uife.request_id
WHERE re.objective_effective = %s
  AND uife.created_at >= %s
  AND uife.created_at < %s
  AND uife.signal_strength = 'strong'
  AND (%s OR COALESCE(re.is_synthetic, FALSE) = FALSE)
  AND (%s OR COALESCE(re.injected_failure, FALSE) = FALSE)
  AND (
    COALESCE(array_length(%s::uuid[], 1), 0) = 0
    OR re.request_id = ANY(%s::uuid[])
  )
""",
            (
                objective,
                since,
                until,
                bool(include_synthetic),
                bool(include_injected_failures),
                scoped_request_ids,
                scoped_request_ids,
            ),
        )
        feedback_rows = cursor.fetchall()
    out: List[float] = []
    for row in feedback_rows:
        try:
            out.append(float(row[0]))
        except (TypeError, ValueError):
            continue
    if out:
        return out

    with conn.cursor() as cursor:
        cursor.execute(
            """
SELECT
  CASE
    WHEN %s = 'reach' THEN reach_value
    WHEN %s = 'conversion' THEN conversion_value
    ELSE engagement_value
  END AS target_value
FROM rec_outcome_events oe
JOIN rec_request_events re ON re.request_id = oe.request_id
WHERE oe.objective_effective = %s
  AND oe.window_hours = %s
  AND oe.matured = TRUE
  AND oe.computed_at >= %s
  AND oe.computed_at < %s
  AND (%s OR COALESCE(re.is_synthetic, FALSE) = FALSE)
  AND (%s OR COALESCE(re.injected_failure, FALSE) = FALSE)
  AND (
    COALESCE(array_length(%s::uuid[], 1), 0) = 0
    OR re.request_id = ANY(%s::uuid[])
  )
""",
            (
                objective,
                objective,
                objective,
                int(window_hours),
                since,
                until,
                bool(include_synthetic),
                bool(include_injected_failures),
                scoped_request_ids,
                scoped_request_ids,
            ),
        )
        rows = cursor.fetchall()
    for row in rows:
        try:
            out.append(float(row[0]))
        except (TypeError, ValueError):
            continue
    return out


def _fetch_feature_values(
    conn,
    objective: str,
    since: datetime,
    until: datetime,
    *,
    include_synthetic: bool,
    include_injected_failures: bool,
    request_ids: Sequence[str] | None = None,
) -> Dict[str, List[float]]:
    scoped_request_ids = list(request_ids or [])
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
  AND (%s OR COALESCE(re.is_synthetic, FALSE) = FALSE)
  AND (%s OR COALESCE(re.injected_failure, FALSE) = FALSE)
  AND (
    COALESCE(array_length(%s::uuid[], 1), 0) = 0
    OR re.request_id = ANY(%s::uuid[])
  )
""",
            (
                objective,
                since,
                until,
                bool(include_synthetic),
                bool(include_injected_failures),
                scoped_request_ids,
                scoped_request_ids,
            ),
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


def _fetch_policy_rates(
    conn,
    objective: str,
    since: datetime,
    until: datetime,
    *,
    include_synthetic: bool,
    include_injected_failures: bool,
    request_ids: Sequence[str] | None = None,
) -> Dict[str, float]:
    scoped_request_ids = list(request_ids or [])
    with conn.cursor() as cursor:
        cursor.execute(
            """
SELECT
  COUNT(*) AS total,
  SUM(CASE WHEN fallback_mode THEN 1 ELSE 0 END) AS fallback_total,
  SUM(CASE WHEN (latency_breakdown_ms->>'constraint_tier_used') = '3' THEN 1 ELSE 0 END) AS tier3_total,
  SUM(
    CASE
      WHEN COALESCE((policy_metadata->'dropped_by_rule'->>'author_cap')::double precision, 0.0) > 0.0
      THEN 1
      ELSE 0
    END
  ) AS author_cap_drop_requests,
  SUM(
    CASE
      WHEN COALESCE((policy_metadata->'dropped_by_rule'->>'language_mismatch')::double precision, 0.0) > 0.0
      THEN 1
      ELSE 0
    END
  ) AS strict_language_drop_requests,
  SUM(
    CASE
      WHEN COALESCE((policy_metadata->'dropped_by_rule'->>'locale_mismatch')::double precision, 0.0) > 0.0
      THEN 1
      ELSE 0
    END
  ) AS strict_locale_drop_requests,
  SUM(
    CASE
      WHEN COALESCE((policy_metadata->>'strict_language')::boolean, FALSE)
      THEN 1
      ELSE 0
    END
  ) AS strict_language_requests,
  SUM(
    CASE
      WHEN COALESCE((policy_metadata->>'strict_locale')::boolean, FALSE)
      THEN 1
      ELSE 0
    END
  ) AS strict_locale_requests
FROM rec_request_events
WHERE objective_effective = %s
  AND served_at >= %s
  AND served_at < %s
  AND (%s OR COALESCE(is_synthetic, FALSE) = FALSE)
  AND (%s OR COALESCE(injected_failure, FALSE) = FALSE)
  AND (
    COALESCE(array_length(%s::uuid[], 1), 0) = 0
    OR request_id = ANY(%s::uuid[])
  )
""",
            (
                objective,
                since,
                until,
                bool(include_synthetic),
                bool(include_injected_failures),
                scoped_request_ids,
                scoped_request_ids,
            ),
        )
        row = cursor.fetchone()
    total = float(row[0] or 0.0) if row else 0.0
    fallback_total = float(row[1] or 0.0) if row else 0.0
    tier3_total = float(row[2] or 0.0) if row else 0.0
    author_cap_drop_requests = float(row[3] or 0.0) if row else 0.0
    strict_language_drop_requests = float(row[4] or 0.0) if row else 0.0
    strict_locale_drop_requests = float(row[5] or 0.0) if row else 0.0
    strict_language_requests = float(row[6] or 0.0) if row else 0.0
    strict_locale_requests = float(row[7] or 0.0) if row else 0.0
    return {
        "sample_count": int(total),
        "fallback_rate": fallback_total / max(1.0, total),
        "constraint_tier_3_rate": tier3_total / max(1.0, total),
        "author_cap_drop_rate": author_cap_drop_requests / max(1.0, total),
        "strict_language_drop_rate": (
            strict_language_drop_requests / max(1.0, strict_language_requests)
        ),
        "strict_locale_drop_rate": (
            strict_locale_drop_requests / max(1.0, strict_locale_requests)
        ),
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


def _load_request_ids(path: Path) -> List[str]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(parsed, dict):
        values = parsed.get("request_ids")
    else:
        values = parsed
    if not isinstance(values, list):
        raise ValueError(f"request id payload must be a list (or {{\"request_ids\": [...]}}): {path}")
    out: List[str] = []
    for item in values:
        raw = str(item or "").strip()
        if raw:
            out.append(raw)
    return out


def _collect_request_id_scope(cli_request_ids: Sequence[str], request_ids_json: Path | None) -> List[str]:
    raw_ids: List[str] = [str(item).strip() for item in cli_request_ids if str(item).strip()]
    if request_ids_json is not None:
        raw_ids.extend(_load_request_ids(request_ids_json))
    seen: set[str] = set()
    normalized: List[str] = []
    for raw in raw_ids:
        try:
            canonical = str(uuid.UUID(raw))
        except ValueError as exc:
            raise ValueError(f"invalid request id in scope: {raw}") from exc
        if canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    return normalized


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
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Include synthetic/test recommendation traffic in drift computation.",
    )
    parser.add_argument(
        "--include-injected-failures",
        action="store_true",
        help="Include chaos/failure-injection traffic in drift computation.",
    )
    parser.add_argument(
        "--min-feature-samples",
        type=int,
        default=25,
        help="Minimum expected/current samples per feature key before drift is trigger-eligible.",
    )
    parser.add_argument(
        "--min-label-samples",
        type=int,
        default=25,
        help="Minimum expected/current samples per label window before drift is trigger-eligible.",
    )
    parser.add_argument(
        "--min-policy-samples",
        type=int,
        default=25,
        help="Minimum baseline/current request samples before policy drift is trigger-eligible.",
    )
    parser.add_argument(
        "--request-id",
        action="append",
        default=[],
        help="Optional request_id scope; may be repeated.",
    )
    parser.add_argument(
        "--request-ids-json",
        type=Path,
        default=None,
        help="Optional JSON file containing request IDs (list or {'request_ids': [...]}) to scope drift.",
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
    scoped_request_ids = _collect_request_id_scope(args.request_id, args.request_ids_json)

    report: Dict[str, Any] = {
        "as_of_date": as_of.date().isoformat(),
        "windows": {
            "baseline_start": baseline_start.isoformat().replace("+00:00", "Z"),
            "baseline_end": baseline_end.isoformat().replace("+00:00", "Z"),
            "current_start": current_start.isoformat().replace("+00:00", "Z"),
            "current_end": as_of.isoformat().replace("+00:00", "Z"),
        },
        "include_synthetic": bool(args.include_synthetic),
        "include_injected_failures": bool(args.include_injected_failures),
        "request_id_scope_count": len(scoped_request_ids),
        "thresholds": {
            "min_feature_samples": int(max(1, args.min_feature_samples)),
            "min_label_samples": int(max(1, args.min_label_samples)),
            "min_policy_samples": int(max(1, args.min_policy_samples)),
        },
        "objectives": {},
    }

    with get_connection(args.db_url) as conn:
        _ensure_request_event_flags(conn)
        objectives = _fetch_objectives(
            conn,
            since=baseline_start,
            until=as_of,
            include_synthetic=bool(args.include_synthetic),
            include_injected_failures=bool(args.include_injected_failures),
            request_ids=scoped_request_ids,
        )
        for objective in objectives:
            feature_expected = _fetch_feature_values(
                conn,
                objective=objective,
                since=baseline_start,
                until=baseline_end,
                include_synthetic=bool(args.include_synthetic),
                include_injected_failures=bool(args.include_injected_failures),
                request_ids=scoped_request_ids,
            )
            feature_actual = _fetch_feature_values(
                conn,
                objective=objective,
                since=current_start,
                until=as_of,
                include_synthetic=bool(args.include_synthetic),
                include_injected_failures=bool(args.include_injected_failures),
                request_ids=scoped_request_ids,
            )
            label_expected = {
                "primary_24h": _fetch_label_values(
                    conn,
                    objective=objective,
                    window_hours=24,
                    since=baseline_start,
                    until=baseline_end,
                    include_synthetic=bool(args.include_synthetic),
                    include_injected_failures=bool(args.include_injected_failures),
                    request_ids=scoped_request_ids,
                ),
                "stability_96h": _fetch_label_values(
                    conn,
                    objective=objective,
                    window_hours=96,
                    since=baseline_start,
                    until=baseline_end,
                    include_synthetic=bool(args.include_synthetic),
                    include_injected_failures=bool(args.include_injected_failures),
                    request_ids=scoped_request_ids,
                ),
            }
            label_actual = {
                "primary_24h": _fetch_label_values(
                    conn,
                    objective=objective,
                    window_hours=24,
                    since=current_start,
                    until=as_of,
                    include_synthetic=bool(args.include_synthetic),
                    include_injected_failures=bool(args.include_injected_failures),
                    request_ids=scoped_request_ids,
                ),
                "stability_96h": _fetch_label_values(
                    conn,
                    objective=objective,
                    window_hours=96,
                    since=current_start,
                    until=as_of,
                    include_synthetic=bool(args.include_synthetic),
                    include_injected_failures=bool(args.include_injected_failures),
                    request_ids=scoped_request_ids,
                ),
            }
            policy_baseline = _fetch_policy_rates(
                conn,
                objective=objective,
                since=baseline_start,
                until=baseline_end,
                include_synthetic=bool(args.include_synthetic),
                include_injected_failures=bool(args.include_injected_failures),
                request_ids=scoped_request_ids,
            )
            policy_current = _fetch_policy_rates(
                conn,
                objective=objective,
                since=current_start,
                until=as_of,
                include_synthetic=bool(args.include_synthetic),
                include_injected_failures=bool(args.include_injected_failures),
                request_ids=scoped_request_ids,
            )
            summary = summarize_drift(
                feature_expected=feature_expected,
                feature_actual=feature_actual,
                label_expected=label_expected,
                label_actual=label_actual,
                policy_baseline=policy_baseline,
                policy_current=policy_current,
                thresholds=DriftThresholds(
                    min_feature_samples=max(1, int(args.min_feature_samples)),
                    min_label_samples=max(1, int(args.min_label_samples)),
                    min_policy_samples=max(1, int(args.min_policy_samples)),
                ),
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
