#!/usr/bin/env python3
"""Build objective-level experiment report with immediate feedback and delayed-support KPIs."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from scraper.db.client import get_connection


def _avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _summarize_variant(values: List[Dict[str, Any]], objective: str) -> Dict[str, Any]:
    if not values:
        return {
            "samples": 0,
            "explicit_feedback_samples": 0,
            "matured_primary_24h_samples": 0,
            "matured_stability_96h_samples": 0,
            "censored_primary_24h_samples": 0,
            "censored_stability_96h_samples": 0,
            "primary_kpi_24h": 0.0,
            "stability_kpi_96h": 0.0,
            "fallback_rate": 0.0,
            "latency_p95_ms": 0.0,
            "policy_violation_rate": 0.0,
            "author_cap_drop_rate": 0.0,
            "strict_language_drop_rate": 0.0,
            "strict_locale_drop_rate": 0.0,
            "comparable_relevant_rate": 0.0,
            "comparable_not_relevant_rate": 0.0,
            "comparable_save_rate": 0.0,
            "comparable_no_good_options_rate": 0.0,
            "recommendation_useful_rate": 0.0,
            "recommendation_not_useful_rate": 0.0,
            "recommendation_save_rate": 0.0,
            "report_export_rate": 0.0,
            "followup_question_rate": 0.0,
            "comparables_opened_per_report": 0.0,
            "details_opened_per_report": 0.0,
        }
    primary_values = [
        float(item[f"{objective}_24h"])
        for item in values
        if item.get(f"{objective}_24h") is not None
    ]
    stability_values = [
        float(item[f"{objective}_96h"])
        for item in values
        if item.get(f"{objective}_96h") is not None
    ]
    latencies = sorted(float(item["latency_total_ms"]) for item in values)
    p95_index = int(round((len(latencies) - 1) * 0.95))
    strict_language_requests = sum(
        1 for item in values if item.get("strict_language_enabled")
    )
    strict_locale_requests = sum(
        1 for item in values if item.get("strict_locale_enabled")
    )
    explicit_feedback_samples = sum(
        1 for item in values if int(item.get("explicit_feedback_count") or 0) > 0
    )
    return {
        "samples": len(values),
        "explicit_feedback_samples": explicit_feedback_samples,
        "matured_primary_24h_samples": len(primary_values),
        "matured_stability_96h_samples": len(stability_values),
        "censored_primary_24h_samples": max(0, len(values) - len(primary_values)),
        "censored_stability_96h_samples": max(0, len(values) - len(stability_values)),
        "primary_kpi_24h": round(_avg(primary_values), 6),
        "stability_kpi_96h": round(_avg(stability_values), 6),
        "fallback_rate": round(
            sum(1 for item in values if item["fallback_mode"]) / max(1, len(values)),
            6,
        ),
        "latency_p95_ms": round(latencies[p95_index] if latencies else 0.0, 6),
        "policy_violation_rate": round(
            sum(1 for item in values if item.get("policy_violation")) / max(1, len(values)),
            6,
        ),
        "author_cap_drop_rate": round(
            sum(1 for item in values if float(item.get("author_cap_drops") or 0.0) > 0.0)
            / max(1, len(values)),
            6,
        ),
        "strict_language_drop_rate": round(
            sum(
                1
                for item in values
                if item.get("strict_language_enabled")
                and float(item.get("language_mismatch_drops") or 0.0) > 0.0
            )
            / max(1, strict_language_requests),
            6,
        ),
        "strict_locale_drop_rate": round(
            sum(
                1
                for item in values
                if item.get("strict_locale_enabled")
                and float(item.get("locale_mismatch_drops") or 0.0) > 0.0
            )
            / max(1, strict_locale_requests),
            6,
        ),
        "comparable_relevant_rate": round(
            sum(1 for item in values if int(item.get("comparable_relevant_count") or 0) > 0)
            / max(1, len(values)),
            6,
        ),
        "comparable_not_relevant_rate": round(
            sum(
                1 for item in values if int(item.get("comparable_not_relevant_count") or 0) > 0
            )
            / max(1, len(values)),
            6,
        ),
        "comparable_save_rate": round(
            sum(1 for item in values if int(item.get("comparable_save_count") or 0) > 0)
            / max(1, len(values)),
            6,
        ),
        "comparable_no_good_options_rate": round(
            sum(
                1
                for item in values
                if int(item.get("comparable_no_good_options_count") or 0) > 0
            )
            / max(1, len(values)),
            6,
        ),
        "recommendation_useful_rate": round(
            sum(1 for item in values if int(item.get("recommendation_useful_count") or 0) > 0)
            / max(1, len(values)),
            6,
        ),
        "recommendation_not_useful_rate": round(
            sum(
                1
                for item in values
                if int(item.get("recommendation_not_useful_count") or 0) > 0
            )
            / max(1, len(values)),
            6,
        ),
        "recommendation_save_rate": round(
            sum(1 for item in values if int(item.get("recommendation_save_count") or 0) > 0)
            / max(1, len(values)),
            6,
        ),
        "report_export_rate": round(
            sum(1 for item in values if int(item.get("report_export_count") or 0) > 0)
            / max(1, len(values)),
            6,
        ),
        "followup_question_rate": round(
            sum(1 for item in values if int(item.get("followup_question_count") or 0) > 0)
            / max(1, len(values)),
            6,
        ),
        "comparables_opened_per_report": round(
            _avg([float(item.get("comparable_open_count") or 0.0) for item in values]),
            6,
        ),
        "details_opened_per_report": round(
            _avg([float(item.get("comparable_details_open_count") or 0.0) for item in values]),
            6,
        ),
    }


def _build_comparison_block(
    *,
    control: Dict[str, Any],
    treatment: Dict[str, Any],
    min_matured_primary_samples_per_variant: int,
    min_matured_stability_samples_per_variant: int,
    min_explicit_feedback_samples_per_variant: int = 10,
) -> Dict[str, Any]:
    control_primary_n = int(control.get("matured_primary_24h_samples") or 0)
    treatment_primary_n = int(treatment.get("matured_primary_24h_samples") or 0)
    control_stability_n = int(control.get("matured_stability_96h_samples") or 0)
    treatment_stability_n = int(treatment.get("matured_stability_96h_samples") or 0)
    primary_ready = (
        control_primary_n >= max(1, int(min_matured_primary_samples_per_variant))
        and treatment_primary_n >= max(1, int(min_matured_primary_samples_per_variant))
    )
    stability_ready = (
        control_stability_n >= max(1, int(min_matured_stability_samples_per_variant))
        and treatment_stability_n >= max(1, int(min_matured_stability_samples_per_variant))
    )
    control_feedback_n = int(control.get("explicit_feedback_samples") or 0)
    treatment_feedback_n = int(treatment.get("explicit_feedback_samples") or 0)
    feedback_ready = (
        control_feedback_n >= max(1, int(min_explicit_feedback_samples_per_variant))
        and treatment_feedback_n >= max(1, int(min_explicit_feedback_samples_per_variant))
    )
    evidence_sufficient = bool(feedback_ready or (primary_ready and stability_ready))
    primary_delta = float(treatment.get("primary_kpi_24h") or 0.0) - float(
        control.get("primary_kpi_24h") or 0.0
    )
    stability_delta = float(treatment.get("stability_kpi_96h") or 0.0) - float(
        control.get("stability_kpi_96h") or 0.0
    )
    fallback_delta = float(treatment.get("fallback_rate") or 0.0) - float(
        control.get("fallback_rate") or 0.0
    )
    latency_delta = float(treatment.get("latency_p95_ms") or 0.0) - float(
        control.get("latency_p95_ms") or 0.0
    )
    policy_violation_delta = float(treatment.get("policy_violation_rate") or 0.0) - float(
        control.get("policy_violation_rate") or 0.0
    )
    comparable_relevant_delta = float(treatment.get("comparable_relevant_rate") or 0.0) - float(
        control.get("comparable_relevant_rate") or 0.0
    )
    recommendation_useful_delta = float(
        treatment.get("recommendation_useful_rate") or 0.0
    ) - float(control.get("recommendation_useful_rate") or 0.0)
    immediate_feedback_delta = round(
        (recommendation_useful_delta * 0.6) + (comparable_relevant_delta * 0.4),
        6,
    )
    if not evidence_sufficient:
        verdict = "insufficient_data"
    elif feedback_ready and immediate_feedback_delta > 0:
        verdict = "treatment_leads"
    elif feedback_ready and immediate_feedback_delta < 0:
        verdict = "control_leads"
    elif primary_delta > 0:
        verdict = "treatment_leads"
    elif primary_delta < 0:
        verdict = "control_leads"
    else:
        verdict = "tie"
    return {
        "evidence_sufficient": evidence_sufficient,
        "thresholds": {
            "min_matured_primary_samples_per_variant": int(
                max(1, min_matured_primary_samples_per_variant)
            ),
            "min_matured_stability_samples_per_variant": int(
                max(1, min_matured_stability_samples_per_variant)
            ),
            "min_explicit_feedback_samples_per_variant": int(
                max(1, min_explicit_feedback_samples_per_variant)
            ),
        },
        "matured_support": {
            "control_primary_24h": control_primary_n,
            "treatment_primary_24h": treatment_primary_n,
            "control_stability_96h": control_stability_n,
            "treatment_stability_96h": treatment_stability_n,
        },
        "feedback_support": {
            "control_explicit_feedback": control_feedback_n,
            "treatment_explicit_feedback": treatment_feedback_n,
            "evidence_basis": "immediate_feedback" if feedback_ready else "delayed_outcomes",
        },
        "deltas": {
            "primary_kpi_24h_treatment_minus_control": round(primary_delta, 6),
            "stability_kpi_96h_treatment_minus_control": round(stability_delta, 6),
            "fallback_rate_treatment_minus_control": round(fallback_delta, 6),
            "latency_p95_ms_treatment_minus_control": round(latency_delta, 6),
            "policy_violation_rate_treatment_minus_control": round(policy_violation_delta, 6),
            "comparable_relevant_rate_treatment_minus_control": round(
                comparable_relevant_delta, 6
            ),
            "recommendation_useful_rate_treatment_minus_control": round(
                recommendation_useful_delta, 6
            ),
            "immediate_feedback_score_treatment_minus_control": immediate_feedback_delta,
        },
        "verdict": verdict,
    }


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


def _fetch_variant_rows(
    conn,
    objective: str,
    since: datetime,
    until: datetime,
    *,
    include_synthetic: bool,
    include_injected_failures: bool,
    request_ids: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    scoped_request_ids = list(request_ids or [])
    with conn.cursor() as cursor:
        cursor.execute(
            """
SELECT
  re.variant,
  re.request_id,
  re.fallback_mode,
  (re.latency_breakdown_ms->>'total')::double precision AS latency_total_ms,
  o24.matured AS matured_24h,
  o96.matured AS matured_96h,
  o24.censorship_reason AS censorship_24h,
  o96.censorship_reason AS censorship_96h,
  o24.engagement_value AS engagement_24h,
  o96.engagement_value AS engagement_96h,
  o24.reach_value AS reach_24h,
  o96.reach_value AS reach_96h,
  o24.conversion_value AS conversion_24h,
  o96.conversion_value AS conversion_96h,
  COALESCE((re.policy_metadata->'dropped_by_rule'->>'author_cap')::double precision, 0.0) AS author_cap_drops,
  COALESCE((re.policy_metadata->'dropped_by_rule'->>'language_mismatch')::double precision, 0.0) AS language_mismatch_drops,
  COALESCE((re.policy_metadata->'dropped_by_rule'->>'locale_mismatch')::double precision, 0.0) AS locale_mismatch_drops,
  COALESCE((re.policy_metadata->'dropped_by_rule'->>'age_limit')::double precision, 0.0) AS age_limit_drops,
  COALESCE((re.policy_metadata->>'strict_language')::boolean, FALSE) AS strict_language_enabled,
  COALESCE((re.policy_metadata->>'strict_locale')::boolean, FALSE) AS strict_locale_enabled,
  COALESCE(fb.comparable_relevant_count, 0) AS comparable_relevant_count,
  COALESCE(fb.comparable_not_relevant_count, 0) AS comparable_not_relevant_count,
  COALESCE(fb.comparable_save_count, 0) AS comparable_save_count,
  COALESCE(fb.comparable_no_good_options_count, 0) AS comparable_no_good_options_count,
  COALESCE(fb.recommendation_useful_count, 0) AS recommendation_useful_count,
  COALESCE(fb.recommendation_not_useful_count, 0) AS recommendation_not_useful_count,
  COALESCE(fb.recommendation_save_count, 0) AS recommendation_save_count,
  COALESCE(fb.report_export_count, 0) AS report_export_count,
  COALESCE(fb.followup_question_count, 0) AS followup_question_count,
  COALESCE(fb.comparable_open_count, 0) AS comparable_open_count,
  COALESCE(fb.comparable_details_open_count, 0) AS comparable_details_open_count,
  COALESCE(fb.explicit_feedback_count, 0) AS explicit_feedback_count
FROM rec_request_events re
LEFT JOIN rec_outcome_events o24
  ON o24.request_id = re.request_id AND o24.window_hours = 24
LEFT JOIN rec_outcome_events o96
  ON o96.request_id = re.request_id AND o96.window_hours = 96
LEFT JOIN (
  SELECT
    request_id,
    SUM(CASE WHEN event_name = 'comparable_marked_relevant' THEN 1 ELSE 0 END) AS comparable_relevant_count,
    SUM(CASE WHEN event_name = 'comparable_marked_not_relevant' THEN 1 ELSE 0 END) AS comparable_not_relevant_count,
    SUM(CASE WHEN event_name = 'comparable_saved' THEN 1 ELSE 0 END) AS comparable_save_count,
    SUM(CASE WHEN event_name = 'comparable_no_good_options' THEN 1 ELSE 0 END) AS comparable_no_good_options_count,
    SUM(CASE WHEN event_name = 'recommendation_marked_useful' THEN 1 ELSE 0 END) AS recommendation_useful_count,
    SUM(CASE WHEN event_name = 'recommendation_marked_not_useful' THEN 1 ELSE 0 END) AS recommendation_not_useful_count,
    SUM(CASE WHEN event_name = 'recommendation_saved' THEN 1 ELSE 0 END) AS recommendation_save_count,
    SUM(CASE WHEN event_name = 'report_exported' THEN 1 ELSE 0 END) AS report_export_count,
    SUM(CASE WHEN event_name = 'report_followup_asked' THEN 1 ELSE 0 END) AS followup_question_count,
    SUM(CASE WHEN event_name = 'comparable_opened' THEN 1 ELSE 0 END) AS comparable_open_count,
    SUM(CASE WHEN event_name = 'comparable_details_opened' THEN 1 ELSE 0 END) AS comparable_details_open_count,
    SUM(
      CASE
        WHEN event_name IN (
          'comparable_marked_relevant',
          'comparable_marked_not_relevant',
          'comparable_saved',
          'comparable_no_good_options',
          'recommendation_marked_useful',
          'recommendation_marked_not_useful',
          'recommendation_saved'
        )
        THEN 1
        ELSE 0
      END
    ) AS explicit_feedback_count
  FROM rec_ui_feedback_events
  GROUP BY request_id
) fb
  ON fb.request_id = re.request_id
WHERE re.objective_effective = %s
  AND re.served_at >= %s
  AND re.served_at < %s
  AND re.variant IN ('control', 'treatment')
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
    out: List[Dict[str, Any]] = []
    for row in rows:
        matured_24h = bool(row[4]) if row[4] is not None else False
        matured_96h = bool(row[5]) if row[5] is not None else False
        engagement_24h = float(row[8]) if matured_24h and row[8] is not None else None
        engagement_96h = float(row[9]) if matured_96h and row[9] is not None else None
        reach_24h = float(row[10]) if matured_24h and row[10] is not None else None
        reach_96h = float(row[11]) if matured_96h and row[11] is not None else None
        conversion_24h = float(row[12]) if matured_24h and row[12] is not None else None
        conversion_96h = float(row[13]) if matured_96h and row[13] is not None else None
        author_cap_drops = float(row[14] or 0.0)
        language_mismatch_drops = float(row[15] or 0.0)
        locale_mismatch_drops = float(row[16] or 0.0)
        age_limit_drops = float(row[17] or 0.0)
        strict_language_enabled = bool(row[18])
        strict_locale_enabled = bool(row[19])
        policy_violation = any(
            value > 0.0
            for value in (
                author_cap_drops,
                language_mismatch_drops,
                locale_mismatch_drops,
                age_limit_drops,
            )
        )
        out.append(
            {
                "variant": str(row[0]),
                "request_id": str(row[1]),
                "fallback_mode": bool(row[2]),
                "latency_total_ms": float(row[3] or 0.0),
                "matured_24h": matured_24h,
                "matured_96h": matured_96h,
                "censorship_24h": str(row[6] or "") if row[6] is not None else None,
                "censorship_96h": str(row[7] or "") if row[7] is not None else None,
                "engagement_24h": engagement_24h,
                "engagement_96h": engagement_96h,
                "reach_24h": reach_24h,
                "reach_96h": reach_96h,
                "conversion_24h": conversion_24h,
                "conversion_96h": conversion_96h,
                "policy_violation": bool(policy_violation),
                "author_cap_drops": author_cap_drops,
                "language_mismatch_drops": language_mismatch_drops,
                "locale_mismatch_drops": locale_mismatch_drops,
                "age_limit_drops": age_limit_drops,
                "strict_language_enabled": strict_language_enabled,
                "strict_locale_enabled": strict_locale_enabled,
                "comparable_relevant_count": int(row[20] or 0),
                "comparable_not_relevant_count": int(row[21] or 0),
                "comparable_save_count": int(row[22] or 0),
                "comparable_no_good_options_count": int(row[23] or 0),
                "recommendation_useful_count": int(row[24] or 0),
                "recommendation_not_useful_count": int(row[25] or 0),
                "recommendation_save_count": int(row[26] or 0),
                "report_export_count": int(row[27] or 0),
                "followup_question_count": int(row[28] or 0),
                "comparable_open_count": int(row[29] or 0),
                "comparable_details_open_count": int(row[30] or 0),
                "explicit_feedback_count": int(row[31] or 0),
            }
        )
    return out


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
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Include synthetic/test recommendation traffic in experiment analysis.",
    )
    parser.add_argument(
        "--include-injected-failures",
        action="store_true",
        help="Include chaos/failure-injection traffic in experiment analysis.",
    )
    parser.add_argument(
        "--min-matured-primary-samples-per-variant",
        type=int,
        default=20,
        help="Minimum matured 24h outcomes in each variant before comparison is evidence-sufficient.",
    )
    parser.add_argument(
        "--min-matured-stability-samples-per-variant",
        type=int,
        default=10,
        help="Minimum matured 96h outcomes in each variant before comparison is evidence-sufficient.",
    )
    parser.add_argument(
        "--min-explicit-feedback-samples-per-variant",
        type=int,
        default=10,
        help="Minimum explicit feedback samples per variant before immediate-feedback comparison is evidence-sufficient.",
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
        help="Optional JSON file containing request IDs (list or {'request_ids': [...]}) to scope experiment analysis.",
    )
    args = parser.parse_args()

    until = datetime.now(timezone.utc)
    since = until - timedelta(days=max(1, int(args.lookback_days)))
    scoped_request_ids = _collect_request_id_scope(args.request_id, args.request_ids_json)
    report: Dict[str, Any] = {
        "generated_at": until.isoformat().replace("+00:00", "Z"),
        "lookback_days": int(args.lookback_days),
        "include_synthetic": bool(args.include_synthetic),
        "include_injected_failures": bool(args.include_injected_failures),
        "request_id_scope_count": len(scoped_request_ids),
        "comparison_thresholds": {
            "min_matured_primary_samples_per_variant": int(
                max(1, args.min_matured_primary_samples_per_variant)
            ),
            "min_matured_stability_samples_per_variant": int(
                max(1, args.min_matured_stability_samples_per_variant)
            ),
            "min_explicit_feedback_samples_per_variant": int(
                max(1, args.min_explicit_feedback_samples_per_variant)
            ),
        },
        "objectives": {},
    }
    with get_connection(args.db_url) as conn:
        _ensure_request_event_flags(conn)
        for objective in _fetch_objectives(
            conn,
            since=since,
            until=until,
            include_synthetic=bool(args.include_synthetic),
            include_injected_failures=bool(args.include_injected_failures),
            request_ids=scoped_request_ids,
        ):
            rows = _fetch_variant_rows(
                conn,
                objective=objective,
                since=since,
                until=until,
                include_synthetic=bool(args.include_synthetic),
                include_injected_failures=bool(args.include_injected_failures),
                request_ids=scoped_request_ids,
            )
            grouped: Dict[str, List[Dict[str, Any]]] = {"control": [], "treatment": []}
            for row in rows:
                grouped.setdefault(str(row["variant"]), []).append(row)
            control_summary = _summarize_variant(grouped.get("control") or [], objective)
            treatment_summary = _summarize_variant(grouped.get("treatment") or [], objective)
            comparison = _build_comparison_block(
                control=control_summary,
                treatment=treatment_summary,
                min_matured_primary_samples_per_variant=int(
                    max(1, args.min_matured_primary_samples_per_variant)
                ),
                min_matured_stability_samples_per_variant=int(
                    max(1, args.min_matured_stability_samples_per_variant)
                ),
                min_explicit_feedback_samples_per_variant=int(
                    max(1, args.min_explicit_feedback_samples_per_variant)
                ),
            )
            report["objectives"][objective] = {
                "control": control_summary,
                "treatment": treatment_summary,
                "comparison": comparison,
            }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
