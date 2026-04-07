#!/usr/bin/env python3
"""Trigger scheduled/drift retrain decisions and persist rec_retrain_runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scraper.db.client import get_connection
from src.recommendation.control_plane import build_retrain_decision, should_trigger_retrain


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metrics(bundle_dir: Path) -> Dict[str, Any]:
    metrics_path = bundle_dir / "metrics" / "objective_metrics.json"
    if not metrics_path.exists():
        return {}
    return _read_json(metrics_path)


def _non_regression_gate(
    *,
    baseline_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    threshold_ratio: float = 0.995,
    retriever_threshold_ratio: float = 0.995,
) -> Tuple[bool, Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {}
    ok = True
    for objective, baseline in baseline_metrics.items():
        baseline_ndcg = float((baseline or {}).get("ranker", {}).get("ndcg@10") or 0.0)
        baseline_mrr = float((baseline or {}).get("ranker", {}).get("mrr@20") or 0.0)
        baseline_recall = float((baseline or {}).get("retriever", {}).get("recall@100") or 0.0)
        candidate = candidate_metrics.get(objective, {})
        candidate_ndcg = float((candidate or {}).get("ranker", {}).get("ndcg@10") or 0.0)
        candidate_mrr = float((candidate or {}).get("ranker", {}).get("mrr@20") or 0.0)
        candidate_recall = float((candidate or {}).get("retriever", {}).get("recall@100") or 0.0)
        ranker_ok = (
            candidate_ndcg >= baseline_ndcg * threshold_ratio
            and candidate_mrr >= baseline_mrr * threshold_ratio
        )
        retriever_ok = (
            candidate_recall >= baseline_recall * retriever_threshold_ratio
            if baseline_recall > 0.0
            else True
        )
        objective_ok = (
            ranker_ok and retriever_ok
        )
        diagnostics[objective] = {
            "baseline_ndcg@10": baseline_ndcg,
            "baseline_mrr@20": baseline_mrr,
            "candidate_ndcg@10": candidate_ndcg,
            "candidate_mrr@20": candidate_mrr,
            "baseline_recall@100": baseline_recall,
            "candidate_recall@100": candidate_recall,
            "ranker_gate_passed": ranker_ok,
            "retriever_gate_passed": retriever_ok,
            "passed": objective_ok,
        }
        if not objective_ok:
            ok = False
    return ok, diagnostics


def _fetch_recent_drift(conn) -> Dict[str, List[Dict[str, str]]]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
SELECT objective_effective, breach_severity, trigger_recommendation
FROM rec_drift_daily
WHERE segment_id = 'global'
ORDER BY drift_date DESC
LIMIT 60
"""
        )
        rows = cursor.fetchall()
    by_objective: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        objective = str(row[0])
        severity = str(row[1] or "ok")
        recommendation = str(row[2] or "none")
        by_objective.setdefault(objective, []).append(
            {"severity": severity, "trigger_recommendation": recommendation}
        )
    return by_objective


def _ensure_ui_feedback_events_table(conn) -> None:
    with conn.cursor() as cursor:
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


def _fetch_recent_outcome_support(
    conn,
    *,
    since: datetime,
    until: datetime,
    include_synthetic: bool,
    include_injected_failures: bool,
    request_ids: Sequence[str] | None = None,
) -> Dict[str, Dict[str, int]]:
    scoped_request_ids = list(request_ids or [])
    with conn.cursor() as cursor:
        cursor.execute(
            """
SELECT
  re.objective_effective,
  COUNT(DISTINCT re.request_id) AS request_count,
  SUM(
    CASE
      WHEN oe.window_hours = 24 AND oe.matured = TRUE THEN 1
      ELSE 0
    END
  ) AS matured_24h_count,
  SUM(
    CASE
      WHEN oe.window_hours = 96 AND oe.matured = TRUE THEN 1
      ELSE 0
    END
  ) AS matured_96h_count
  ,
  COALESCE(SUM(COALESCE(uife.strong_feedback_count, 0)), 0) AS strong_feedback_count
  ,
  COALESCE(SUM(COALESCE(uife.explicit_positive_count, 0)), 0) AS explicit_positive_count
  ,
  COALESCE(SUM(COALESCE(uife.explicit_negative_count, 0)), 0) AS explicit_negative_count
  ,
  COALESCE(SUM(COALESCE(uife.has_explicit_positive, 0)), 0) AS explicit_positive_request_count
  ,
  COALESCE(SUM(COALESCE(uife.has_explicit_negative, 0)), 0) AS explicit_negative_request_count
  ,
  COALESCE(SUM(COALESCE(uife.has_no_good_options, 0)), 0) AS no_good_option_request_count
FROM rec_request_events re
LEFT JOIN rec_outcome_events oe
  ON oe.request_id = re.request_id
LEFT JOIN (
  SELECT
    request_id,
    SUM(
      CASE
        WHEN signal_strength = 'strong'
         AND event_name IN (
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
    ) AS strong_feedback_count,
    SUM(
      CASE
        WHEN signal_strength = 'strong'
         AND event_name IN (
           'comparable_marked_relevant',
           'comparable_saved',
           'recommendation_marked_useful',
           'recommendation_saved'
         )
        THEN 1
        ELSE 0
      END
    ) AS explicit_positive_count,
    SUM(
      CASE
        WHEN signal_strength = 'strong'
         AND event_name IN (
           'comparable_marked_not_relevant',
           'comparable_no_good_options',
           'recommendation_marked_not_useful'
         )
        THEN 1
        ELSE 0
      END
    ) AS explicit_negative_count,
    MAX(
      CASE
        WHEN signal_strength = 'strong'
         AND event_name IN (
           'comparable_marked_relevant',
           'comparable_saved',
           'recommendation_marked_useful',
           'recommendation_saved'
         )
        THEN 1
        ELSE 0
      END
    ) AS has_explicit_positive,
    MAX(
      CASE
        WHEN signal_strength = 'strong'
         AND event_name IN (
           'comparable_marked_not_relevant',
           'recommendation_marked_not_useful'
         )
        THEN 1
        ELSE 0
      END
    ) AS has_explicit_negative,
    MAX(
      CASE
        WHEN signal_strength = 'strong'
         AND event_name = 'comparable_no_good_options'
        THEN 1
        ELSE 0
      END
    ) AS has_no_good_options
  FROM rec_ui_feedback_events
  GROUP BY request_id
) uife
  ON uife.request_id = re.request_id
WHERE re.served_at >= %s
  AND re.served_at < %s
  AND (%s OR COALESCE(re.is_synthetic, FALSE) = FALSE)
  AND (%s OR COALESCE(re.injected_failure, FALSE) = FALSE)
  AND (
    COALESCE(array_length(%s::uuid[], 1), 0) = 0
    OR re.request_id = ANY(%s::uuid[])
  )
GROUP BY re.objective_effective
ORDER BY re.objective_effective ASC
"""
            ,
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
    out: Dict[str, Dict[str, int]] = {}
    for row in rows:
        objective = str(row[0] or "").strip()
        if not objective:
            continue
        out[objective] = {
            "request_count": int(row[1] or 0),
            "matured_24h_count": int(row[2] or 0),
            "matured_96h_count": int(row[3] or 0),
            "strong_feedback_count": int(row[4] or 0) if len(row) > 4 else 0,
            "explicit_positive_count": int(row[5] or 0) if len(row) > 5 else 0,
            "explicit_negative_count": int(row[6] or 0) if len(row) > 6 else 0,
            "explicit_positive_request_count": int(row[7] or 0) if len(row) > 7 else 0,
            "explicit_negative_request_count": int(row[8] or 0) if len(row) > 8 else 0,
            "no_good_option_request_count": int(row[9] or 0) if len(row) > 9 else 0,
        }
    return out


def _apply_drift_trigger_support_gate(
    *,
    trigger: bool,
    reason: str,
    support: Dict[str, int],
    min_matured_24h: int,
    min_matured_96h: int,
) -> Tuple[bool, str, bool]:
    strong_feedback_count = int(support.get("strong_feedback_count") or 0)
    contrast_keys_present = any(
        key in support
        for key in (
            "explicit_positive_request_count",
            "explicit_negative_request_count",
            "no_good_option_request_count",
        )
    )
    contrast_ok = True
    if contrast_keys_present:
        explicit_positive_request_count = int(support.get("explicit_positive_request_count") or 0)
        explicit_negative_request_count = int(support.get("explicit_negative_request_count") or 0)
        no_good_option_request_count = int(support.get("no_good_option_request_count") or 0)
        contrast_ok = explicit_positive_request_count >= 1 and (
            explicit_negative_request_count >= 1 or no_good_option_request_count >= 1
        )
    strong_feedback_ok = strong_feedback_count >= max(1, int(min_matured_24h)) and contrast_ok
    delayed_outcome_ok = (
        int(support.get("matured_24h_count") or 0) >= max(1, int(min_matured_24h))
        and int(support.get("matured_96h_count") or 0) >= max(1, int(min_matured_96h))
    )
    support_ok = strong_feedback_ok or delayed_outcome_ok
    if bool(trigger) and str(reason) == "drift_trigger" and not support_ok:
        return False, "insufficient_feedback_support", False
    return bool(trigger), str(reason), bool(support_ok)


def _insert_retrain_run(conn, payload: Dict[str, Any]) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
INSERT INTO rec_retrain_runs (
  run_id, trigger_source, status, selected_bundle_id, previous_bundle_id, drift_evidence, decision_payload
) VALUES (
  %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb
)
ON CONFLICT (run_id) DO UPDATE SET
  status = EXCLUDED.status,
  selected_bundle_id = EXCLUDED.selected_bundle_id,
  previous_bundle_id = EXCLUDED.previous_bundle_id,
  drift_evidence = EXCLUDED.drift_evidence,
  decision_payload = EXCLUDED.decision_payload,
  created_at = NOW()
""",
            (
                payload["run_id"],
                payload["trigger_source"],
                payload["status"],
                payload.get("selected_bundle_id"),
                payload.get("previous_bundle_id"),
                json.dumps(payload.get("drift_evidence") or {}, ensure_ascii=False),
                json.dumps(payload.get("decision_payload") or {}, ensure_ascii=False),
            ),
        )


def _run_command(command: List[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _load_drift_observations_from_report(path: Path) -> Dict[str, List[Dict[str, str]]]:
    payload = _read_json(path)
    objectives = payload.get("objectives")
    if not isinstance(objectives, dict):
        return {}
    out: Dict[str, List[Dict[str, str]]] = {}
    for objective, summary in objectives.items():
        if not isinstance(summary, dict):
            continue
        severity = str(summary.get("severity") or "ok").strip().lower()
        trigger_recommendation = str(summary.get("trigger_recommendation") or "none").strip().lower()
        out[str(objective)] = [
            {
                "severity": severity or "ok",
                "trigger_recommendation": trigger_recommendation or "none",
            }
        ]
    return out


def _resolve_trigger_source(
    *,
    trigger_any: bool,
    scheduled_due: bool,
    objective_triggers: Dict[str, Dict[str, Any]],
) -> str:
    if any(
        bool(item.get("trigger")) and str(item.get("reason") or "") == "drift_trigger"
        for item in objective_triggers.values()
    ):
        return "drift_trigger"
    if any(
        bool(item.get("trigger")) and str(item.get("reason") or "") == "scheduled_weekly"
        for item in objective_triggers.values()
    ):
        return "scheduled_weekly"
    if bool(scheduled_due):
        return "scheduled_weekly"
    if bool(trigger_any):
        return "drift_trigger"
    return "no_trigger"


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
    parser = argparse.ArgumentParser(description="Run retrain trigger controller.")
    parser.add_argument("--db-url", type=str, default=None, help="Optional Postgres URL override.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/recommender"),
        help="Recommender artifact root.",
    )
    parser.add_argument(
        "--datamart-json",
        type=Path,
        default=None,
        help="Optional prebuilt datamart JSON path.",
    )
    parser.add_argument(
        "--execute-train",
        action="store_true",
        help="Actually run retraining commands. Otherwise dry-run.",
    )
    parser.add_argument(
        "--scheduled-weekday",
        type=int,
        default=0,
        help="Weekly scheduled weekday in UTC, Monday=0..Sunday=6.",
    )
    parser.add_argument(
        "--force-scheduled",
        action="store_true",
        help="Force scheduled path even if weekday does not match.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/control_plane/retrain_decision.json"),
        help="Decision output path.",
    )
    parser.add_argument(
        "--outcome-lookback-days",
        type=int,
        default=14,
        help="Recent outcome lookback window used for drift-trigger support gating.",
    )
    parser.add_argument(
        "--min-matured-24h",
        type=int,
        default=20,
        help="Minimum matured 24h outcomes required for drift-trigger retrain.",
    )
    parser.add_argument(
        "--min-matured-96h",
        type=int,
        default=10,
        help="Minimum matured 96h outcomes required for drift-trigger retrain.",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Include synthetic traffic when evaluating retrain support evidence.",
    )
    parser.add_argument(
        "--include-injected-failures",
        action="store_true",
        help="Include injected-failure traffic when evaluating retrain support evidence.",
    )
    parser.add_argument(
        "--drift-report-json",
        type=Path,
        default=None,
        help="Optional drift report JSON. Required when request-id scope is set.",
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
        help="Optional JSON file containing request IDs (list or {'request_ids': [...]}) to scope retrain support.",
    )
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    scoped_request_ids = _collect_request_id_scope(args.request_id, args.request_ids_json)
    scheduled_due = bool(args.force_scheduled or now.weekday() == int(args.scheduled_weekday))
    run_id = f"retrain-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    decision_payload: Dict[str, Any] = {
        "run_id": run_id,
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "scheduled_due": scheduled_due,
        "dry_run": not args.execute_train,
        "support_thresholds": {
            "outcome_lookback_days": int(max(1, args.outcome_lookback_days)),
            "min_matured_24h": int(max(1, args.min_matured_24h)),
            "min_matured_96h": int(max(1, args.min_matured_96h)),
            "include_synthetic": bool(args.include_synthetic),
            "include_injected_failures": bool(args.include_injected_failures),
        },
        "request_id_scope_count": len(scoped_request_ids),
    }

    with get_connection(args.db_url) as conn:
        _ensure_ui_feedback_events_table(conn)
        if scoped_request_ids and args.drift_report_json is None:
            raise ValueError("request-id scope requires --drift-report-json for scoped drift evidence.")
        if args.drift_report_json is not None:
            if not args.drift_report_json.exists():
                raise ValueError(f"drift report JSON not found: {args.drift_report_json}")
            drift_by_objective = _load_drift_observations_from_report(args.drift_report_json)
            decision_payload["drift_evidence_source"] = "report_json"
        else:
            drift_by_objective = _fetch_recent_drift(conn)
            decision_payload["drift_evidence_source"] = "rec_drift_daily"
        support_since = now - timedelta(days=max(1, int(args.outcome_lookback_days)))
        support_by_objective = _fetch_recent_outcome_support(
            conn,
            since=support_since,
            until=now,
            include_synthetic=bool(args.include_synthetic),
            include_injected_failures=bool(args.include_injected_failures),
            request_ids=scoped_request_ids,
        )
        decision_payload["objective_support"] = support_by_objective
        objective_triggers: Dict[str, Dict[str, Any]] = {}
        trigger_any = False
        objective_keys = sorted(set(drift_by_objective.keys()) | set(support_by_objective.keys()))
        for objective in objective_keys:
            observations = list(drift_by_objective.get(objective) or [])
            severities = [str(item.get("severity") or "ok") for item in observations]
            recommendations = [
                str(item.get("trigger_recommendation") or "none")
                for item in observations
            ]
            trigger, reason = should_trigger_retrain(
                recent_severities=severities[:5],
                scheduled_due=scheduled_due,
                consecutive_critical_required=2,
                recent_trigger_recommendations=recommendations[:5],
            )
            support = support_by_objective.get(
                objective,
                {
                    "request_count": 0,
                    "matured_24h_count": 0,
                    "matured_96h_count": 0,
                },
            )
            trigger, reason, support_ok = _apply_drift_trigger_support_gate(
                trigger=trigger,
                reason=reason,
                support=support,
                min_matured_24h=int(max(1, args.min_matured_24h)),
                min_matured_96h=int(max(1, args.min_matured_96h)),
            )
            objective_triggers[objective] = {
                "recent_severities": severities[:5],
                "recent_trigger_recommendations": recommendations[:5],
                "support": support,
                "support_sufficient_for_drift_trigger": bool(support_ok),
                "trigger": trigger,
                "reason": reason,
            }
            if trigger:
                trigger_any = True
        trigger_source = _resolve_trigger_source(
            trigger_any=trigger_any,
            scheduled_due=scheduled_due,
            objective_triggers=objective_triggers,
        )

        decision_payload["objective_triggers"] = objective_triggers
        decision_payload["trigger_any"] = trigger_any or scheduled_due
        decision_payload["trigger_source"] = trigger_source

        previous_bundle_ref = args.artifact_root / "latest"
        previous_bundle_dir = (
            Path(previous_bundle_ref.read_text(encoding="utf-8").strip())
            if previous_bundle_ref.is_file() and not previous_bundle_ref.is_dir()
            else previous_bundle_ref.resolve()
            if previous_bundle_ref.exists()
            else None
        )
        baseline_metrics = (
            _load_metrics(previous_bundle_dir) if isinstance(previous_bundle_dir, Path) and previous_bundle_dir.exists() else {}
        )
        decision_payload["previous_bundle_dir"] = str(previous_bundle_dir) if previous_bundle_dir else None

        selected_bundle_id: Optional[str] = None
        status = "skipped"
        gate_diag: Dict[str, Any] = {}
        if decision_payload["trigger_any"] and args.execute_train:
            if args.datamart_json is None or not args.datamart_json.exists():
                raise ValueError("execute-train requires --datamart-json path.")
            run_name = f"retrain-{now.strftime('%Y%m%d-%H%M%S')}"
            output = _run_command(
                [
                    "python3",
                    "scripts/train_recommender.py",
                    str(args.datamart_json),
                    "--artifact-root",
                    str(args.artifact_root),
                    "--run-name",
                    run_name,
                ]
            )
            parsed = json.loads(output)
            candidate_bundle = Path(str(parsed["bundle_dir"]))
            candidate_metrics = _load_metrics(candidate_bundle)
            passed, gate_diag = _non_regression_gate(
                baseline_metrics=baseline_metrics,
                candidate_metrics=candidate_metrics,
                threshold_ratio=0.995,
            )
            if passed:
                status = "promoted"
                selected_bundle_id = candidate_bundle.name
            else:
                status = "rejected_non_regression"
                selected_bundle_id = previous_bundle_dir.name if isinstance(previous_bundle_dir, Path) else None
        elif decision_payload["trigger_any"]:
            status = "dry_run_ready"

        decision = build_retrain_decision(
            trigger_source=trigger_source,
            selected_bundle_id=selected_bundle_id,
            previous_bundle_id=previous_bundle_dir.name if isinstance(previous_bundle_dir, Path) else None,
            promoted=status == "promoted",
            objective_metrics=gate_diag,
            drift_evidence=objective_triggers,
        )
        decision_payload["status"] = status
        decision_payload["decision"] = decision
        _insert_retrain_run(
            conn,
            {
                "run_id": run_id,
                "trigger_source": trigger_source,
                "status": status,
                "selected_bundle_id": decision.get("selected_bundle_id"),
                "previous_bundle_id": decision.get("previous_bundle_id"),
                "drift_evidence": objective_triggers,
                "decision_payload": decision_payload,
            },
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(decision_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
