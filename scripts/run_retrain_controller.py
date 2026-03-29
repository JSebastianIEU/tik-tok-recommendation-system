#!/usr/bin/env python3
"""Trigger scheduled/drift retrain decisions and persist rec_retrain_runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
) -> Tuple[bool, Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {}
    ok = True
    for objective, baseline in baseline_metrics.items():
        baseline_ndcg = float((baseline or {}).get("ranker", {}).get("ndcg@10") or 0.0)
        baseline_mrr = float((baseline or {}).get("ranker", {}).get("mrr@20") or 0.0)
        candidate = candidate_metrics.get(objective, {})
        candidate_ndcg = float((candidate or {}).get("ranker", {}).get("ndcg@10") or 0.0)
        candidate_mrr = float((candidate or {}).get("ranker", {}).get("mrr@20") or 0.0)
        objective_ok = (
            candidate_ndcg >= baseline_ndcg * threshold_ratio
            and candidate_mrr >= baseline_mrr * threshold_ratio
        )
        diagnostics[objective] = {
            "baseline_ndcg@10": baseline_ndcg,
            "baseline_mrr@20": baseline_mrr,
            "candidate_ndcg@10": candidate_ndcg,
            "candidate_mrr@20": candidate_mrr,
            "passed": objective_ok,
        }
        if not objective_ok:
            ok = False
    return ok, diagnostics


def _fetch_recent_drift(conn) -> Dict[str, List[str]]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
SELECT objective_effective, breach_severity
FROM rec_drift_daily
WHERE segment_id = 'global'
ORDER BY drift_date DESC
LIMIT 60
"""
        )
        rows = cursor.fetchall()
    by_objective: Dict[str, List[str]] = {}
    for row in rows:
        objective = str(row[0])
        severity = str(row[1] or "ok")
        by_objective.setdefault(objective, []).append(severity)
    return by_objective


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
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    scheduled_due = bool(args.force_scheduled or now.weekday() == int(args.scheduled_weekday))
    run_id = f"retrain-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    decision_payload: Dict[str, Any] = {
        "run_id": run_id,
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "scheduled_due": scheduled_due,
        "dry_run": not args.execute_train,
    }

    with get_connection(args.db_url) as conn:
        drift_by_objective = _fetch_recent_drift(conn)
        objective_triggers: Dict[str, Dict[str, Any]] = {}
        trigger_any = False
        trigger_source = "no_trigger"
        for objective, severities in drift_by_objective.items():
            trigger, reason = should_trigger_retrain(
                recent_severities=severities[:5],
                scheduled_due=scheduled_due,
                consecutive_critical_required=2,
            )
            objective_triggers[objective] = {
                "recent_severities": severities[:5],
                "trigger": trigger,
                "reason": reason,
            }
            if trigger:
                trigger_any = True
                if reason == "drift_trigger":
                    trigger_source = "drift_trigger"
        if not trigger_any and scheduled_due:
            trigger_source = "scheduled_weekly"

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

