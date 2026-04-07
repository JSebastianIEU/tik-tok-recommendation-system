#!/usr/bin/env python3
"""Run live end-to-end recommendation validation with optional failure injection."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

RUNTIME_OBJECTIVES = ("reach", "engagement", "conversion")
EXPERIMENT_VARIANTS = ("control", "treatment")


def build_validation_request_plan(request_count: int) -> List[Dict[str, str]]:
    total = max(1, int(request_count))
    plan: List[Dict[str, str]] = []
    for idx in range(total):
        objective = RUNTIME_OBJECTIVES[(idx // len(EXPERIMENT_VARIANTS)) % len(RUNTIME_OBJECTIVES)]
        variant = EXPERIMENT_VARIANTS[idx % len(EXPERIMENT_VARIANTS)]
        plan.append(
            {
                "objective": str(objective),
                "force_variant": str(variant),
            }
        )
    return plan


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_dumps(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def http_get_json(url: str, timeout_sec: float = 3.0) -> Dict[str, Any]:
    req = urllib_request.Request(url=url, method="GET")
    with urllib_request.urlopen(req, timeout=max(0.1, float(timeout_sec))) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object from {url}, got {type(parsed).__name__}.")
    return parsed


def http_post_json(url: str, payload: Dict[str, Any], timeout_sec: float = 8.0) -> Tuple[int, Dict[str, Any]]:
    body = _json_dumps(payload)
    req = urllib_request.Request(
        url=url,
        method="POST",
        data=body,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib_request.urlopen(req, timeout=max(0.1, float(timeout_sec))) as response:
            status = int(response.getcode() or 200)
            raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            return status, parsed if isinstance(parsed, dict) else {"raw": parsed}
    except urllib_error.HTTPError as exc:
        raw = exc.read().decode("utf-8") if exc.fp is not None else ""
        try:
            parsed = json.loads(raw) if raw else {}
        except Exception:
            parsed = {"raw_error": raw}
        if isinstance(parsed, dict):
            return int(exc.code), parsed
        return int(exc.code), {"raw_error": raw}


def wait_for_json_endpoint(
    url: str,
    *,
    timeout_sec: float = 45.0,
    interval_sec: float = 0.5,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    start = time.time()
    last_error: Optional[str] = None
    while time.time() - start <= max(1.0, timeout_sec):
        try:
            payload = http_get_json(url, timeout_sec=min(3.0, timeout_sec))
            return True, payload, None
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(max(0.1, interval_sec))
    return False, None, last_error


def validate_recommendation_payload(payload: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "payload_not_object"
    items = payload.get("items")
    if not isinstance(items, list):
        return False, "items_missing"
    if len(items) == 0:
        return True, "ok_empty_items"
    first = items[0]
    if not isinstance(first, dict):
        return False, "first_item_not_object"
    candidate_id = str(first.get("candidate_id") or "").strip()
    if not candidate_id:
        return False, "candidate_id_missing"
    try:
        float(first.get("score"))
    except (TypeError, ValueError):
        return False, "score_missing"
    try:
        int(first.get("rank"))
    except (TypeError, ValueError):
        return False, "rank_missing"
    return True, "ok"


def _resolve_bundle_dir(bundle_ref: Path) -> Path:
    if bundle_ref.is_file():
        target = bundle_ref.read_text(encoding="utf-8").strip()
        return Path(target).resolve()
    return bundle_ref.resolve()


def _bundle_candidate_ids(bundle_ref: Path, *, limit: int = 500) -> List[str]:
    bundle_dir = _resolve_bundle_dir(bundle_ref)
    retriever_manifest = bundle_dir / "retriever" / "manifest.json"
    if not retriever_manifest.exists():
        return []
    payload = json.loads(retriever_manifest.read_text(encoding="utf-8"))
    row_metadata = payload.get("row_metadata")
    if not isinstance(row_metadata, dict):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for _, meta in row_metadata.items():
        if not isinstance(meta, dict):
            continue
        video_id = str(meta.get("video_id") or "").strip()
        if not video_id or video_id in seen:
            continue
        seen.add(video_id)
        out.append(video_id)
        if len(out) >= max(1, int(limit)):
            break
    return out


def build_recommendation_payload(
    index: int,
    *,
    compat_override: Optional[Dict[str, str]] = None,
    candidate_ids: Optional[List[str]] = None,
    injected_failure: bool = False,
    objective_override: Optional[str] = None,
    force_variant: Optional[str] = None,
) -> Dict[str, Any]:
    objective = str(objective_override or RUNTIME_OBJECTIVES[index % len(RUNTIME_OBJECTIVES)])
    variant = str(force_variant or "").strip().lower()
    description = (
        f"E2E validation sample {index} quick tutorial growth tips {objective} "
        f"{variant if variant in EXPERIMENT_VARIANTS else 'auto'}"
    )
    hashtags = [f"#growth{index % 4}", "#tips", "#shortform"]
    experiment_payload: Dict[str, Any] = {"id": "e2e_live_validation"}
    if variant in EXPERIMENT_VARIANTS:
        experiment_payload["force_variant"] = variant
    base: Dict[str, Any] = {
        "description": description,
        "hashtags": hashtags,
        "mentions": [],
        "as_of_time": "2026-01-15T00:00:00Z",
        "objective": objective,
        "audience": "general",
        "content_type": "tutorial",
        "primary_cta": "save",
        "locale": "en-US",
        "language": "en",
        "top_k": 10,
        "retrieve_k": 120,
        "debug": True,
        "trajectory_controls": {"enabled": True},
        "graph_controls": {"enable_graph_branch": True},
        "explainability": {
            "enabled": True,
            "top_features": 4,
            "neighbor_k": 2,
            "run_counterfactuals": False,
        },
        "candidate_ids": list(candidate_ids or []),
        "portfolio": {
            "enabled": True,
            "weights": {"reach": 0.45, "conversion": 0.35, "durability": 0.20},
            "risk_aversion": 0.10,
            "candidate_pool_cap": 120,
        },
        "routing": {
            "track": "post_publication",
            "allow_fallback": True,
            "required_compat": compat_override or {},
        },
        "policy_overrides": {"max_items_per_author": 2},
        "experiment": experiment_payload,
        "traffic": {
            "class": "synthetic",
            "injected_failure": bool(injected_failure),
        },
    }
    return base


@dataclass
class ManagedProcess:
    name: str
    command: List[str]
    cwd: Path
    env: Dict[str, str]
    process: subprocess.Popen[str]
    log_path: Path
    _log_fp: Any


def spawn_process(
    *,
    name: str,
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    logs_dir: Path,
) -> ManagedProcess:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.log"
    log_fp = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return ManagedProcess(
        name=name,
        command=command,
        cwd=cwd,
        env=env,
        process=process,
        log_path=log_path,
        _log_fp=log_fp,
    )


def stop_process(proc: ManagedProcess, *, grace_sec: float = 6.0) -> None:
    if proc.process.poll() is not None:
        try:
            proc._log_fp.close()
        except Exception:
            pass
        return
    try:
        proc.process.terminate()
        deadline = time.time() + max(0.5, grace_sec)
        while time.time() < deadline:
            if proc.process.poll() is not None:
                break
            time.sleep(0.1)
        if proc.process.poll() is None:
            proc.process.kill()
    finally:
        try:
            proc._log_fp.close()
        except Exception:
            pass


def run_job_command(command: List[str], *, timeout_sec: float = 240.0) -> Tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_sec)),
        )
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    output = (completed.stdout or "").strip()
    err = (completed.stderr or "").strip()
    if completed.returncode != 0:
        return False, f"returncode={completed.returncode}; stderr={err}; stdout={output}"
    return True, output


def fetch_feedback_db_counts(db_url: str, request_ids: List[str]) -> Dict[str, int]:
    if not request_ids:
        return {"request_events": 0, "candidate_events": 0, "served_outputs": 0}
    try:
        from scraper.db.client import get_connection
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from scraper.db.client import get_connection
    with get_connection(db_url) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM rec_request_events WHERE request_id = ANY(%s::uuid[])",
                (request_ids,),
            )
            request_count = int(cursor.fetchone()[0] or 0)
            cursor.execute(
                "SELECT COUNT(*) FROM rec_candidate_events WHERE request_id = ANY(%s::uuid[])",
                (request_ids,),
            )
            candidate_count = int(cursor.fetchone()[0] or 0)
            cursor.execute(
                "SELECT COUNT(*) FROM rec_served_outputs WHERE request_id = ANY(%s::uuid[])",
                (request_ids,),
            )
            served_count = int(cursor.fetchone()[0] or 0)
    return {
        "request_events": request_count,
        "candidate_events": candidate_count,
        "served_outputs": served_count,
    }


def _write_request_ids_scope(path: Path, request_ids: List[str]) -> Optional[Path]:
    normalized = [str(item).strip() for item in request_ids if str(item).strip()]
    if not normalized:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"request_ids": normalized}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_control_plane_jobs(
    *,
    db_url: str,
    cp_dir: Path,
    include_injected_failures: bool,
    scope_args: Optional[List[str]] = None,
) -> Dict[str, Tuple[List[str], Path]]:
    scoped_args = list(scope_args or [])
    injected_args = ["--include-injected-failures"] if bool(include_injected_failures) else []
    return {
        "outcome_attribution": (
            [
                "python3",
                "scripts/run_outcome_attribution.py",
                "--db-url",
                str(db_url),
                "--lookback-hours",
                "240",
                "--output-json",
                str(cp_dir / "outcome_attribution_report.json"),
                *scoped_args,
                "--include-synthetic",
                *injected_args,
            ],
            cp_dir / "outcome_attribution_report.json",
        ),
        "drift_monitor": (
            [
                "python3",
                "scripts/run_drift_monitor.py",
                "--db-url",
                str(db_url),
                "--output-json",
                str(cp_dir / "drift_report.json"),
                *scoped_args,
                "--include-synthetic",
                *injected_args,
            ],
            cp_dir / "drift_report.json",
        ),
        "experiment_analysis": (
            [
                "python3",
                "scripts/run_experiment_analysis.py",
                "--db-url",
                str(db_url),
                "--lookback-days",
                "14",
                "--output-json",
                str(cp_dir / "experiment_report.json"),
                *scoped_args,
                "--include-synthetic",
                *injected_args,
            ],
            cp_dir / "experiment_report.json",
        ),
        "retrain_controller": (
            [
                "python3",
                "scripts/run_retrain_controller.py",
                "--db-url",
                str(db_url),
                "--output-json",
                str(cp_dir / "retrain_decision.json"),
                "--drift-report-json",
                str(cp_dir / "drift_report.json"),
                *scoped_args,
                "--include-synthetic",
                *injected_args,
            ],
            cp_dir / "retrain_decision.json",
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live E2E recommender flow validation.")
    parser.add_argument("--node-url", type=str, default="http://127.0.0.1:5174")
    parser.add_argument("--python-url", type=str, default="http://127.0.0.1:8081")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--frontend-dir", type=Path, default=Path("frontend"))
    parser.add_argument("--bundle-dir", type=Path, default=Path("artifacts/recommender/latest"))
    parser.add_argument("--db-url", type=str, default="")
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--request-timeout-sec", type=float, default=12.0)
    parser.add_argument("--launch-python", action="store_true")
    parser.add_argument("--launch-node", action="store_true")
    parser.add_argument("--python-port", type=int, default=8081)
    parser.add_argument("--node-port", type=int, default=5174)
    parser.add_argument("--inject-compat-mismatch", action="store_true")
    parser.add_argument("--inject-python-down", action="store_true")
    parser.add_argument("--run-control-plane", action="store_true")
    parser.add_argument(
        "--control-plane-include-injected-failures",
        action="store_true",
        help="When running control-plane jobs, include injected failure traffic.",
    )
    parser.add_argument(
        "--control-plane-scope",
        type=str,
        default="current_run",
        choices=("current_run", "lookback"),
        help="Control-plane job scope: current run request IDs or standard lookback windows.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/control_plane/live_e2e_validation_report.json"),
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    frontend_dir = (repo_root / args.frontend_dir).resolve()
    logs_dir = (repo_root / "artifacts" / "control_plane" / "live_e2e_logs").resolve()
    output_json = args.output_json if args.output_json.is_absolute() else (repo_root / args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    checks: List[Dict[str, Any]] = []
    started_processes: List[ManagedProcess] = []
    request_ids: List[str] = []

    report: Dict[str, Any] = {
        "generated_at": _utc_now_iso(),
        "node_url": args.node_url,
        "python_url": args.python_url,
        "launch_python": bool(args.launch_python),
        "launch_node": bool(args.launch_node),
        "run_control_plane": bool(args.run_control_plane),
        "control_plane_include_injected_failures": bool(
            args.control_plane_include_injected_failures
        ),
        "control_plane_scope": str(args.control_plane_scope),
        "checks": checks,
        "requests": [],
        "failure_injection": {},
        "feedback_db_counts": None,
        "job_reports": {},
        "logs": {},
    }

    def add_check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    try:
        bundle_candidate_ids = _bundle_candidate_ids(args.bundle_dir, limit=500)
        report["candidate_ids_source"] = {
            "count": len(bundle_candidate_ids),
            "mode": "retriever_manifest_video_ids",
        }
        if args.launch_python:
            python_env = dict(os.environ)
            python_env["PYTHONPATH"] = str(repo_root)
            python_proc = spawn_process(
                name="python_service",
                command=[
                    "python3",
                    "scripts/serve_recommender.py",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(int(args.python_port)),
                    "--bundle-dir",
                    str(args.bundle_dir),
                ],
                cwd=repo_root,
                env=python_env,
                logs_dir=logs_dir,
            )
            started_processes.append(python_proc)
            report["logs"]["python_service"] = str(python_proc.log_path)
            add_check("python_service_spawned", True, {"pid": python_proc.process.pid})

        if args.launch_node:
            node_env = dict(os.environ)
            node_env["PORT"] = str(int(args.node_port))
            node_env["RECOMMENDER_BASE_URL"] = args.python_url
            node_env["RECOMMENDER_ENABLED"] = "true"
            # Local live validation needs a realistic compatibility/network timeout budget.
            node_env["RECOMMENDER_BUDGET_NETWORK_MS"] = node_env.get(
                "RECOMMENDER_BUDGET_NETWORK_MS",
                "1200",
            )
            node_env["RECOMMENDER_COMPAT_CACHE_TTL_MS"] = node_env.get(
                "RECOMMENDER_COMPAT_CACHE_TTL_MS",
                "1000",
            )
            node_env["RECOMMENDER_COMPAT_CHECK_INTERVAL_MS"] = node_env.get(
                "RECOMMENDER_COMPAT_CHECK_INTERVAL_MS",
                "5000",
            )
            if args.db_url.strip():
                node_env["RECOMMENDER_FEEDBACK_ENABLED"] = "true"
                node_env["RECOMMENDER_FEEDBACK_DB_URL"] = args.db_url.strip()
            else:
                node_env["RECOMMENDER_FEEDBACK_ENABLED"] = "false"
                node_env["RECOMMENDER_FEEDBACK_DB_URL"] = ""
            node_proc = spawn_process(
                name="node_gateway",
                command=["npm", "run", "server"],
                cwd=frontend_dir,
                env=node_env,
                logs_dir=logs_dir,
            )
            started_processes.append(node_proc)
            report["logs"]["node_gateway"] = str(node_proc.log_path)
            add_check("node_gateway_spawned", True, {"pid": node_proc.process.pid})

        python_ready, python_payload, python_error = wait_for_json_endpoint(
            f"{args.python_url.rstrip('/')}/v1/health",
            timeout_sec=60.0 if args.launch_python else 15.0,
        )
        add_check(
            "python_health_ready",
            python_ready,
            python_payload if python_ready else {"error": python_error},
        )

        node_ready, node_payload, node_error = wait_for_json_endpoint(
            f"{args.node_url.rstrip('/')}/recommender-gateway-metrics",
            timeout_sec=70.0 if args.launch_node else 20.0,
        )
        add_check(
            "node_gateway_ready",
            node_ready,
            node_payload if node_ready else {"error": node_error},
        )

        if not python_ready or not node_ready:
            raise RuntimeError("Required endpoints are not ready for live E2E validation.")

        # Allow any early warmup compatibility failures to expire before request assertions.
        time.sleep(1.5)

        request_plan = build_validation_request_plan(max(1, int(args.requests)))
        report["request_plan"] = request_plan
        for idx, plan_item in enumerate(request_plan):
            payload = build_recommendation_payload(
                idx,
                candidate_ids=bundle_candidate_ids,
                objective_override=str(plan_item.get("objective") or ""),
                force_variant=str(plan_item.get("force_variant") or ""),
            )
            status, response_payload = http_post_json(
                f"{args.node_url.rstrip('/')}/recommendations",
                payload,
                timeout_sec=float(args.request_timeout_sec),
            )
            valid, reason = validate_recommendation_payload(response_payload)
            request_id = str(response_payload.get("request_id") or "").strip()
            if request_id:
                request_ids.append(request_id)
            report["requests"].append(
                {
                    "index": idx,
                    "status": status,
                    "valid": valid,
                    "reason": reason,
                    "fallback_mode": bool(response_payload.get("fallback_mode", False)),
                    "request_id": request_id or None,
                    "objective_effective": response_payload.get("objective_effective"),
                    "variant": response_payload.get("variant"),
                    "items_count": len(response_payload.get("items") or []),
                }
            )

        valid_count = sum(1 for row in report["requests"] if row["valid"])
        non_fallback_count = sum(1 for row in report["requests"] if not row["fallback_mode"])
        add_check(
            "recommendation_requests_valid",
            valid_count == len(report["requests"]),
            {"valid_count": valid_count, "total": len(report["requests"])},
        )
        add_check(
            "recommender_upstream_served",
            non_fallback_count > 0,
            {
                "non_fallback_count": non_fallback_count,
                "total": len(report["requests"]),
                "note": "At least one live request should be served by Python recommender, not fallback.",
            },
        )

        if args.inject_compat_mismatch:
            mismatch_payload = build_recommendation_payload(
                999,
                compat_override={"feature_schema_hash": "mismatch-e2e-live"},
                candidate_ids=bundle_candidate_ids,
                injected_failure=True,
            )
            mismatch_status, mismatch_response = http_post_json(
                f"{args.node_url.rstrip('/')}/recommendations",
                mismatch_payload,
                timeout_sec=float(args.request_timeout_sec),
            )
            mismatch_fallback = bool(mismatch_response.get("fallback_mode", False))
            report["failure_injection"]["compat_mismatch"] = {
                "status": mismatch_status,
                "fallback_mode": mismatch_fallback,
                "fallback_reason": mismatch_response.get("fallback_reason"),
                "request_id": mismatch_response.get("request_id"),
            }
            add_check(
                "failure_injection_compat_mismatch",
                mismatch_status == 200 and mismatch_fallback,
                report["failure_injection"]["compat_mismatch"],
            )

        if args.inject_python_down:
            python_proc = next((proc for proc in started_processes if proc.name == "python_service"), None)
            if python_proc is None:
                add_check(
                    "failure_injection_python_down",
                    False,
                    {"error": "inject-python-down requires --launch-python"},
                )
            else:
                stop_process(python_proc)
                time.sleep(1.0)
                down_payload = build_recommendation_payload(1001, injected_failure=True)
                down_payload["candidate_ids"] = list(bundle_candidate_ids)
                down_status, down_response = http_post_json(
                    f"{args.node_url.rstrip('/')}/recommendations",
                    down_payload,
                    timeout_sec=float(args.request_timeout_sec),
                )
                down_fallback = bool(down_response.get("fallback_mode", False))
                report["failure_injection"]["python_down"] = {
                    "status": down_status,
                    "fallback_mode": down_fallback,
                    "fallback_reason": down_response.get("fallback_reason"),
                    "request_id": down_response.get("request_id"),
                }
                add_check(
                    "failure_injection_python_down",
                    down_status == 200 and down_fallback,
                    report["failure_injection"]["python_down"],
                )

        if args.db_url.strip():
            db_counts = fetch_feedback_db_counts(args.db_url.strip(), request_ids=request_ids)
            report["feedback_db_counts"] = db_counts
            add_check(
                "feedback_events_persisted",
                db_counts["request_events"] >= max(1, len(request_ids))
                and db_counts["served_outputs"] >= max(1, len(request_ids)),
                db_counts,
            )
        else:
            add_check("feedback_events_persisted", True, {"skipped": "db_url_not_provided"})

        if args.run_control_plane:
            if not args.db_url.strip():
                add_check("control_plane_jobs", False, {"error": "--run-control-plane requires --db-url"})
            else:
                cp_dir = (repo_root / "artifacts" / "control_plane" / "live_e2e").resolve()
                cp_dir.mkdir(parents=True, exist_ok=True)
                scoped_ids_path: Optional[Path] = None
                if str(args.control_plane_scope) == "current_run":
                    scoped_ids_path = _write_request_ids_scope(
                        cp_dir / "request_ids_scope.json",
                        request_ids,
                    )
                report["control_plane_request_scope"] = {
                    "mode": str(args.control_plane_scope),
                    "request_count": len(request_ids),
                    "scope_file": str(scoped_ids_path) if scoped_ids_path is not None else None,
                }
                scope_args = (
                    ["--request-ids-json", str(scoped_ids_path)] if scoped_ids_path is not None else []
                )
                failed_closed = False
                if str(args.control_plane_scope) == "current_run" and scoped_ids_path is None:
                    fail_detail = {
                        "error": "current_run scope requested but no request_ids were captured; failing closed",
                        "mode": "current_run",
                        "request_count": len(request_ids),
                    }
                    report["control_plane_request_scope"]["fail_closed"] = True
                    add_check("control_plane_jobs", False, fail_detail)
                    jobs = {}
                    failed_closed = True
                else:
                    jobs = build_control_plane_jobs(
                        db_url=args.db_url.strip(),
                        cp_dir=cp_dir,
                        include_injected_failures=bool(args.control_plane_include_injected_failures),
                        scope_args=scope_args,
                    )
                all_jobs_ok = True
                for job_name, (command, output_path) in jobs.items():
                    ok, output = run_job_command(command, timeout_sec=300.0)
                    output_ok = ok and output_path.exists()
                    parsed_preview: Dict[str, Any] = {}
                    if output_ok:
                        try:
                            parsed = json.loads(output_path.read_text(encoding="utf-8"))
                            if isinstance(parsed, dict):
                                parsed_preview = {
                                    "keys": sorted(list(parsed.keys()))[:20],
                                    "path": str(output_path),
                                }
                        except Exception:
                            output_ok = False
                    report["job_reports"][job_name] = {
                        "ok": output_ok,
                        "command_ok": ok,
                        "output": output,
                        "output_path": str(output_path),
                        "preview": parsed_preview,
                    }
                    if not output_ok:
                        all_jobs_ok = False
                if not failed_closed:
                    add_check("control_plane_jobs", all_jobs_ok, report["job_reports"])
        else:
            add_check("control_plane_jobs", True, {"skipped": "run_control_plane_disabled"})

        try:
            gateway_metrics = http_get_json(
                f"{args.node_url.rstrip('/')}/recommender-gateway-metrics",
                timeout_sec=5.0,
            )
            report["gateway_metrics_after_run"] = gateway_metrics
        except Exception as exc:  # noqa: BLE001
            report["gateway_metrics_after_run"] = {"error": str(exc)}

    except Exception as exc:  # noqa: BLE001
        add_check("execution_error", False, {"error": str(exc)})
    finally:
        for proc in reversed(started_processes):
            stop_process(proc)

    report["request_ids"] = request_ids
    report["overall_passed"] = all(bool(item.get("passed")) for item in checks)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
