#!/usr/bin/env python3
"""Assert control-plane live E2E gate thresholds from a JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _get_nested(payload: Dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _find_check(report: Dict[str, Any], name: str) -> Dict[str, Any] | None:
    checks = report.get("checks")
    if not isinstance(checks, list):
        return None
    for item in checks:
        if isinstance(item, dict) and str(item.get("name") or "") == name:
            return item
    return None


def _extract_allowed_reasons(values: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for value in values:
        raw = str(value or "").strip()
        if raw:
            out.add(raw)
    return out


def evaluate_gate(
    report: Dict[str, Any],
    *,
    latency_p95_threshold: float,
    allowed_fallback_reasons: Sequence[str],
) -> Dict[str, Any]:
    allowed = _extract_allowed_reasons(allowed_fallback_reasons)
    failures: List[str] = []

    overall_passed = bool(report.get("overall_passed"))
    if not overall_passed:
        failures.append("overall_passed_false")

    control_plane_check = _find_check(report, "control_plane_jobs")
    control_plane_check_passed = bool(
        control_plane_check and control_plane_check.get("passed") is True
    )
    if not control_plane_check_passed:
        failures.append("control_plane_jobs_check_failed")

    job_reports = report.get("job_reports")
    job_report_failures: List[str] = []
    if isinstance(job_reports, dict):
        for job_name, job_report in sorted(job_reports.items()):
            if not isinstance(job_report, dict) or job_report.get("ok") is not True:
                job_report_failures.append(str(job_name))
    elif job_reports is not None:
        job_report_failures.append("job_reports_not_object")
    if job_report_failures:
        failures.append("job_reports_failed")

    stage_latency = _get_nested(report, ["gateway_metrics_after_run", "stage_latency"])
    latency_value = None
    latency_source = None
    if isinstance(stage_latency, dict):
        total_latency = stage_latency.get("total")
        if isinstance(total_latency, dict) and total_latency.get("p95_ms") is not None:
            latency_value = float(total_latency["p95_ms"])
            latency_source = "total"
        else:
            roundtrip_latency = stage_latency.get("python_roundtrip")
            if isinstance(roundtrip_latency, dict) and roundtrip_latency.get("p95_ms") is not None:
                latency_value = float(roundtrip_latency["p95_ms"])
                latency_source = "python_roundtrip"
    if latency_value is None:
        failures.append("latency_missing")
    elif latency_value > float(latency_p95_threshold):
        failures.append("latency_p95_exceeded")

    fallback_by_reason = _get_nested(report, ["gateway_metrics_after_run", "fallback_by_reason"])
    observed_fallback_reasons: List[str] = []
    if isinstance(fallback_by_reason, dict):
        observed_fallback_reasons = [
            str(reason)
            for reason, count in sorted(fallback_by_reason.items())
            if int(count or 0) > 0
        ]
        disallowed = sorted(reason for reason in observed_fallback_reasons if reason not in allowed)
        if disallowed:
            failures.append("fallback_reasons_disallowed")
    elif fallback_by_reason is not None:
        failures.append("fallback_by_reason_invalid")

    passed = not failures
    return {
        "passed": passed,
        "failures": failures,
        "input": {
            "overall_passed": overall_passed,
            "control_plane_jobs_check_passed": control_plane_check_passed,
            "job_report_failures": job_report_failures,
            "latency_p95_ms": latency_value,
            "latency_source": latency_source,
            "latency_p95_threshold": float(latency_p95_threshold),
            "observed_fallback_reasons": observed_fallback_reasons,
            "allowed_fallback_reasons": sorted(allowed),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Assert control-plane live E2E sign-off gate.")
    parser.add_argument("--input-json", type=Path, required=True, help="Live E2E JSON report path.")
    parser.add_argument(
        "--latency-p95-ms-threshold",
        type=float,
        default=250.0,
        help="Maximum allowed p95 latency for gateway_metrics_after_run.stage_latency.total.",
    )
    parser.add_argument(
        "--allowed-fallback-reason",
        action="append",
        default=[],
        help="Allowed fallback reason. May be repeated.",
    )
    args = parser.parse_args()

    try:
        report = _load_json(args.input_json)
        summary = evaluate_gate(
            report,
            latency_p95_threshold=float(args.latency_p95_ms_threshold),
            allowed_fallback_reasons=args.allowed_fallback_reason,
        )
        summary["input"]["report_path"] = str(args.input_json)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["passed"] else 1
    except Exception as exc:  # noqa: BLE001
        summary = {
            "passed": False,
            "failures": ["exception"],
            "error": str(exc),
            "input": {
                "report_path": str(args.input_json),
                "latency_p95_threshold": float(args.latency_p95_ms_threshold),
                "allowed_fallback_reasons": sorted(
                    _extract_allowed_reasons(args.allowed_fallback_reason)
                ),
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
