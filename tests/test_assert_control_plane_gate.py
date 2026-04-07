from __future__ import annotations

import json
from pathlib import Path

from scripts.assert_control_plane_gate import evaluate_gate, main


def _write_report(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _base_report() -> dict:
    return {
        "overall_passed": True,
        "checks": [
            {"name": "control_plane_jobs", "passed": True},
            {"name": "recommendation_requests_valid", "passed": True},
        ],
        "job_reports": {
            "outcome_attribution": {"ok": True},
            "drift_monitor": {"ok": True},
            "experiment_analysis": {"ok": True},
            "retrain_controller": {"ok": True},
        },
        "gateway_metrics_after_run": {
            "stage_latency": {
                "total": {"p95_ms": 149.0},
                "python_roundtrip": {"p95_ms": 151.0},
            },
            "fallback_by_reason": {
                "incompatible_artifact": 1,
                "fetch failed": 1,
            },
        },
    }


def test_evaluate_gate_passes_with_allowed_fallbacks():
    summary = evaluate_gate(
        _base_report(),
        latency_p95_threshold=200.0,
        allowed_fallback_reasons=["incompatible_artifact", "fetch failed"],
    )
    assert summary["passed"] is True
    assert summary["failures"] == []
    assert summary["input"]["latency_source"] == "total"


def test_evaluate_gate_fails_on_latency_and_disallowed_fallback():
    report = _base_report()
    report["gateway_metrics_after_run"]["stage_latency"]["total"]["p95_ms"] = 300.0
    report["gateway_metrics_after_run"]["fallback_by_reason"]["unexpected"] = 1
    summary = evaluate_gate(
        report,
        latency_p95_threshold=250.0,
        allowed_fallback_reasons=["incompatible_artifact", "fetch failed"],
    )
    assert summary["passed"] is False
    assert "latency_p95_exceeded" in summary["failures"]
    assert "fallback_reasons_disallowed" in summary["failures"]


def test_main_emits_json_and_exit_code_for_pass(tmp_path, capsys, monkeypatch):
    report_path = _write_report(tmp_path / "report.json", _base_report())
    monkeypatch.setattr(
        "sys.argv",
        [
            "assert_control_plane_gate.py",
            "--input-json",
            str(report_path),
            "--latency-p95-ms-threshold",
            "200",
            "--allowed-fallback-reason",
            "incompatible_artifact",
            "--allowed-fallback-reason",
            "fetch failed",
        ],
    )
    assert main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["passed"] is True


def test_main_emits_json_and_exit_code_for_fail(tmp_path, capsys, monkeypatch):
    report = _base_report()
    report["overall_passed"] = False
    report_path = _write_report(tmp_path / "report.json", report)
    monkeypatch.setattr(
        "sys.argv",
        [
            "assert_control_plane_gate.py",
            "--input-json",
            str(report_path),
            "--latency-p95-ms-threshold",
            "200",
            "--allowed-fallback-reason",
            "incompatible_artifact",
            "--allowed-fallback-reason",
            "fetch failed",
        ],
    )
    assert main() == 1
    out = json.loads(capsys.readouterr().out)
    assert out["passed"] is False
    assert "overall_passed_false" in out["failures"]
