from __future__ import annotations

from scripts.run_live_e2e_validation import (
    _write_request_ids_scope,
    build_control_plane_jobs,
    build_validation_request_plan,
    build_recommendation_payload,
    validate_recommendation_payload,
)


def test_build_recommendation_payload_includes_portfolio_and_routing():
    payload = build_recommendation_payload(
        2,
        candidate_ids=["v001", "v002"],
        objective_override="conversion",
        force_variant="control",
    )
    assert payload["objective"] in {"reach", "engagement", "conversion"}
    assert payload["objective"] == "conversion"
    assert payload["portfolio"]["enabled"] is True
    assert payload["portfolio"]["weights"]["durability"] == 0.20
    assert payload["routing"]["track"] == "post_publication"
    assert payload["routing"]["allow_fallback"] is True
    assert payload["candidate_ids"] == ["v001", "v002"]
    assert payload["experiment"]["id"] == "e2e_live_validation"
    assert payload["experiment"]["force_variant"] == "control"
    assert payload["traffic"]["class"] == "synthetic"
    assert payload["traffic"]["injected_failure"] is False


def test_build_validation_request_plan_balances_variants_per_objective():
    plan = build_validation_request_plan(6)
    assert len(plan) == 6
    assert plan == [
        {"objective": "reach", "force_variant": "control"},
        {"objective": "reach", "force_variant": "treatment"},
        {"objective": "engagement", "force_variant": "control"},
        {"objective": "engagement", "force_variant": "treatment"},
        {"objective": "conversion", "force_variant": "control"},
        {"objective": "conversion", "force_variant": "treatment"},
    ]


def test_validate_recommendation_payload_happy_path():
    payload = {
        "items": [
            {
                "candidate_id": "vid-1",
                "rank": 1,
                "score": 0.91,
            }
        ]
    }
    ok, reason = validate_recommendation_payload(payload)
    assert ok is True
    assert reason == "ok"


def test_validate_recommendation_payload_rejects_missing_items():
    payload = {"items": []}
    ok, reason = validate_recommendation_payload(payload)
    assert ok is True
    assert reason == "ok_empty_items"


def test_write_request_ids_scope_writes_json_payload(tmp_path):
    scope_path = tmp_path / "request_scope.json"
    written = _write_request_ids_scope(
        scope_path,
        [
            "019d3f88-1bfb-7be5-8568-93da88e1b3c5",
            "019d3f88-2bd7-736a-99a0-00d98ebbd69a",
        ],
    )
    assert written == scope_path
    parsed = scope_path.read_text(encoding="utf-8")
    assert "\"request_ids\"" in parsed


def test_write_request_ids_scope_returns_none_when_empty(tmp_path):
    scope_path = tmp_path / "request_scope.json"
    assert _write_request_ids_scope(scope_path, []) is None
    assert not scope_path.exists()


def test_build_control_plane_jobs_propagates_scope_and_flags(tmp_path):
    cp_dir = tmp_path / "control_plane"
    jobs = build_control_plane_jobs(
        db_url="postgresql://example",
        cp_dir=cp_dir,
        include_injected_failures=True,
        scope_args=["--request-ids-json", str(cp_dir / "request_ids_scope.json")],
    )
    for name in ("outcome_attribution", "drift_monitor", "experiment_analysis", "retrain_controller"):
        command, _ = jobs[name]
        assert "--request-ids-json" in command
        assert "--include-synthetic" in command
        assert "--include-injected-failures" in command
    retrain_command, _ = jobs["retrain_controller"]
    assert "--drift-report-json" in retrain_command


def test_current_run_scope_uses_ids_and_fails_closed_when_empty(tmp_path):
    scope_path = tmp_path / "request_ids_scope.json"
    populated = _write_request_ids_scope(
        scope_path,
        [
            "019d3f88-1bfb-7be5-8568-93da88e1b3c5",
            "019d3f88-2bd7-736a-99a0-00d98ebbd69a",
        ],
    )
    assert populated == scope_path
    jobs = build_control_plane_jobs(
        db_url="postgresql://example",
        cp_dir=tmp_path / "control_plane",
        include_injected_failures=False,
        scope_args=["--request-ids-json", str(scope_path)],
    )
    for name in ("outcome_attribution", "drift_monitor", "experiment_analysis", "retrain_controller"):
        command, _ = jobs[name]
        assert "--request-ids-json" in command
    assert _write_request_ids_scope(tmp_path / "empty_scope.json", []) is None
