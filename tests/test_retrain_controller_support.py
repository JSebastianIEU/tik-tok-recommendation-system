from __future__ import annotations

from datetime import datetime, timezone

from scripts.run_retrain_controller import (
    _apply_drift_trigger_support_gate,
    _collect_request_id_scope,
    _fetch_recent_outcome_support,
    _load_drift_observations_from_report,
    _non_regression_gate,
    _resolve_trigger_source,
)


class _FakeCursor:
    def __init__(self) -> None:
        self.last_sql = ""
        self.last_params = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.last_sql = str(sql)
        self.last_params = tuple(params or ())

    def fetchall(self):
        return [
            ("reach", 30, 24, 12),
            ("engagement", 10, 5, 1),
        ]


class _FakeConn:
    def __init__(self) -> None:
        self.cursor_instance = _FakeCursor()

    def cursor(self):
        return self.cursor_instance


def test_apply_drift_trigger_support_gate_blocks_low_support():
    trigger, reason, support_ok = _apply_drift_trigger_support_gate(
        trigger=True,
        reason="drift_trigger",
        support={"matured_24h_count": 3, "matured_96h_count": 1},
        min_matured_24h=10,
        min_matured_96h=5,
    )
    assert trigger is False
    assert reason == "insufficient_feedback_support"
    assert support_ok is False


def test_apply_drift_trigger_support_gate_allows_scheduled_trigger_without_support():
    trigger, reason, support_ok = _apply_drift_trigger_support_gate(
        trigger=True,
        reason="scheduled_weekly",
        support={"matured_24h_count": 0, "matured_96h_count": 0},
        min_matured_24h=10,
        min_matured_96h=5,
    )
    assert trigger is True
    assert reason == "scheduled_weekly"
    assert support_ok is False


def test_apply_drift_trigger_support_gate_requires_feedback_contrast_when_available():
    trigger, reason, support_ok = _apply_drift_trigger_support_gate(
        trigger=True,
        reason="drift_trigger",
        support={
            "strong_feedback_count": 12,
            "explicit_positive_request_count": 4,
            "explicit_negative_request_count": 0,
            "no_good_option_request_count": 0,
            "matured_24h_count": 0,
            "matured_96h_count": 0,
        },
        min_matured_24h=10,
        min_matured_96h=5,
    )
    assert trigger is False
    assert reason == "insufficient_feedback_support"
    assert support_ok is False


def test_fetch_recent_outcome_support_uses_filters_and_counts():
    conn = _FakeConn()
    out = _fetch_recent_outcome_support(
        conn,
        since=datetime(2026, 3, 1, tzinfo=timezone.utc),
        until=datetime(2026, 3, 8, tzinfo=timezone.utc),
        include_synthetic=False,
        include_injected_failures=False,
    )
    assert "rec_outcome_events" in conn.cursor_instance.last_sql
    assert "is_synthetic" in conn.cursor_instance.last_sql
    assert "injected_failure" in conn.cursor_instance.last_sql
    assert len(conn.cursor_instance.last_params) == 6
    assert conn.cursor_instance.last_params[4] == []
    assert conn.cursor_instance.last_params[5] == []
    assert out["reach"]["matured_24h_count"] == 24
    assert out["reach"]["matured_96h_count"] == 12
    assert out["engagement"]["request_count"] == 10


def test_fetch_recent_outcome_support_applies_request_id_scope():
    conn = _FakeConn()
    request_ids = ["019d4388-69c1-7693-aa7c-afc3d1940839"]
    _fetch_recent_outcome_support(
        conn,
        since=datetime(2026, 3, 1, tzinfo=timezone.utc),
        until=datetime(2026, 3, 8, tzinfo=timezone.utc),
        include_synthetic=True,
        include_injected_failures=False,
        request_ids=request_ids,
    )
    assert conn.cursor_instance.last_params[4] == request_ids
    assert conn.cursor_instance.last_params[5] == request_ids


def test_collect_request_id_scope_rejects_invalid_uuid():
    try:
        _collect_request_id_scope(["not-a-uuid"], None)
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid request id")


def test_load_drift_observations_from_report(tmp_path):
    report_path = tmp_path / "drift_report.json"
    report_path.write_text(
        '{"objectives":{"reach":{"severity":"critical","trigger_recommendation":"retrain_candidate"}}}',
        encoding="utf-8",
    )
    out = _load_drift_observations_from_report(report_path)
    assert out["reach"][0]["severity"] == "critical"
    assert out["reach"][0]["trigger_recommendation"] == "retrain_candidate"


def test_resolve_trigger_source_prefers_scheduled_when_scheduled_trigger_present():
    source = _resolve_trigger_source(
        trigger_any=True,
        scheduled_due=True,
        objective_triggers={
            "reach": {"trigger": True, "reason": "scheduled_weekly"},
            "engagement": {"trigger": False, "reason": "no_trigger"},
        },
    )
    assert source == "scheduled_weekly"


def test_non_regression_gate_fails_when_retriever_regresses():
    ok, diagnostics = _non_regression_gate(
        baseline_metrics={
            "reach": {
                "ranker": {"ndcg@10": 0.50, "mrr@20": 0.60},
                "retriever": {"recall@100": 0.40},
            }
        },
        candidate_metrics={
            "reach": {
                "ranker": {"ndcg@10": 0.51, "mrr@20": 0.61},
                "retriever": {"recall@100": 0.35},
            }
        },
        threshold_ratio=0.995,
        retriever_threshold_ratio=0.995,
    )
    assert ok is False
    assert diagnostics["reach"]["ranker_gate_passed"] is True
    assert diagnostics["reach"]["retriever_gate_passed"] is False


def test_non_regression_gate_passes_when_ranker_and_retriever_hold():
    ok, diagnostics = _non_regression_gate(
        baseline_metrics={
            "reach": {
                "ranker": {"ndcg@10": 0.50, "mrr@20": 0.60},
                "retriever": {"recall@100": 0.40},
            }
        },
        candidate_metrics={
            "reach": {
                "ranker": {"ndcg@10": 0.51, "mrr@20": 0.61},
                "retriever": {"recall@100": 0.41},
            }
        },
        threshold_ratio=0.995,
        retriever_threshold_ratio=0.995,
    )
    assert ok is True
    assert diagnostics["reach"]["ranker_gate_passed"] is True
    assert diagnostics["reach"]["retriever_gate_passed"] is True
