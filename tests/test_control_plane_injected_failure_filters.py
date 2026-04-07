from __future__ import annotations

from datetime import datetime, timezone

from scripts.run_drift_monitor import _fetch_objectives as _fetch_drift_objectives
from scripts.run_drift_monitor import _fetch_policy_rates as _fetch_policy_rates
from scripts.run_experiment_analysis import _fetch_variant_rows as _fetch_experiment_variant_rows
from scripts.run_outcome_attribution import _fetch_request_heads as _fetch_outcome_heads


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
        return []

    def fetchone(self):
        return (0, 0, 0)


class _FakeConn:
    def __init__(self) -> None:
        self.cursor_instance = _FakeCursor()

    def cursor(self):
        return self.cursor_instance


class _FakePolicyCursor(_FakeCursor):
    def fetchone(self):
        # total, fallback_total, tier3_total, author_cap_drop_requests,
        # strict_language_drop_requests, strict_locale_drop_requests,
        # strict_language_requests, strict_locale_requests
        return (10, 3, 2, 4, 1, 2, 2, 4)


class _FakePolicyConn(_FakeConn):
    def __init__(self) -> None:
        self.cursor_instance = _FakePolicyCursor()


def test_outcome_attribution_filters_injected_failures_by_default():
    conn = _FakeConn()
    _fetch_outcome_heads(
        conn,
        since=datetime(2026, 3, 1, tzinfo=timezone.utc),
        include_synthetic=False,
        include_injected_failures=False,
    )
    assert "injected_failure" in conn.cursor_instance.last_sql
    assert len(conn.cursor_instance.last_params) == 5
    assert conn.cursor_instance.last_params[2] is False
    assert conn.cursor_instance.last_params[3] == []
    assert conn.cursor_instance.last_params[4] == []


def test_drift_monitor_objective_scan_supports_injected_failure_filter():
    conn = _FakeConn()
    _fetch_drift_objectives(
        conn,
        since=datetime(2026, 3, 1, tzinfo=timezone.utc),
        until=datetime(2026, 3, 8, tzinfo=timezone.utc),
        include_synthetic=True,
        include_injected_failures=False,
    )
    assert "injected_failure" in conn.cursor_instance.last_sql
    assert len(conn.cursor_instance.last_params) == 6
    assert conn.cursor_instance.last_params[3] is False
    assert conn.cursor_instance.last_params[4] == []
    assert conn.cursor_instance.last_params[5] == []


def test_experiment_rows_support_injected_failure_filter():
    conn = _FakeConn()
    _fetch_experiment_variant_rows(
        conn,
        objective="reach",
        since=datetime(2026, 3, 1, tzinfo=timezone.utc),
        until=datetime(2026, 3, 8, tzinfo=timezone.utc),
        include_synthetic=True,
        include_injected_failures=True,
    )
    assert "injected_failure" in conn.cursor_instance.last_sql
    assert len(conn.cursor_instance.last_params) == 7
    assert conn.cursor_instance.last_params[4] is True
    assert conn.cursor_instance.last_params[5] == []
    assert conn.cursor_instance.last_params[6] == []


def test_drift_policy_rates_use_policy_metadata_fields():
    conn = _FakePolicyConn()
    rates = _fetch_policy_rates(
        conn,
        objective="reach",
        since=datetime(2026, 3, 1, tzinfo=timezone.utc),
        until=datetime(2026, 3, 8, tzinfo=timezone.utc),
        include_synthetic=False,
        include_injected_failures=False,
    )
    assert "policy_metadata" in conn.cursor_instance.last_sql
    assert rates["sample_count"] == 10
    assert rates["fallback_rate"] == 0.3
    assert rates["author_cap_drop_rate"] == 0.4
    assert rates["strict_language_drop_rate"] == 0.5
    assert rates["strict_locale_drop_rate"] == 0.5
