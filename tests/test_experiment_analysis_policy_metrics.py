from __future__ import annotations

from datetime import datetime, timezone

from scripts.run_experiment_analysis import _fetch_variant_rows


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
            (
                "treatment",  # variant
                "018f0f57-21cb-7f81-8d17-6efec2b5f2be",  # request_id
                False,  # fallback_mode
                42.0,  # latency_total_ms
                True,  # matured_24h
                True,  # matured_96h
                None,  # censorship_24h
                None,  # censorship_96h
                0.31,  # engagement_24h
                0.29,  # engagement_96h
                5.8,  # reach_24h
                6.1,  # reach_96h
                8.0,  # conversion_24h
                7.5,  # conversion_96h
                1.0,  # author_cap_drops
                1.0,  # language_mismatch_drops
                0.0,  # locale_mismatch_drops
                0.0,  # age_limit_drops
                True,  # strict_language_enabled
                False,  # strict_locale_enabled
                2,  # comparable_relevant_count
                0,  # comparable_not_relevant_count
                1,  # comparable_save_count
                1,  # comparable_no_good_options_count
                1,  # recommendation_useful_count
                0,  # recommendation_not_useful_count
                1,  # recommendation_save_count
                1,  # report_export_count
                1,  # followup_question_count
                3,  # comparable_open_count
                2,  # comparable_details_open_count
                6,  # explicit_feedback_count
            )
        ]


class _FakeConn:
    def __init__(self) -> None:
        self.cursor_instance = _FakeCursor()

    def cursor(self):
        return self.cursor_instance


def test_fetch_variant_rows_extracts_policy_violation_fields():
    conn = _FakeConn()
    rows = _fetch_variant_rows(
        conn,
        objective="engagement",
        since=datetime(2026, 3, 1, tzinfo=timezone.utc),
        until=datetime(2026, 3, 8, tzinfo=timezone.utc),
        include_synthetic=False,
        include_injected_failures=False,
    )
    assert "policy_metadata" in conn.cursor_instance.last_sql
    assert len(conn.cursor_instance.last_params) == 7
    assert len(rows) == 1
    row = rows[0]
    assert row["variant"] == "treatment"
    assert row["policy_violation"] is True
    assert row["author_cap_drops"] == 1.0
    assert row["language_mismatch_drops"] == 1.0
    assert row["strict_language_enabled"] is True
    assert row["strict_locale_enabled"] is False
    assert row["comparable_no_good_options_count"] == 1
    assert row["recommendation_useful_count"] == 1
    assert row["explicit_feedback_count"] == 6
