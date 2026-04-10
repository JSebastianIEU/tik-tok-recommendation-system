from __future__ import annotations

from datetime import datetime, timezone

from src.recommendation.control_plane import (
    DriftThresholds,
    build_outcome_event,
    build_retrain_decision,
    ks_statistic,
    population_stability_index,
    should_trigger_retrain,
    summarize_drift,
)


def test_build_outcome_event_marks_not_matured_censorship():
    served_at = datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc)
    as_of_run = datetime(2026, 3, 27, 10, 0, tzinfo=timezone.utc)
    payload = build_outcome_event(
        request_id="018f0f57-21cb-7f81-8d17-6efec2b5f2be",
        objective_effective="engagement",
        served_at=served_at,
        as_of_run_time=as_of_run,
        window_hours=24,
        snapshots=[],
    )
    assert payload["matured"] is False
    assert payload["censorship_reason"] == "not_matured_24h"


def test_build_outcome_event_computes_values_when_snapshot_available():
    served_at = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    as_of_run = datetime(2026, 3, 27, 12, 0, tzinfo=timezone.utc)
    snapshots = [
        {
            "event_time": "2026-03-20T13:00:00Z",
            "ingested_at": "2026-03-20T13:00:00Z",
            "plays": 1000,
            "likes": 100,
            "comments_count": 20,
            "shares": 10,
        },
        {
            "event_time": "2026-03-21T11:00:00Z",
            "ingested_at": "2026-03-21T11:00:00Z",
            "plays": 1500,
            "likes": 120,
            "comments_count": 25,
            "shares": 15,
        },
    ]
    payload = build_outcome_event(
        request_id="018f0f57-21cb-7f81-8d17-6efec2b5f2be",
        objective_effective="engagement",
        served_at=served_at,
        as_of_run_time=as_of_run,
        window_hours=24,
        snapshots=snapshots,
    )
    assert payload["matured"] is True
    assert payload["censorship_reason"] is None
    assert payload["engagement_value"] > 0
    assert payload["reach_value"] > 0


def test_drift_metrics_and_thresholds():
    psi_value = population_stability_index([0.1, 0.2, 0.3], [0.9, 1.0, 1.1])
    ks_value = ks_statistic([0.1, 0.2, 0.3], [0.9, 1.0, 1.1])
    assert psi_value > 0
    assert ks_value > 0

    summary = summarize_drift(
        feature_expected={"mandatory_score": [0.1, 0.2, 0.3]},
        feature_actual={"mandatory_score": [2.0, 2.1, 2.2]},
        label_expected={"primary_24h": [0.1, 0.2, 0.3]},
        label_actual={"primary_24h": [1.0, 1.1, 1.2]},
        policy_baseline={"fallback_rate": 0.01, "sample_count": 30},
        policy_current={"fallback_rate": 0.10, "sample_count": 30},
        thresholds=DriftThresholds(
            min_feature_samples=1,
            min_label_samples=1,
            min_policy_samples=1,
        ),
    )
    assert summary["severity"] == "critical"
    assert summary["trigger_recommendation"] == "retrain_candidate"


def test_drift_summary_marks_insufficient_data_when_support_is_too_low():
    summary = summarize_drift(
        feature_expected={"mandatory_score": [0.1]},
        feature_actual={"mandatory_score": [3.0]},
        label_expected={"primary_24h": [0.1]},
        label_actual={"primary_24h": [1.2]},
        policy_baseline={"fallback_rate": 0.01, "sample_count": 1},
        policy_current={"fallback_rate": 0.80, "sample_count": 1},
        thresholds=DriftThresholds(
            min_feature_samples=10,
            min_label_samples=10,
            min_policy_samples=10,
        ),
    )
    assert summary["severity"] == "insufficient_data"
    assert summary["trigger_recommendation"] == "insufficient_data"
    assert summary["support"]["streams_sufficient"] is False
    assert summary["breaches"]["policy"] is False


def test_drift_policy_breach_triggers_on_strict_language_drop_rate():
    summary = summarize_drift(
        feature_expected={"mandatory_score": [0.1, 0.2, 0.3]},
        feature_actual={"mandatory_score": [0.1, 0.21, 0.31]},
        label_expected={"primary_24h": [0.1, 0.2, 0.3]},
        label_actual={"primary_24h": [0.1, 0.2, 0.3]},
        policy_baseline={"fallback_rate": 0.01, "sample_count": 50},
        policy_current={
            "fallback_rate": 0.01,
            "sample_count": 50,
            "strict_language_drop_rate": 0.35,
        },
        thresholds=DriftThresholds(
            min_feature_samples=1,
            min_label_samples=1,
            min_policy_samples=1,
            policy_strict_language_drop_rate=0.20,
        ),
    )
    assert summary["breaches"]["policy"] is True
    assert summary["policy_breach_components"]["strict_language_drop_rate"] is True
    assert summary["severity"] == "critical"


def test_should_trigger_retrain_with_consecutive_critical():
    triggered, reason = should_trigger_retrain(
        recent_severities=["warning", "critical", "critical"],
        scheduled_due=False,
        consecutive_critical_required=2,
    )
    assert triggered is True
    assert reason == "drift_trigger"


def test_should_not_trigger_retrain_for_critical_without_retrain_recommendation():
    triggered, reason = should_trigger_retrain(
        recent_severities=["critical", "critical"],
        recent_trigger_recommendations=["insufficient_data", "insufficient_data"],
        scheduled_due=False,
        consecutive_critical_required=2,
    )
    assert triggered is False
    assert reason == "no_trigger"


def test_build_retrain_decision_payload_shape():
    decision = build_retrain_decision(
        trigger_source="scheduled_weekly",
        selected_bundle_id="bundle-123",
        previous_bundle_id="bundle-122",
        promoted=True,
        objective_metrics={"engagement": {"passed": True}},
        drift_evidence={"engagement": {"recent": ["critical", "critical"]}},
    )
    assert decision["trigger_source"] == "scheduled_weekly"
    assert decision["selected_bundle_id"] == "bundle-123"
    assert decision["promoted"] is True
    assert "generated_at" in decision
