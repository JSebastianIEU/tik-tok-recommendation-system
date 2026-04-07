from __future__ import annotations

from scripts.run_experiment_analysis import _build_comparison_block


def test_experiment_comparison_marks_insufficient_when_support_low():
    comparison = _build_comparison_block(
        control={
            "primary_kpi_24h": 0.40,
            "stability_kpi_96h": 0.30,
            "fallback_rate": 0.02,
            "latency_p95_ms": 100.0,
            "policy_violation_rate": 0.01,
            "matured_primary_24h_samples": 3,
            "matured_stability_96h_samples": 2,
        },
        treatment={
            "primary_kpi_24h": 0.45,
            "stability_kpi_96h": 0.35,
            "fallback_rate": 0.01,
            "latency_p95_ms": 105.0,
            "policy_violation_rate": 0.02,
            "matured_primary_24h_samples": 4,
            "matured_stability_96h_samples": 1,
        },
        min_matured_primary_samples_per_variant=10,
        min_matured_stability_samples_per_variant=5,
    )
    assert comparison["evidence_sufficient"] is False
    assert comparison["verdict"] == "insufficient_data"


def test_experiment_comparison_reports_treatment_lead_when_supported():
    comparison = _build_comparison_block(
        control={
            "primary_kpi_24h": 0.40,
            "stability_kpi_96h": 0.30,
            "fallback_rate": 0.03,
            "latency_p95_ms": 110.0,
            "policy_violation_rate": 0.02,
            "matured_primary_24h_samples": 20,
            "matured_stability_96h_samples": 10,
        },
        treatment={
            "primary_kpi_24h": 0.46,
            "stability_kpi_96h": 0.32,
            "fallback_rate": 0.02,
            "latency_p95_ms": 112.0,
            "policy_violation_rate": 0.03,
            "matured_primary_24h_samples": 22,
            "matured_stability_96h_samples": 11,
        },
        min_matured_primary_samples_per_variant=10,
        min_matured_stability_samples_per_variant=5,
    )
    assert comparison["evidence_sufficient"] is True
    assert comparison["verdict"] == "treatment_leads"
    assert (
        comparison["deltas"]["primary_kpi_24h_treatment_minus_control"] > 0.0
    )


def test_experiment_comparison_prefers_immediate_feedback_when_supported():
    comparison = _build_comparison_block(
        control={
            "primary_kpi_24h": 0.60,
            "stability_kpi_96h": 0.40,
            "fallback_rate": 0.02,
            "latency_p95_ms": 100.0,
            "policy_violation_rate": 0.01,
            "matured_primary_24h_samples": 2,
            "matured_stability_96h_samples": 1,
            "explicit_feedback_samples": 12,
            "comparable_relevant_rate": 0.30,
            "recommendation_useful_rate": 0.25,
        },
        treatment={
            "primary_kpi_24h": 0.55,
            "stability_kpi_96h": 0.35,
            "fallback_rate": 0.02,
            "latency_p95_ms": 102.0,
            "policy_violation_rate": 0.01,
            "matured_primary_24h_samples": 2,
            "matured_stability_96h_samples": 1,
            "explicit_feedback_samples": 14,
            "comparable_relevant_rate": 0.45,
            "recommendation_useful_rate": 0.50,
        },
        min_matured_primary_samples_per_variant=10,
        min_matured_stability_samples_per_variant=5,
        min_explicit_feedback_samples_per_variant=10,
    )
    assert comparison["evidence_sufficient"] is True
    assert comparison["feedback_support"]["evidence_basis"] == "immediate_feedback"
    assert comparison["verdict"] == "treatment_leads"
