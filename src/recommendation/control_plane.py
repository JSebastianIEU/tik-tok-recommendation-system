from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            return None
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class DriftThresholds:
    feature_psi_mandatory: float = 0.20
    feature_psi_optional: float = 0.30
    label_ks: float = 0.12
    label_relative_mean_delta: float = 0.20
    policy_fallback_rate: float = 0.05
    policy_wow_fallback_delta_pp: float = 0.02


def classify_censorship_reason(
    *,
    served_at: datetime,
    as_of_run_time: datetime,
    window_hours: int,
    snapshot_ingested_at: Optional[datetime],
) -> Optional[str]:
    maturity_cutoff = served_at + timedelta(hours=int(window_hours))
    if maturity_cutoff > as_of_run_time:
        return f"not_matured_{window_hours}h"
    if snapshot_ingested_at is None:
        return "missing_snapshot"
    if snapshot_ingested_at > as_of_run_time:
        return "ingested_after_cutoff"
    return None


def select_snapshot_as_of(
    snapshots: Sequence[Dict[str, Any]],
    *,
    boundary_time: datetime,
    as_of_run_time: datetime,
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_event: Optional[datetime] = None
    for snapshot in snapshots:
        event_time = _parse_dt(snapshot.get("event_time") or snapshot.get("scraped_at"))
        if event_time is None or event_time > boundary_time:
            continue
        ingested_at = _parse_dt(snapshot.get("ingested_at") or snapshot.get("scraped_at"))
        if ingested_at is None or ingested_at > as_of_run_time:
            continue
        if best_event is None or event_time > best_event:
            best = snapshot
            best_event = event_time
    return best


def compute_objective_values_from_snapshot(snapshot: Dict[str, Any]) -> Dict[str, float]:
    views = max(1.0, _to_float(snapshot.get("plays") or snapshot.get("views"), 0.0))
    likes = _to_float(snapshot.get("likes"), 0.0)
    comments = _to_float(snapshot.get("comments_count"), 0.0)
    shares = _to_float(snapshot.get("shares"), 0.0)
    return {
        "reach": math.log1p(max(0.0, views)),
        "engagement": (likes + comments + shares) / max(1.0, views),
        "conversion": (shares * 1000.0) / max(1.0, views),
    }


def build_outcome_event(
    *,
    request_id: str,
    objective_effective: str,
    served_at: datetime,
    as_of_run_time: datetime,
    window_hours: int,
    snapshots: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    boundary_time = served_at + timedelta(hours=int(window_hours))
    selected = select_snapshot_as_of(
        snapshots,
        boundary_time=boundary_time,
        as_of_run_time=as_of_run_time,
    )
    snapshot_ingested_at = _parse_dt(
        (selected or {}).get("ingested_at") or (selected or {}).get("scraped_at")
    )
    censorship_reason = classify_censorship_reason(
        served_at=served_at,
        as_of_run_time=as_of_run_time,
        window_hours=window_hours,
        snapshot_ingested_at=snapshot_ingested_at,
    )
    matured = censorship_reason is None and selected is not None
    values = compute_objective_values_from_snapshot(selected or {}) if matured else {
        "reach": 0.0,
        "engagement": 0.0,
        "conversion": 0.0,
    }
    return {
        "request_id": request_id,
        "objective_effective": objective_effective,
        "window_hours": int(window_hours),
        "matured": bool(matured),
        "censorship_reason": censorship_reason,
        "reach_value": float(values["reach"]),
        "engagement_value": float(values["engagement"]),
        "conversion_value": float(values["conversion"]),
    }


def population_stability_index(
    expected: Sequence[float],
    actual: Sequence[float],
    bins: int = 10,
) -> float:
    if not expected or not actual:
        return 0.0
    min_value = min(min(expected), min(actual))
    max_value = max(max(expected), max(actual))
    if max_value <= min_value:
        return 0.0
    step = (max_value - min_value) / float(max(1, bins))
    eps = 1e-9
    psi = 0.0
    for i in range(max(1, bins)):
        low = min_value + i * step
        high = max_value if i == bins - 1 else low + step
        exp_share = sum(1 for value in expected if (value >= low and value <= high)) / max(
            1, len(expected)
        )
        act_share = sum(1 for value in actual if (value >= low and value <= high)) / max(
            1, len(actual)
        )
        exp_share = max(exp_share, eps)
        act_share = max(act_share, eps)
        psi += (act_share - exp_share) * math.log(act_share / exp_share)
    return float(psi)


def ks_statistic(expected: Sequence[float], actual: Sequence[float]) -> float:
    if not expected or not actual:
        return 0.0
    expected_sorted = sorted(float(item) for item in expected)
    actual_sorted = sorted(float(item) for item in actual)
    points = sorted(set(expected_sorted + actual_sorted))
    if not points:
        return 0.0
    max_gap = 0.0
    for point in points:
        ecdf_e = sum(1 for item in expected_sorted if item <= point) / max(1, len(expected_sorted))
        ecdf_a = sum(1 for item in actual_sorted if item <= point) / max(1, len(actual_sorted))
        max_gap = max(max_gap, abs(ecdf_e - ecdf_a))
    return float(max_gap)


def relative_mean_delta(expected: Sequence[float], actual: Sequence[float]) -> float:
    if not expected or not actual:
        return 0.0
    mean_e = sum(float(item) for item in expected) / max(1, len(expected))
    mean_a = sum(float(item) for item in actual) / max(1, len(actual))
    if abs(mean_e) <= 1e-9:
        return 0.0 if abs(mean_a) <= 1e-9 else 1.0
    return float(abs(mean_a - mean_e) / abs(mean_e))


def summarize_drift(
    *,
    feature_expected: Dict[str, Sequence[float]],
    feature_actual: Dict[str, Sequence[float]],
    label_expected: Dict[str, Sequence[float]],
    label_actual: Dict[str, Sequence[float]],
    policy_baseline: Dict[str, float],
    policy_current: Dict[str, float],
    thresholds: DriftThresholds = DriftThresholds(),
) -> Dict[str, Any]:
    feature_drift: Dict[str, Dict[str, float]] = {}
    feature_breach_mandatory = False
    feature_breach_optional = False
    for key, expected_values in feature_expected.items():
        actual_values = feature_actual.get(key, [])
        psi_value = population_stability_index(expected_values, actual_values)
        feature_drift[key] = {"psi": round(psi_value, 6)}
        if key.startswith("mandatory_") and psi_value > thresholds.feature_psi_mandatory:
            feature_breach_mandatory = True
        if key.startswith("optional_") and psi_value > thresholds.feature_psi_optional:
            feature_breach_optional = True

    label_drift: Dict[str, Dict[str, float]] = {}
    label_breach = False
    for key, expected_values in label_expected.items():
        actual_values = label_actual.get(key, [])
        ks_value = ks_statistic(expected_values, actual_values)
        mean_delta = relative_mean_delta(expected_values, actual_values)
        label_drift[key] = {
            "ks": round(ks_value, 6),
            "relative_mean_delta": round(mean_delta, 6),
        }
        if ks_value > thresholds.label_ks or mean_delta > thresholds.label_relative_mean_delta:
            label_breach = True

    baseline_fallback = float(policy_baseline.get("fallback_rate") or 0.0)
    current_fallback = float(policy_current.get("fallback_rate") or 0.0)
    policy_delta = current_fallback - baseline_fallback
    policy_drift = {
        "fallback_rate": round(current_fallback, 6),
        "fallback_rate_wow_delta": round(policy_delta, 6),
        "constraint_tier_3_rate": round(float(policy_current.get("constraint_tier_3_rate") or 0.0), 6),
        "author_cap_drop_rate": round(float(policy_current.get("author_cap_drop_rate") or 0.0), 6),
        "strict_language_drop_rate": round(
            float(policy_current.get("strict_language_drop_rate") or 0.0), 6
        ),
    }
    policy_breach = (
        current_fallback > thresholds.policy_fallback_rate
        or policy_delta > thresholds.policy_wow_fallback_delta_pp
    )

    if feature_breach_mandatory or label_breach or policy_breach:
        severity = "critical"
    elif feature_breach_optional:
        severity = "warning"
    else:
        severity = "ok"

    return {
        "feature_drift": feature_drift,
        "label_drift": label_drift,
        "policy_drift": policy_drift,
        "breaches": {
            "feature_mandatory": feature_breach_mandatory,
            "feature_optional": feature_breach_optional,
            "label": label_breach,
            "policy": policy_breach,
        },
        "severity": severity,
        "trigger_recommendation": (
            "retrain_candidate" if severity == "critical" else "monitor" if severity == "warning" else "none"
        ),
    }


def should_trigger_retrain(
    *,
    recent_severities: Sequence[str],
    scheduled_due: bool,
    consecutive_critical_required: int = 2,
) -> Tuple[bool, str]:
    normalized = [str(item).strip().lower() for item in recent_severities if str(item).strip()]
    trailing_critical = 0
    for severity in reversed(normalized):
        if severity == "critical":
            trailing_critical += 1
        else:
            break
    if trailing_critical >= max(1, int(consecutive_critical_required)):
        return True, "drift_trigger"
    if scheduled_due:
        return True, "scheduled_weekly"
    return False, "no_trigger"


def build_retrain_decision(
    *,
    trigger_source: str,
    selected_bundle_id: Optional[str],
    previous_bundle_id: Optional[str],
    promoted: bool,
    objective_metrics: Dict[str, Any],
    drift_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "trigger_source": trigger_source,
        "selected_bundle_id": selected_bundle_id,
        "previous_bundle_id": previous_bundle_id,
        "promoted": bool(promoted),
        "objective_metrics": objective_metrics,
        "drift_evidence": drift_evidence,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


__all__ = [
    "DriftThresholds",
    "build_outcome_event",
    "population_stability_index",
    "ks_statistic",
    "relative_mean_delta",
    "summarize_drift",
    "should_trigger_retrain",
    "build_retrain_decision",
    "select_snapshot_as_of",
    "classify_censorship_reason",
    "compute_objective_values_from_snapshot",
]

