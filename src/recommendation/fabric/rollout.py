from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class PromotionThresholds:
    mandatory_coverage_drop_max_pp: float = 1.0
    latency_p95_ms_max: float = 150.0
    psi_mandatory_max: float = 0.2
    psi_optional_max: float = 0.3
    offline_quality_regression_max_pct: float = 1.0


def evaluate_shadow_promotion(
    *,
    baseline: Dict[str, float],
    shadow: Dict[str, float],
    thresholds: PromotionThresholds | None = None,
) -> Tuple[bool, Dict[str, str]]:
    cfg = thresholds or PromotionThresholds()
    failures: Dict[str, str] = {}

    coverage_drop_pp = (baseline.get("mandatory_coverage_rate", 0.0) - shadow.get("mandatory_coverage_rate", 0.0)) * 100.0
    if coverage_drop_pp > cfg.mandatory_coverage_drop_max_pp:
        failures["mandatory_coverage"] = (
            f"drop {coverage_drop_pp:.3f}pp exceeds {cfg.mandatory_coverage_drop_max_pp:.3f}pp"
        )

    if shadow.get("latency_p95_ms", 0.0) > cfg.latency_p95_ms_max:
        failures["latency_p95_ms"] = (
            f"{shadow.get('latency_p95_ms', 0.0):.3f} > {cfg.latency_p95_ms_max:.3f}"
        )

    if shadow.get("psi_mandatory", 0.0) > cfg.psi_mandatory_max:
        failures["psi_mandatory"] = (
            f"{shadow.get('psi_mandatory', 0.0):.3f} > {cfg.psi_mandatory_max:.3f}"
        )
    if shadow.get("psi_optional", 0.0) > cfg.psi_optional_max:
        failures["psi_optional"] = (
            f"{shadow.get('psi_optional', 0.0):.3f} > {cfg.psi_optional_max:.3f}"
        )

    base_quality = baseline.get("offline_quality_metric", 0.0)
    shadow_quality = shadow.get("offline_quality_metric", 0.0)
    if base_quality > 0:
        regression_pct = max(0.0, ((base_quality - shadow_quality) / base_quality) * 100.0)
        if regression_pct > cfg.offline_quality_regression_max_pct:
            failures["offline_quality_metric"] = (
                f"regression {regression_pct:.3f}% exceeds {cfg.offline_quality_regression_max_pct:.3f}%"
            )

    return len(failures) == 0, failures
