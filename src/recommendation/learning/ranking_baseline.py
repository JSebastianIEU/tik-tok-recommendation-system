from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .baseline_common import (
    DEFAULT_RANKING_WEIGHTS,
    OBJECTIVE_RANKING_WEIGHTS,
    as_float,
    as_int,
    clamp,
    round_score,
    sanitize_probability,
)
from .retrieval_baseline import (
    audience_compatibility,
    content_type_compatibility,
    cta_alignment,
    locale_compatibility,
    objective_compatibility,
)


def support_confidence_score(level: str, score: float) -> float:
    tier_floor = 0.82 if level == "full" else 0.52 if level == "partial" else 0.0
    return round_score(clamp((tier_floor * 0.55) + (score * 0.45), 0.0, 1.0), 6)


def freshness_score(posted_at: Optional[datetime], reference_date: datetime) -> float:
    if posted_at is None:
        return 0.55
    age_days = max(0.0, (reference_date - posted_at).total_seconds() / 86400.0)
    return round_score(clamp(math.exp((-math.log(2.0) * age_days) / 60.0), 0.0, 1.0), 6)


def reference_usefulness(candidate: Dict[str, Any], reference_date: datetime) -> float:
    comment_trace = candidate["comment_trace"]
    performance_quality = sanitize_probability(candidate["support_score"], 0.0)
    freshness = freshness_score(candidate.get("posted_at"), reference_date)
    comment_richness = sanitize_probability(comment_trace.get("value_prop_coverage"), 0.0)
    share_signal = sanitize_probability(comment_trace.get("on_topic_ratio"), 0.0)
    return round_score(
        clamp(
            (performance_quality * 0.40)
            + (freshness * 0.20)
            + (comment_richness * 0.20)
            + (share_signal * 0.10)
            + (candidate["support_score"] * 0.10),
            0.0,
            1.0,
        ),
        6,
    )


def score_components_for_candidate(
    *,
    query_profile: Dict[str, Any],
    candidate: Dict[str, Any],
    reference_date: datetime,
) -> Dict[str, float]:
    semantic_relevance = round_score(
        clamp(
            (candidate["retrieval_branch_scores"]["semantic"] * 0.70)
            + (candidate["retrieval_branch_scores"]["hashtag_topic"] * 0.30),
            0.0,
            1.0,
        ),
        6,
    )
    intent_alignment = round_score(
        clamp(
            (
                objective_compatibility(
                    query_profile["objective"], candidate["objective_guess"]
                )
                * 0.25
            )
            + (
                content_type_compatibility(
                    query_profile["content_type"], candidate["content_type"]
                )
                * 0.30
            )
            + (cta_alignment(query_profile["primary_cta"], candidate["text"]) * 0.20)
            + (
                audience_compatibility(query_profile["audience"], candidate["audience_tokens"])
                * 0.15
            )
            + (
                locale_compatibility(
                    query_profile["locale"],
                    query_profile["language"],
                    candidate["locale"],
                    candidate["language"],
                )
                * 0.10
            ),
            0.0,
            1.0,
        ),
        6,
    )
    return {
        "semantic_relevance": semantic_relevance,
        "intent_alignment": intent_alignment,
        "reference_usefulness": reference_usefulness(candidate, reference_date),
        "support_confidence": support_confidence_score(
            candidate["support_level"], candidate["support_score"]
        ),
    }


def ranking_reasons(candidate: Dict[str, Any], score_components: Dict[str, float]) -> List[str]:
    ordered = sorted(score_components.items(), key=lambda item: item[1], reverse=True)
    reasons = [f"strong_{name}" for name, _ in ordered[:2]]
    if len(candidate["retrieval_branches"]) >= 2:
        reasons.append("multi_branch_retrieval_match")
    if candidate["support_level"] == "full":
        reasons.append("fully_supported_reference")
    return reasons


def rank_shortlist(
    *,
    shortlist: Sequence[Dict[str, Any]],
    query_profile: Dict[str, Any],
    effective_objective: str,
    portfolio: Optional[Dict[str, Any]],
    rankers_available: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    reference_date = max(
        [item["posted_at"] for item in shortlist if isinstance(item.get("posted_at"), datetime)]
        or [datetime.now(timezone.utc)]
    )
    ranking_weights = OBJECTIVE_RANKING_WEIGHTS.get(
        effective_objective, DEFAULT_RANKING_WEIGHTS
    )
    ranked: List[Dict[str, Any]] = []
    for item in shortlist:
        components = score_components_for_candidate(
            query_profile=query_profile,
            candidate=item,
            reference_date=reference_date,
        )
        raw_score = round_score(
            clamp(
                (components["semantic_relevance"] * ranking_weights["semantic_relevance"])
                + (components["intent_alignment"] * ranking_weights["intent_alignment"])
                + (components["reference_usefulness"] * ranking_weights["reference_usefulness"])
                + (components["support_confidence"] * ranking_weights["support_confidence"]),
                0.0,
                1.0,
            ),
            6,
        )
        normalized_components = {
            key: sanitize_probability(value, 0.0) for key, value in components.items()
        }
        ranked.append(
            {
                **item,
                "score_components": normalized_components,
                "score_raw": raw_score,
                "score_calibrated": raw_score,
                "score": raw_score,
                "score_mean": raw_score,
                "score_std": 0.0,
                "confidence": normalized_components["support_confidence"],
                "global_score_mean": raw_score,
                "segment_blend_weight": 0.0,
                "selected_ranker_id": "baseline_weighted",
                "policy_penalty": 0.0,
                "policy_bonus": 0.0,
                "policy_adjusted_score": raw_score,
                "calibration_trace": {
                    "score_raw": raw_score,
                    "score_calibrated": raw_score,
                    "calibrator_segment_id": "baseline_identity",
                    "requested_segment_id": "baseline_weighted",
                    "calibrator_method": "identity",
                    "calibrator_support_count": 0,
                    "calibration_fallback_used": False,
                },
                "policy_trace": {
                    "policy_version": "policy.baseline.v1",
                    "applied_rules": [],
                },
                "ranking_reasons": ranking_reasons(item, normalized_components),
            }
        )

    portfolio_payload = portfolio if isinstance(portfolio, dict) else {}
    portfolio_requested = bool(portfolio_payload.get("enabled", False))
    available_rankers = set(str(item) for item in rankers_available)
    portfolio_supported = portfolio_requested and "reach" in available_rankers and "conversion" in available_rankers
    portfolio_fallback_reason = None
    if portfolio_requested and not portfolio_supported:
        portfolio_fallback_reason = next(
            (
                f"missing_ranker_{objective_id}"
                for objective_id in ("reach", "conversion")
                if objective_id not in available_rankers
            ),
            "portfolio_unavailable",
        )

    portfolio_weights_raw = (
        portfolio_payload.get("weights")
        if isinstance(portfolio_payload.get("weights"), dict)
        else {}
    )
    reach_weight = max(0.0, as_float(portfolio_weights_raw.get("reach"), 0.45))
    conversion_weight = max(0.0, as_float(portfolio_weights_raw.get("conversion"), 0.35))
    durability_weight = max(0.0, as_float(portfolio_weights_raw.get("durability"), 0.20))
    total_weight = max(1e-9, reach_weight + conversion_weight + durability_weight)
    portfolio_weights = {
        "reach": reach_weight / total_weight,
        "conversion": conversion_weight / total_weight,
        "durability": durability_weight / total_weight,
    }
    risk_aversion = max(0.0, as_float(portfolio_payload.get("risk_aversion"), 0.10))

    for item in ranked:
        utility_before = float(item["score"])
        if portfolio_supported:
            reach_score = item["score_components"]["semantic_relevance"]
            conversion_score = item["score_components"]["intent_alignment"]
            durability_score = item["score_components"]["reference_usefulness"]
            utility_before = round_score(
                (portfolio_weights["reach"] * reach_score)
                + (portfolio_weights["conversion"] * conversion_score)
                + (portfolio_weights["durability"] * durability_score)
                - (risk_aversion * max(0.0, 1.0 - item["confidence"])),
                6,
            )
            item["portfolio_trace"] = {
                "portfolio_version": "policy.baseline.v1",
                "weights": {key: round_score(value, 6) for key, value in portfolio_weights.items()},
                "risk_aversion": round_score(risk_aversion, 6),
                "utility_before_policy": utility_before,
                "utility_after_policy": utility_before,
                "components": {
                    "reach_score": reach_score,
                    "conversion_score": conversion_score,
                    "durability_score": durability_score,
                },
            }
        item["portfolio_utility"] = utility_before

    ranked.sort(
        key=lambda item: (
            -(float(item["portfolio_utility"]) if portfolio_supported else float(item["score"])),
            -float(item["score"]),
            -float(item["retrieval_branch_scores"]["fused_retrieval"]),
        )
    )

    return ranked, {
        "weights": ranking_weights,
        "portfolio_requested": portfolio_requested,
        "portfolio_supported": portfolio_supported,
        "portfolio_fallback_reason": portfolio_fallback_reason,
        "portfolio_weights": portfolio_weights,
        "risk_aversion": risk_aversion,
        "candidate_pool_cap": as_int(portfolio_payload.get("candidate_pool_cap"), 120),
    }
