from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple


POLICY_RERANK_VERSION = "policy_reranker.v2"
PORTFOLIO_SELECTION_VERSION = "portfolio_selection.v2"


def _to_utc_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _norm_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


@dataclass
class PolicyRerankerConfig:
    version: str = POLICY_RERANK_VERSION
    max_items_per_author: int = 2
    require_language_match: bool = False
    require_locale_match: bool = False
    max_candidate_age_days: int = 180
    diversity_topic_weight: float = 0.05
    diversity_author_weight: float = 0.03
    freshness_bonus_weight: float = 0.04
    freshness_half_life_hours: float = 168.0
    locale_fit_bonus: float = 0.02

    @classmethod
    def from_payload(cls, payload: Optional[Dict[str, Any]]) -> "PolicyRerankerConfig":
        if not isinstance(payload, dict):
            return cls()
        strict_language = payload.get("strict_language")
        strict_locale = payload.get("strict_locale")
        require_language_match = (
            bool(strict_language)
            if strict_language is not None
            else bool(payload.get("require_language_match", False))
        )
        require_locale_match = (
            bool(strict_locale)
            if strict_locale is not None
            else bool(payload.get("require_locale_match", False))
        )
        return cls(
            version=str(payload.get("version") or POLICY_RERANK_VERSION),
            max_items_per_author=max(1, int(payload.get("max_items_per_author") or 2)),
            require_language_match=require_language_match,
            require_locale_match=require_locale_match,
            max_candidate_age_days=max(1, int(payload.get("max_candidate_age_days") or 180)),
            diversity_topic_weight=float(payload.get("diversity_topic_weight") or 0.05),
            diversity_author_weight=float(payload.get("diversity_author_weight") or 0.03),
            freshness_bonus_weight=float(payload.get("freshness_bonus_weight") or 0.04),
            freshness_half_life_hours=max(
                1.0,
                float(payload.get("freshness_half_life_hours") or 168.0),
            ),
            locale_fit_bonus=float(payload.get("locale_fit_bonus") or 0.02),
        )

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioSelectionConfig:
    version: str = PORTFOLIO_SELECTION_VERSION
    enabled: bool = False
    reach_weight: float = 0.45
    conversion_weight: float = 0.35
    durability_weight: float = 0.20
    risk_aversion: float = 0.10
    candidate_pool_cap: int = 120
    fallback_reason: Optional[str] = None

    @classmethod
    def from_payload(
        cls, payload: Optional[Dict[str, Any]]
    ) -> "PortfolioSelectionConfig":
        if not isinstance(payload, dict):
            return cls()
        weights = payload.get("weights") if isinstance(payload.get("weights"), dict) else {}
        return cls(
            version=str(payload.get("version") or PORTFOLIO_SELECTION_VERSION),
            enabled=bool(payload.get("enabled", False)),
            reach_weight=float(weights.get("reach") or 0.45),
            conversion_weight=float(weights.get("conversion") or 0.35),
            durability_weight=float(weights.get("durability") or 0.20),
            risk_aversion=float(payload.get("risk_aversion") or 0.10),
            candidate_pool_cap=max(1, int(payload.get("candidate_pool_cap") or 120)),
            fallback_reason=str(payload.get("fallback_reason") or "").strip() or None,
        )

    def normalized_weights(self) -> Dict[str, float]:
        raw = {
            "reach": max(0.0, float(self.reach_weight)),
            "conversion": max(0.0, float(self.conversion_weight)),
            "durability": max(0.0, float(self.durability_weight)),
        }
        total = raw["reach"] + raw["conversion"] + raw["durability"]
        if total <= 1e-6:
            return {"reach": 0.45, "conversion": 0.35, "durability": 0.20}
        return {
            "reach": raw["reach"] / total,
            "conversion": raw["conversion"] / total,
            "durability": raw["durability"] / total,
        }


class PolicyReranker:
    def __init__(self, config: Optional[PolicyRerankerConfig] = None) -> None:
        self.config = config or PolicyRerankerConfig()

    def rerank(
        self,
        *,
        ranked_items: Sequence[Dict[str, Any]],
        query_context: Dict[str, Any],
        top_k: int,
        overrides: Optional[Dict[str, Any]] = None,
        portfolio: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        config = PolicyRerankerConfig.from_payload(
            {
                **self.config.to_payload(),
                **(overrides or {}),
            }
        )
        portfolio_config = PortfolioSelectionConfig.from_payload(portfolio)
        portfolio_weights = portfolio_config.normalized_weights()
        portfolio_mode = bool(portfolio_config.enabled)
        portfolio_fallback_reason: Optional[str] = portfolio_config.fallback_reason
        if portfolio_mode and portfolio_fallback_reason:
            portfolio_mode = False
        query_as_of = _to_utc_dt(query_context.get("as_of_time")) or datetime.now(timezone.utc)
        query_language = _norm_text(query_context.get("language"))
        query_locale = _norm_text(query_context.get("locale"))
        selected: List[Dict[str, Any]] = []
        remaining = [dict(item) for item in ranked_items]
        candidate_pool_size = len(remaining)
        candidates_trimmed = 0
        if portfolio_mode and len(remaining) > portfolio_config.candidate_pool_cap:
            candidates_trimmed = len(remaining) - portfolio_config.candidate_pool_cap
            remaining = remaining[: portfolio_config.candidate_pool_cap]

        author_counts: Dict[str, int] = {}
        topic_counts: Dict[str, int] = {}
        dropped_by_rule: Dict[str, int] = {
            "author_cap": 0,
            "language_mismatch": 0,
            "locale_mismatch": 0,
            "age_limit": 0,
            "portfolio_component_missing": 0,
        }
        first_rejection_by_candidate: Dict[str, str] = {}
        portfolio_missing_component_count = 0

        def _hard_constraint_reason(item: Dict[str, Any]) -> Optional[str]:
            author_id = str(item.get("_author_id") or "unknown")
            if author_counts.get(author_id, 0) >= config.max_items_per_author:
                return "author_cap"
            candidate_language = _norm_text(item.get("_language"))
            candidate_locale = _norm_text(item.get("_locale"))
            if (
                config.require_language_match
                and query_language
                and candidate_language
                and candidate_language != query_language
            ):
                return "language_mismatch"
            if (
                config.require_locale_match
                and query_locale
                and candidate_locale
                and candidate_locale != query_locale
            ):
                return "locale_mismatch"
            candidate_as_of = _to_utc_dt(item.get("_candidate_as_of_time"))
            if candidate_as_of is not None:
                age_days = max(0.0, (query_as_of - candidate_as_of).total_seconds() / 86400.0)
                if age_days > float(config.max_candidate_age_days):
                    return "age_limit"
            return None

        def _portfolio_components(item: Dict[str, Any]) -> Optional[Dict[str, float]]:
            raw = item.get("portfolio_components")
            if not isinstance(raw, dict):
                return None
            try:
                reach = float(raw.get("reach_score"))
                conversion = float(raw.get("conversion_score"))
                durability = float(raw.get("durability_score"))
                uncertainty = float(raw.get("uncertainty_penalty") or 0.0)
            except (TypeError, ValueError):
                return None
            if not all(math.isfinite(v) for v in (reach, conversion, durability, uncertainty)):
                return None
            return {
                "reach_score": max(0.0, min(1.0, reach)),
                "conversion_score": max(0.0, min(1.0, conversion)),
                "durability_score": max(0.0, min(1.0, durability)),
                "uncertainty_penalty": max(0.0, min(1.0, uncertainty)),
            }

        while remaining and len(selected) < max(1, int(top_k)):
            candidates: List[Dict[str, Any]] = []
            for item in remaining:
                reason = _hard_constraint_reason(item)
                if reason:
                    candidate_key = str(item.get("candidate_row_id") or item.get("candidate_id") or "")
                    if candidate_key and candidate_key not in first_rejection_by_candidate:
                        first_rejection_by_candidate[candidate_key] = reason
                        dropped_by_rule[reason] = dropped_by_rule.get(reason, 0) + 1
                    continue
                candidates.append(item)
            if not candidates:
                break
            scored: List[Tuple[float, float, str, Dict[str, Any]]] = []
            for item in candidates:
                base = float(item.get("score_calibrated") or item.get("score") or 0.0)
                portfolio_trace: Optional[Dict[str, Any]] = None
                if portfolio_mode:
                    components = _portfolio_components(item)
                    if components is None:
                        candidate_key = str(
                            item.get("candidate_row_id") or item.get("candidate_id") or ""
                        )
                        if (
                            candidate_key
                            and candidate_key not in first_rejection_by_candidate
                        ):
                            first_rejection_by_candidate[
                                candidate_key
                            ] = "portfolio_component_missing"
                            dropped_by_rule["portfolio_component_missing"] = (
                                dropped_by_rule.get("portfolio_component_missing", 0) + 1
                            )
                            portfolio_missing_component_count += 1
                        continue
                    utility_before_policy = (
                        (portfolio_weights["reach"] * components["reach_score"])
                        + (portfolio_weights["conversion"] * components["conversion_score"])
                        + (portfolio_weights["durability"] * components["durability_score"])
                        - (
                            max(0.0, float(portfolio_config.risk_aversion))
                            * components["uncertainty_penalty"]
                        )
                    )
                    base = float(utility_before_policy)
                    portfolio_trace = {
                        "portfolio_version": portfolio_config.version,
                        "components": {
                            "reach_score": round(components["reach_score"], 6),
                            "conversion_score": round(components["conversion_score"], 6),
                            "durability_score": round(components["durability_score"], 6),
                            "uncertainty_penalty": round(
                                components["uncertainty_penalty"], 6
                            ),
                        },
                        "weights": {
                            "reach": round(portfolio_weights["reach"], 6),
                            "conversion": round(portfolio_weights["conversion"], 6),
                            "durability": round(portfolio_weights["durability"], 6),
                        },
                        "risk_aversion": round(
                            max(0.0, float(portfolio_config.risk_aversion)),
                            6,
                        ),
                        "utility_before_policy": round(float(utility_before_policy), 6),
                        "utility_after_policy": None,
                        "marginal_contribution": None,
                    }
                author_id = str(item.get("_author_id") or "unknown")
                topic_key = str(item.get("_topic_key") or "unknown")
                penalty = (
                    config.diversity_author_weight * float(author_counts.get(author_id, 0))
                    + config.diversity_topic_weight * float(topic_counts.get(topic_key, 0))
                )
                bonus = 0.0
                candidate_as_of = _to_utc_dt(item.get("_candidate_as_of_time"))
                if candidate_as_of is not None:
                    age_hours = max(0.0, (query_as_of - candidate_as_of).total_seconds() / 3600.0)
                    freshness = math.exp(-age_hours / max(1.0, config.freshness_half_life_hours))
                    bonus += config.freshness_bonus_weight * freshness
                candidate_language = _norm_text(item.get("_language"))
                candidate_locale = _norm_text(item.get("_locale"))
                locale_match = bool(
                    (query_locale and candidate_locale and query_locale == candidate_locale)
                    or (query_language and candidate_language and query_language == candidate_language)
                )
                if locale_match:
                    bonus += config.locale_fit_bonus
                adjusted = base + bonus - penalty
                candidate_id = str(item.get("candidate_id") or item.get("candidate_row_id") or "")
                with_policy = dict(item)
                with_policy["policy_penalty"] = float(penalty)
                with_policy["policy_bonus"] = float(bonus)
                with_policy["policy_adjusted_score"] = float(adjusted)
                if portfolio_trace is not None:
                    portfolio_trace["utility_after_policy"] = round(float(adjusted), 6)
                    with_policy["portfolio_trace"] = portfolio_trace
                    components = dict(portfolio_trace.get("components") or {})
                    components["base_portfolio_utility"] = round(float(base), 6)
                    with_policy["portfolio_components"] = components
                with_policy["policy_trace"] = {
                    "selected": False,
                    "selected_round": None,
                    "hard_constraints": {
                        "author_cap_passed": True,
                        "language_passed": True,
                        "locale_passed": True,
                        "age_limit_passed": True,
                    },
                    "soft_components": {
                        "diversity_author_penalty": round(
                            config.diversity_author_weight * float(author_counts.get(author_id, 0)),
                            6,
                        ),
                        "diversity_topic_penalty": round(
                            config.diversity_topic_weight * float(topic_counts.get(topic_key, 0)),
                            6,
                        ),
                        "freshness_bonus": round(
                            max(0.0, bonus - (config.locale_fit_bonus if locale_match else 0.0)),
                            6,
                        ),
                        "locale_fit_bonus": round(
                            config.locale_fit_bonus if locale_match else 0.0,
                            6,
                        ),
                    },
                    "decision": "eligible",
                }
                scored.append((adjusted, base, candidate_id, with_policy))
            if not scored:
                break
            scored.sort(
                key=lambda item: (
                    -item[0],  # adjusted desc
                    -item[1],  # calibrated desc
                    item[2],   # candidate id asc
                )
            )
            best_item = scored[0][3]
            best_item["policy_trace"] = {
                **(best_item.get("policy_trace") if isinstance(best_item.get("policy_trace"), dict) else {}),
                "selected": True,
                "selected_round": len(selected) + 1,
                "decision": "selected",
            }
            if isinstance(best_item.get("portfolio_trace"), dict):
                best_item["portfolio_trace"] = {
                    **best_item["portfolio_trace"],
                    "selected_round": len(selected) + 1,
                    "marginal_contribution": round(
                        float(best_item.get("policy_adjusted_score") or 0.0),
                        6,
                    ),
                }
            selected.append(best_item)
            author_key = str(best_item.get("_author_id") or "unknown")
            topic_key = str(best_item.get("_topic_key") or "unknown")
            author_counts[author_key] = author_counts.get(author_key, 0) + 1
            topic_counts[topic_key] = topic_counts.get(topic_key, 0) + 1
            selected_row_id = str(best_item.get("candidate_row_id") or "")
            remaining = [
                item
                for item in remaining
                if str(item.get("candidate_row_id") or "") != selected_row_id
            ]

        rejected_candidates: List[Dict[str, Any]] = []
        for item in ranked_items:
            candidate_key = str(item.get("candidate_row_id") or item.get("candidate_id") or "")
            if not candidate_key:
                continue
            if any(str(chosen.get("candidate_row_id") or chosen.get("candidate_id") or "") == candidate_key for chosen in selected):
                continue
            reason = first_rejection_by_candidate.get(candidate_key)
            if reason:
                rejected_candidates.append(
                    {
                        "candidate_id": str(item.get("candidate_id") or ""),
                        "candidate_row_id": str(item.get("candidate_row_id") or ""),
                        "reason": reason,
                    }
                )

        if portfolio_mode and len(selected) == 0 and not portfolio_fallback_reason:
            portfolio_fallback_reason = "empty_portfolio_selection"
            portfolio_mode = False

        return selected, {
            "policy_version": config.version,
            "max_items_per_author": config.max_items_per_author,
            "constraint_violations": dropped_by_rule,
            "dropped_by_rule": dropped_by_rule,
            "rejected_candidates": rejected_candidates,
            "selected_count": len(selected),
            "strict_language": bool(config.require_language_match),
            "strict_locale": bool(config.require_locale_match),
            "portfolio_mode": bool(portfolio_mode),
            "portfolio_metadata": {
                "version": portfolio_config.version,
                "enabled_requested": bool(portfolio_config.enabled),
                "enabled": bool(portfolio_mode),
                "weights_effective": {
                    "reach": round(portfolio_weights["reach"], 6),
                    "conversion": round(portfolio_weights["conversion"], 6),
                    "durability": round(portfolio_weights["durability"], 6),
                },
                "risk_aversion": round(max(0.0, float(portfolio_config.risk_aversion)), 6),
                "candidate_pool_size": candidate_pool_size,
                "candidate_pool_cap": int(portfolio_config.candidate_pool_cap),
                "candidates_trimmed": int(candidates_trimmed),
                "missing_component_count": int(portfolio_missing_component_count),
                "selected_count": len(selected),
                "fallback_reason": portfolio_fallback_reason,
            },
        }
