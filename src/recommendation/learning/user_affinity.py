from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .baseline_common import (
    as_float,
    as_int,
    clamp,
    jaccard,
    normalize_text,
    round_score,
    sanitize_probability,
    tokenize,
)


USER_AFFINITY_VERSION = "user_affinity.v1"
USER_AFFINITY_MAX_SCORE_SHIFT = 0.08
USER_AFFINITY_MIN_QUERY_GUARD = 0.18
USER_AFFINITY_FULL_CONFIDENCE_EVENT_COUNT = 24.0
USER_AFFINITY_FULL_CONFIDENCE_REQUEST_COUNT = 8.0


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
                timezone.utc
            )
        except ValueError:
            return None
    return None


def _normalize_weight_map(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, float] = {}
    for raw_key, raw_weight in value.items():
        key = normalize_text(raw_key).lstrip("#@")
        if not key:
            continue
        weight = max(0.0, as_float(raw_weight, 0.0))
        if weight <= 0.0:
            continue
        out[key] = round_score(weight, 6)
    return out


def _normalize_token_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return tokenize(value)
    if isinstance(value, str):
        return tokenize([value])
    return []


def _weighted_match_score(
    *,
    candidate_tokens: Sequence[str],
    positive_weights: Dict[str, float],
    negative_weights: Dict[str, float],
) -> float:
    candidate_set = {normalize_text(token).lstrip("#@") for token in candidate_tokens if normalize_text(token)}
    candidate_set.discard("")
    if not candidate_set:
        return 0.5
    positive_total = sum(positive_weights.values()) or 0.0
    negative_total = sum(negative_weights.values()) or 0.0
    positive_hits = sum(weight for token, weight in positive_weights.items() if token in candidate_set)
    negative_hits = sum(weight for token, weight in negative_weights.items() if token in candidate_set)
    positive_score = 0.0 if positive_total <= 0.0 else positive_hits / positive_total
    negative_score = 0.0 if negative_total <= 0.0 else negative_hits / negative_total
    return sanitize_probability(0.5 + ((positive_score - negative_score) * 0.5), 0.5)


def _value_affinity_score(
    *,
    candidate_value: Any,
    positive_weights: Dict[str, float],
    negative_weights: Dict[str, float],
) -> float:
    normalized = normalize_text(candidate_value).lstrip("#@")
    if not normalized:
        return 0.5
    positive_total = sum(positive_weights.values()) or 0.0
    negative_total = sum(negative_weights.values()) or 0.0
    positive_score = (
        0.0 if positive_total <= 0.0 else positive_weights.get(normalized, 0.0) / positive_total
    )
    negative_score = (
        0.0 if negative_total <= 0.0 else negative_weights.get(normalized, 0.0) / negative_total
    )
    return sanitize_probability(0.5 + ((positive_score - negative_score) * 0.5), 0.5)


def _preference_block(
    user_context: Dict[str, Any],
    key: str,
) -> Dict[str, Dict[str, float]]:
    block = user_context.get(key)
    if not isinstance(block, dict):
        return {
            "topics_positive": {},
            "topics_negative": {},
            "content_types_positive": {},
            "content_types_negative": {},
            "hashtags_positive": {},
            "hashtags_negative": {},
            "authors_positive": {},
            "authors_negative": {},
        }
    return {
        "topics_positive": _normalize_weight_map(block.get("topics_positive")),
        "topics_negative": _normalize_weight_map(block.get("topics_negative")),
        "content_types_positive": _normalize_weight_map(block.get("content_types_positive")),
        "content_types_negative": _normalize_weight_map(block.get("content_types_negative")),
        "hashtags_positive": _normalize_weight_map(block.get("hashtags_positive")),
        "hashtags_negative": _normalize_weight_map(block.get("hashtags_negative")),
        "authors_positive": _normalize_weight_map(block.get("authors_positive")),
        "authors_negative": _normalize_weight_map(block.get("authors_negative")),
    }


def _merge_preference_blocks(
    primary: Dict[str, Dict[str, float]],
    fallback: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key in primary.keys() | fallback.keys():
        merged: Dict[str, float] = {}
        for token, weight in fallback.get(key, {}).items():
            merged[token] = merged.get(token, 0.0) + float(weight)
        for token, weight in primary.get(key, {}).items():
            merged[token] = merged.get(token, 0.0) + float(weight)
        out[key] = {token: round_score(weight, 6) for token, weight in merged.items() if weight > 0.0}
    return out


def build_user_affinity_context(
    *,
    user_context: Optional[Dict[str, Any]],
    effective_objective: str,
    as_of: datetime,
) -> Dict[str, Any]:
    payload = user_context if isinstance(user_context, dict) else {}
    creator_id = str(payload.get("creator_id") or "").strip().lower()
    support = payload.get("support") if isinstance(payload.get("support"), dict) else {}
    explicit_positive_count = max(0, as_int(support.get("explicit_positive_count"), 0))
    explicit_negative_count = max(0, as_int(support.get("explicit_negative_count"), 0))
    explicit_request_count = max(0, as_int(support.get("explicit_request_count"), 0))
    objective_request_count = max(0, as_int(support.get("objective_request_count"), 0))
    last_feedback_at = _parse_dt(payload.get("last_feedback_at"))
    days_since_last_feedback = (
        max(0.0, (as_of - last_feedback_at).total_seconds() / 86400.0)
        if last_feedback_at is not None
        else 999.0
    )
    recency_factor = clamp(1.0 - (days_since_last_feedback / 180.0), 0.0, 1.0)
    explicit_total = explicit_positive_count + explicit_negative_count
    event_factor = clamp(explicit_total / USER_AFFINITY_FULL_CONFIDENCE_EVENT_COUNT, 0.0, 1.0)
    request_factor = clamp(
        max(explicit_request_count, objective_request_count)
        / USER_AFFINITY_FULL_CONFIDENCE_REQUEST_COUNT,
        0.0,
        1.0,
    )
    balance_factor = 0.0
    if explicit_positive_count > 0 and explicit_negative_count > 0:
        balance_factor = min(explicit_positive_count, explicit_negative_count) / max(
            explicit_positive_count, explicit_negative_count
        )
    elif explicit_positive_count > 0:
        balance_factor = 0.35
    objective_bonus = 1.0
    built_for_objective = str(payload.get("objective_effective") or "").strip()
    if built_for_objective and built_for_objective != effective_objective:
        objective_bonus = 0.85
    confidence = sanitize_probability(
        (event_factor * 0.45)
        + (request_factor * 0.20)
        + (balance_factor * 0.20)
        + (recency_factor * 0.15),
        0.0,
    )
    confidence = sanitize_probability(confidence * objective_bonus, 0.0)

    objective_preferences = _preference_block(payload, "objective_preferences")
    global_preferences = _preference_block(payload, "global_preferences")
    merged_preferences = _merge_preference_blocks(objective_preferences, global_preferences)

    return {
        "enabled": bool(creator_id),
        "creator_id": creator_id or None,
        "objective_effective": effective_objective,
        "explicit_positive_count": explicit_positive_count,
        "explicit_negative_count": explicit_negative_count,
        "explicit_request_count": explicit_request_count,
        "objective_request_count": objective_request_count,
        "days_since_last_feedback": round_score(days_since_last_feedback, 4),
        "confidence": confidence,
        "preferences": merged_preferences,
        "global_preferences_present": any(bool(values) for values in global_preferences.values()),
        "objective_preferences_present": any(
            bool(values) for values in objective_preferences.values()
        ),
        "version": USER_AFFINITY_VERSION,
    }


def score_candidate_user_affinity(
    *,
    candidate: Dict[str, Any],
    affinity_context: Dict[str, Any],
) -> Dict[str, Any]:
    preferences = (
        affinity_context.get("preferences")
        if isinstance(affinity_context.get("preferences"), dict)
        else {}
    )
    candidate_topic = str(candidate.get("topic_key") or "")
    candidate_content_type = str(candidate.get("content_type") or "")
    candidate_author_id = str(candidate.get("author_id") or "")
    candidate_hashtags = _normalize_token_list(candidate.get("hashtags"))
    candidate_keywords = _normalize_token_list(candidate.get("keywords"))
    candidate_topic_tokens = tokenize([candidate_topic, *candidate_hashtags, *candidate_keywords])
    topic_score = _weighted_match_score(
        candidate_tokens=candidate_topic_tokens,
        positive_weights=preferences.get("topics_positive", {}),
        negative_weights=preferences.get("topics_negative", {}),
    )
    hashtag_score = _weighted_match_score(
        candidate_tokens=[*candidate_hashtags, *candidate_keywords],
        positive_weights=preferences.get("hashtags_positive", {}),
        negative_weights=preferences.get("hashtags_negative", {}),
    )
    content_type_score = _value_affinity_score(
        candidate_value=candidate_content_type,
        positive_weights=preferences.get("content_types_positive", {}),
        negative_weights=preferences.get("content_types_negative", {}),
    )
    author_score = _value_affinity_score(
        candidate_value=candidate_author_id,
        positive_weights=preferences.get("authors_positive", {}),
        negative_weights=preferences.get("authors_negative", {}),
    )
    affinity_score = sanitize_probability(
        (topic_score * 0.42)
        + (hashtag_score * 0.28)
        + (content_type_score * 0.20)
        + (author_score * 0.10),
        0.5,
    )
    return {
        "score": affinity_score,
        "components": {
            "topic": round_score(topic_score, 6),
            "hashtags": round_score(hashtag_score, 6),
            "content_type": round_score(content_type_score, 6),
            "author": round_score(author_score, 6),
        },
        "candidate_topic_tokens": candidate_topic_tokens,
    }


def apply_user_affinity_blend(
    *,
    items: Sequence[Dict[str, Any]],
    affinity_context: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    context = affinity_context if isinstance(affinity_context, dict) else {}
    if not items:
        return [], {
            "enabled": bool(context.get("enabled")),
            "applied": False,
            "reason": "empty_shortlist",
            "version": USER_AFFINITY_VERSION,
        }
    if not bool(context.get("enabled")):
        return [dict(item) for item in items], {
            "enabled": False,
            "applied": False,
            "reason": "missing_creator_context",
            "version": USER_AFFINITY_VERSION,
        }
    confidence = sanitize_probability(context.get("confidence"), 0.0)
    if confidence <= 0.01:
        out = []
        for item in items:
            updated = dict(item)
            updated["user_affinity_score"] = 0.5
            updated["user_affinity_trace"] = {
                "version": USER_AFFINITY_VERSION,
                "applied": False,
                "reason": "insufficient_profile_confidence",
                "confidence": round_score(confidence, 6),
            }
            out.append(updated)
        return out, {
            "enabled": True,
            "applied": False,
            "reason": "insufficient_profile_confidence",
            "confidence": round_score(confidence, 6),
            "version": USER_AFFINITY_VERSION,
        }

    reranked: List[Dict[str, Any]] = []
    max_abs_shift = USER_AFFINITY_MAX_SCORE_SHIFT * confidence
    for item in items:
        updated = dict(item)
        affinity = score_candidate_user_affinity(candidate=item, affinity_context=context)
        semantic_relevance = sanitize_probability(
            ((item.get("score_components") or {}).get("semantic_relevance") if isinstance(item.get("score_components"), dict) else 0.0),
            0.0,
        )
        query_guard = clamp(
            (semantic_relevance - USER_AFFINITY_MIN_QUERY_GUARD)
            / max(1e-6, 1.0 - USER_AFFINITY_MIN_QUERY_GUARD),
            0.0,
            1.0,
        )
        affinity_margin = (float(affinity["score"]) - 0.5) * 2.0
        score_shift = max_abs_shift * query_guard * affinity_margin
        baseline_score = sanitize_probability(updated.get("score"), 0.0)
        final_score = sanitize_probability(baseline_score + score_shift, baseline_score)
        updated["user_affinity_score"] = round_score(float(affinity["score"]), 6)
        updated["user_affinity_trace"] = {
            "version": USER_AFFINITY_VERSION,
            "applied": True,
            "creator_id": context.get("creator_id"),
            "confidence": round_score(confidence, 6),
            "query_guard": round_score(query_guard, 6),
            "score_shift": round_score(score_shift, 6),
            "components": affinity["components"],
        }
        updated["score_raw"] = round_score(final_score, 6)
        updated["score_calibrated"] = round_score(final_score, 6)
        updated["score"] = round_score(final_score, 6)
        updated["score_mean"] = round_score(final_score, 6)
        updated["global_score_mean"] = round_score(final_score, 6)
        updated["segment_blend_weight"] = round_score(confidence, 6)
        updated["selected_ranker_id"] = str(updated.get("selected_ranker_id") or "baseline_weighted")
        updated["confidence"] = round_score(
            max(
                sanitize_probability(updated.get("confidence"), 0.0),
                confidence * query_guard,
            ),
            6,
        )
        reranked.append(updated)

    reranked.sort(
        key=lambda item: (
            -float(item.get("score", 0.0) or 0.0),
            -float(item.get("baseline_score", item.get("score", 0.0)) or 0.0),
            -float(
                (
                    (item.get("retrieval_branch_scores") or {}).get("fused_retrieval")
                    if isinstance(item.get("retrieval_branch_scores"), dict)
                    else 0.0
                )
                or 0.0
            ),
        )
    )
    return reranked, {
        "enabled": True,
        "applied": True,
        "reason": None,
        "version": USER_AFFINITY_VERSION,
        "creator_id": context.get("creator_id"),
        "confidence": round_score(confidence, 6),
        "max_score_shift": round_score(max_abs_shift, 6),
        "global_preferences_present": bool(context.get("global_preferences_present")),
        "objective_preferences_present": bool(context.get("objective_preferences_present")),
        "explicit_positive_count": int(context.get("explicit_positive_count") or 0),
        "explicit_negative_count": int(context.get("explicit_negative_count") or 0),
    }


__all__ = [
    "USER_AFFINITY_VERSION",
    "build_user_affinity_context",
    "score_candidate_user_affinity",
    "apply_user_affinity_blend",
]
