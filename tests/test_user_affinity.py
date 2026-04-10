from __future__ import annotations

from datetime import datetime, timezone

from src.recommendation.learning import (
    USER_AFFINITY_VERSION,
    apply_user_affinity_blend,
    build_user_affinity_context,
    score_candidate_user_affinity,
)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _base_item(
    *,
    candidate_id: str,
    score: float,
    topic_key: str,
    content_type: str,
    author_id: str,
    hashtags: list[str],
    semantic_relevance: float = 0.82,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "score": score,
        "score_raw": score,
        "score_calibrated": score,
        "score_mean": score,
        "global_score_mean": score,
        "baseline_score": score,
        "segment_blend_weight": 0.0,
        "selected_ranker_id": "baseline_weighted",
        "confidence": 0.52,
        "topic_key": topic_key,
        "content_type": content_type,
        "author_id": author_id,
        "hashtags": hashtags,
        "keywords": hashtags,
        "policy_penalty": 0.0,
        "policy_bonus": 0.0,
        "policy_adjusted_score": score,
        "retrieval_branch_scores": {"fused_retrieval": score},
        "score_components": {
            "semantic_relevance": semantic_relevance,
            "intent_alignment": 0.61,
            "reference_usefulness": 0.58,
            "support_confidence": 0.57,
        },
    }


def test_build_user_affinity_context_uses_recency_support_and_objective_scope():
    context = build_user_affinity_context(
        user_context={
            "creator_id": "Creator-7",
            "objective_effective": "engagement",
            "last_feedback_at": "2026-04-01T00:00:00Z",
            "support": {
                "explicit_positive_count": 8,
                "explicit_negative_count": 5,
                "explicit_request_count": 6,
                "objective_request_count": 4,
            },
            "global_preferences": {
                "topics_positive": {"tech": 2.0},
                "hashtags_positive": {"iphone": 1.0},
            },
            "objective_preferences": {
                "topics_positive": {"reviews": 1.5},
                "content_types_positive": {"review": 1.0},
            },
        },
        effective_objective="engagement",
        as_of=_dt("2026-04-06T00:00:00Z"),
    )
    assert context["enabled"] is True
    assert context["creator_id"] == "creator-7"
    assert context["objective_preferences_present"] is True
    assert context["global_preferences_present"] is True
    assert 0.0 < float(context["confidence"]) <= 1.0
    assert context["preferences"]["topics_positive"]["tech"] > 0.0
    assert context["preferences"]["topics_positive"]["reviews"] > 0.0
    assert context["version"] == USER_AFFINITY_VERSION


def test_score_candidate_user_affinity_reflects_positive_and_negative_memory():
    context = build_user_affinity_context(
        user_context={
            "creator_id": "creator-1",
            "objective_effective": "engagement",
            "last_feedback_at": "2026-04-04T00:00:00Z",
            "support": {
                "explicit_positive_count": 10,
                "explicit_negative_count": 7,
                "explicit_request_count": 7,
                "objective_request_count": 5,
            },
            "global_preferences": {
                "topics_positive": {"tech": 2.0},
                "topics_negative": {"dance": 2.0},
                "content_types_positive": {"review": 1.0},
                "content_types_negative": {"challenge": 1.0},
                "hashtags_positive": {"iphone": 1.0},
                "hashtags_negative": {"trend": 1.0},
                "authors_positive": {"creator-a": 1.0},
                "authors_negative": {"creator-b": 1.0},
            },
        },
        effective_objective="engagement",
        as_of=_dt("2026-04-06T00:00:00Z"),
    )
    positive = score_candidate_user_affinity(
        candidate={
            "topic_key": "tech",
            "content_type": "review",
            "author_id": "creator-a",
            "hashtags": ["iphone", "tech"],
            "keywords": ["iphone"],
        },
        affinity_context=context,
    )
    negative = score_candidate_user_affinity(
        candidate={
            "topic_key": "dance",
            "content_type": "challenge",
            "author_id": "creator-b",
            "hashtags": ["trend"],
            "keywords": ["trend"],
        },
        affinity_context=context,
    )
    assert positive["score"] > 0.5
    assert negative["score"] < 0.5
    assert positive["components"]["topic"] > negative["components"]["topic"]


def test_apply_user_affinity_blend_is_query_first_and_confidence_gated():
    context = build_user_affinity_context(
        user_context={
            "creator_id": "creator-9",
            "objective_effective": "engagement",
            "last_feedback_at": "2026-04-05T00:00:00Z",
            "support": {
                "explicit_positive_count": 18,
                "explicit_negative_count": 10,
                "explicit_request_count": 12,
                "objective_request_count": 9,
            },
            "global_preferences": {
                "topics_positive": {"tech": 3.0},
                "topics_negative": {"dance": 3.0},
                "content_types_positive": {"review": 2.0},
                "content_types_negative": {"challenge": 2.0},
                "hashtags_positive": {"iphone": 2.0},
                "hashtags_negative": {"trend": 2.0},
            },
        },
        effective_objective="engagement",
        as_of=_dt("2026-04-06T00:00:00Z"),
    )
    strong_match = _base_item(
        candidate_id="c1",
        score=0.62,
        topic_key="tech",
        content_type="review",
        author_id="creator-a",
        hashtags=["iphone", "review"],
        semantic_relevance=0.84,
    )
    weak_match = _base_item(
        candidate_id="c2",
        score=0.61,
        topic_key="dance",
        content_type="challenge",
        author_id="creator-b",
        hashtags=["trend", "dance"],
        semantic_relevance=0.83,
    )
    reranked, meta = apply_user_affinity_blend(
        items=[weak_match, strong_match],
        affinity_context=context,
    )
    assert meta["applied"] is True
    assert reranked[0]["candidate_id"] == "c1"
    assert reranked[0]["user_affinity_score"] > reranked[1]["user_affinity_score"]
    assert reranked[0]["score"] > reranked[0]["baseline_score"]
    assert reranked[1]["score"] < reranked[1]["baseline_score"]

    low_query_item = _base_item(
        candidate_id="c3",
        score=0.40,
        topic_key="tech",
        content_type="review",
        author_id="creator-a",
        hashtags=["iphone"],
        semantic_relevance=0.05,
    )
    low_query_ranked, _ = apply_user_affinity_blend(
        items=[low_query_item],
        affinity_context=context,
    )
    assert abs(low_query_ranked[0]["score"] - low_query_ranked[0]["baseline_score"]) < 1e-6


def test_apply_user_affinity_blend_abstains_without_confidence():
    context = build_user_affinity_context(
        user_context={
            "creator_id": "creator-0",
            "objective_effective": "engagement",
            "support": {
                "explicit_positive_count": 0,
                "explicit_negative_count": 0,
                "explicit_request_count": 0,
                "objective_request_count": 0,
            },
        },
        effective_objective="engagement",
        as_of=_dt("2026-04-06T00:00:00Z"),
    )
    items, meta = apply_user_affinity_blend(
        items=[
            _base_item(
                candidate_id="c1",
                score=0.7,
                topic_key="tech",
                content_type="review",
                author_id="creator-a",
                hashtags=["iphone"],
            )
        ],
        affinity_context=context,
    )
    assert meta["applied"] is False
    assert meta["reason"] == "insufficient_profile_confidence"
    assert items[0]["user_affinity_trace"]["applied"] is False
