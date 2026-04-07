from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .baseline_common import (
    BASELINE_COMMENT_VERSION,
    COMMUNITY_TERMS,
    CONVERSION_TERMS,
    ENGAGEMENT_TERMS,
    SUPPORT_FULL_THRESHOLD,
    SUPPORT_PARTIAL_THRESHOLD,
    clamp,
    derive_language,
    normalize_locale,
    normalize_text,
    round_score,
    safe_text,
    sanitize_probability,
    tokenize,
)
from .query_contract import infer_topic_key, normalize_content_type
from .temporal import parse_dt
from ..semantic_processor import process_text


def infer_candidate_objective(payload: Dict[str, Any]) -> str:
    processed = process_text(
        text=" ".join(
            [
                safe_text(payload.get("text")),
                safe_text(payload.get("caption")),
                " ".join(str(item) for item in list(payload.get("keywords") or [])),
            ]
        ).strip(),
        explicit_hashtags=list(payload.get("hashtags") or []),
        explicit_mentions=list(payload.get("mentions") or []),
    )
    text = normalize_text(
        " ".join(
            [
                processed.lexical_text,
                " ".join(processed.hashtags),
                " ".join(str(item) for item in list(payload.get("keywords") or [])),
            ]
        )
    )
    if any(term in text for term in CONVERSION_TERMS):
        return "conversion"
    if any(term in text for term in COMMUNITY_TERMS):
        return "community"
    if any(term in text for term in ENGAGEMENT_TERMS):
        return "engagement"
    return "reach"


def default_comment_trace(text: str, query_tokens: List[str]) -> Dict[str, Any]:
    candidate_tokens = process_text(text=text).lexical_tokens or tokenize([text])
    overlap = 0.0
    if query_tokens and candidate_tokens:
        query_set = set(query_tokens)
        candidate_set = set(candidate_tokens)
        union = len(query_set | candidate_set)
        overlap = 0.0 if union == 0 else len(query_set & candidate_set) / union
    return {
        "source": "baseline_inferred",
        "available": False,
        "taxonomy_version": BASELINE_COMMENT_VERSION,
        "dominant_intents": [],
        "confusion_index": round_score(1.0 - overlap, 6),
        "help_seeking_index": round_score(min(1.0, overlap + 0.15), 6),
        "sentiment_volatility": 0.0,
        "sentiment_shift_early_late": 0.0,
        "reply_depth_max": 0.0,
        "reply_branch_factor": 0.0,
        "reply_ratio": 0.0,
        "root_thread_concentration": 0.0,
        "alignment_score": round_score(overlap, 6),
        "value_prop_coverage": round_score(min(1.0, overlap + 0.10), 6),
        "on_topic_ratio": round_score(min(1.0, overlap + 0.05), 6),
        "artifact_drift_ratio": round_score(max(0.0, 0.30 - overlap), 6),
        "alignment_shift_early_late": 0.0,
        "alignment_confidence": 0.35,
        "alignment_method_version": "baseline_overlap.v1",
        "confidence": 0.35,
        "missingness_flags": ["comment_intelligence_unavailable"],
    }


def coerce_comment_intelligence(payload: Dict[str, Any]) -> Dict[str, Any]:
    hints = payload.get("signal_hints")
    if not isinstance(hints, dict):
        return {}
    raw = hints.get("comment_intelligence")
    if not isinstance(raw, dict):
        return {}
    dominant = raw.get("dominant_intents")
    dominant_intents = (
        [str(item) for item in dominant if str(item).strip()]
        if isinstance(dominant, list)
        else []
    )
    return {
        "source": str(raw.get("source") or "request_hint"),
        "available": bool(raw.get("available", True)),
        "taxonomy_version": str(raw.get("taxonomy_version") or BASELINE_COMMENT_VERSION),
        "dominant_intents": dominant_intents,
        "confusion_index": sanitize_probability(raw.get("confusion_index"), 0.0),
        "help_seeking_index": sanitize_probability(raw.get("help_seeking_index"), 0.0),
        "sentiment_volatility": sanitize_probability(raw.get("sentiment_volatility"), 0.0),
        "sentiment_shift_early_late": float(raw.get("sentiment_shift_early_late") or 0.0),
        "reply_depth_max": float(raw.get("reply_depth_max") or 0.0),
        "reply_branch_factor": sanitize_probability(raw.get("reply_branch_factor"), 0.0),
        "reply_ratio": sanitize_probability(raw.get("reply_ratio"), 0.0),
        "root_thread_concentration": sanitize_probability(
            raw.get("root_thread_concentration"), 0.0
        ),
        "alignment_score": sanitize_probability(raw.get("alignment_score"), 0.0),
        "value_prop_coverage": sanitize_probability(raw.get("value_prop_coverage"), 0.0),
        "on_topic_ratio": sanitize_probability(raw.get("on_topic_ratio"), 0.0),
        "artifact_drift_ratio": sanitize_probability(raw.get("artifact_drift_ratio"), 0.0),
        "alignment_shift_early_late": float(raw.get("alignment_shift_early_late") or 0.0),
        "alignment_confidence": sanitize_probability(
            raw.get("alignment_confidence"), float(raw.get("confidence") or 0.35)
        ),
        "alignment_method_version": str(raw.get("alignment_method_version") or "request_hint"),
        "confidence": sanitize_probability(raw.get("confidence"), 0.35),
        "missingness_flags": [str(item) for item in list(raw.get("missingness_flags") or [])],
    }


def coerce_manifest_comment_intelligence(row: Dict[str, Any]) -> Dict[str, Any]:
    features = row.get("features")
    if not isinstance(features, dict):
        return {}
    missingness = row.get("missingness")
    missingness_flags = (
        sorted(str(key) for key in missingness.keys()) if isinstance(missingness, dict) else []
    )
    dominant = features.get("dominant_intents")
    dominant_intents = (
        [str(item) for item in dominant if str(item).strip()]
        if isinstance(dominant, list)
        else []
    )
    return {
        "source": "manifest_snapshot",
        "available": True,
        "taxonomy_version": str(row.get("taxonomy_version") or BASELINE_COMMENT_VERSION),
        "dominant_intents": dominant_intents,
        "confusion_index": sanitize_probability(features.get("confusion_index"), 0.0),
        "help_seeking_index": sanitize_probability(features.get("help_seeking_index"), 0.0),
        "sentiment_volatility": sanitize_probability(features.get("sentiment_volatility"), 0.0),
        "sentiment_shift_early_late": float(features.get("sentiment_shift_early_late") or 0.0),
        "reply_depth_max": float(features.get("reply_depth_max") or 0.0),
        "reply_branch_factor": sanitize_probability(features.get("reply_branch_factor"), 0.0),
        "reply_ratio": sanitize_probability(features.get("reply_ratio"), 0.0),
        "root_thread_concentration": sanitize_probability(
            features.get("root_thread_concentration"), 0.0
        ),
        "alignment_score": sanitize_probability(features.get("alignment_score"), 0.0),
        "value_prop_coverage": sanitize_probability(features.get("value_prop_coverage"), 0.0),
        "on_topic_ratio": sanitize_probability(features.get("on_topic_ratio"), 0.0),
        "artifact_drift_ratio": sanitize_probability(features.get("artifact_drift_ratio"), 0.0),
        "alignment_shift_early_late": float(features.get("alignment_shift_early_late") or 0.0),
        "alignment_confidence": sanitize_probability(
            features.get("alignment_confidence"),
            sanitize_probability(features.get("confidence"), 0.35),
        ),
        "alignment_method_version": str(features.get("alignment_method_version") or "manifest"),
        "confidence": sanitize_probability(features.get("confidence"), 0.35),
        "missingness_flags": missingness_flags,
    }


def coerce_trajectory_features(payload: Dict[str, Any]) -> Dict[str, Any]:
    hints = payload.get("signal_hints")
    if not isinstance(hints, dict):
        return {}
    raw = hints.get("trajectory_features")
    if not isinstance(raw, dict):
        raw = hints.get("trajectory")
    if not isinstance(raw, dict):
        return {}
    regime_probs = raw.get("regime_probabilities")
    if not isinstance(regime_probs, dict):
        regime_probs = {}
    return {
        "source": str(raw.get("source") or "request_hint"),
        "regime_pred": str(raw.get("regime_pred") or "balanced"),
        "regime_probabilities": {
            "spike": sanitize_probability(regime_probs.get("spike"), 0.0),
            "balanced": sanitize_probability(regime_probs.get("balanced"), 1.0),
            "durable": sanitize_probability(regime_probs.get("durable"), 0.0),
        },
        "regime_confidence": sanitize_probability(raw.get("regime_confidence"), 0.0),
        "durability_ratio": sanitize_probability(raw.get("durability_ratio"), 0.0),
        "available_ratio": sanitize_probability(raw.get("available_ratio"), 0.0),
    }


def prepare_candidate(
    *,
    payload: Dict[str, Any],
    as_of: datetime,
    query_profile: Dict[str, Any],
    manifest_comment_lookup: Callable[[str, datetime], Optional[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    candidate_id = str(
        payload.get("candidate_id") or payload.get("video_id") or payload.get("row_id") or ""
    ).strip()
    if not candidate_id:
        return None
    raw_text = safe_text(payload.get("text")) or " ".join(
        [
            safe_text(payload.get("caption")),
        ]
    ).strip()
    processed = process_text(
        text=raw_text,
        explicit_hashtags=list(payload.get("hashtags") or []),
        explicit_mentions=list(payload.get("mentions") or []),
    )
    text = processed.semantic_text or raw_text
    author_id = safe_text(payload.get("author_id")).lower()
    hashtags = list(processed.hashtags)
    keywords = [
        normalize_text(tag)
        for tag in list(payload.get("keywords") or [])
        if normalize_text(tag)
    ]
    tokens = list(dict.fromkeys([*processed.semantic_tokens, *keywords]))
    lexical_tokens = list(dict.fromkeys([*processed.lexical_tokens, *keywords, *hashtags]))
    content_type = normalize_content_type(payload.get("content_type"))
    locale = normalize_locale(payload.get("locale"))
    language = derive_language(payload.get("language"), locale)
    comment_trace = manifest_comment_lookup(candidate_id, as_of)
    if not comment_trace:
        comment_trace = coerce_comment_intelligence(payload)
    if not comment_trace:
        comment_trace = default_comment_trace(text, query_profile.get("lexical_tokens") or query_profile["tokens"])
    trajectory = coerce_trajectory_features(payload)

    support_flags: List[str] = []
    hard_fail = False
    if not author_id:
        support_flags.append("missing_author")
        hard_fail = True
    if not (text or hashtags or keywords):
        support_flags.append("missing_content_signal")
        hard_fail = True

    if hard_fail:
        support_level = "low"
        support_score = 0.0
    else:
        score = 0.0
        if text:
            score += 0.28
        else:
            support_flags.append("missing_text")
        if hashtags:
            score += 0.10
        else:
            support_flags.append("missing_hashtags")
        if keywords:
            score += 0.06
        else:
            support_flags.append("missing_keywords")
        score += 0.12
        if payload.get("topic_key") or infer_topic_key(payload) != "general":
            score += 0.05
        else:
            support_flags.append("missing_topic_key")
        if parse_dt(payload.get("posted_at")) or parse_dt(payload.get("as_of_time")):
            score += 0.05
        else:
            support_flags.append("missing_freshness")
        if comment_trace.get("available"):
            score += 0.10
        else:
            support_flags.append("missing_comment_intelligence")
        if language:
            score += 0.08
        else:
            support_flags.append("missing_language")
        if locale:
            score += 0.08
        else:
            support_flags.append("missing_locale")
        if content_type != "other":
            score += 0.08
        else:
            support_flags.append("missing_content_type")
        support_score = round_score(clamp(score, 0.0, 1.0), 4)
        if support_score >= SUPPORT_FULL_THRESHOLD:
            support_level = "full"
        elif support_score >= SUPPORT_PARTIAL_THRESHOLD:
            support_level = "partial"
        else:
            support_level = "low"

    return {
        "candidate_id": candidate_id,
        "candidate_row_id": candidate_id,
        "author_id": author_id or "unknown",
        "topic_key": infer_topic_key(payload),
        "text": text,
        "raw_text": raw_text,
        "semantic_text": text,
        "lexical_text": processed.lexical_text,
        "tokens": tokens,
        "semantic_tokens": tokens,
        "lexical_tokens": lexical_tokens,
        "hashtags": hashtags,
        "mentions": list(processed.mentions),
        "emoji_tokens": list(processed.emoji_tokens),
        "keywords": keywords,
        "content_type": content_type,
        "objective_guess": infer_candidate_objective(payload),
        "locale": locale,
        "language": language,
        "posted_at": parse_dt(payload.get("posted_at")) or parse_dt(payload.get("as_of_time")) or as_of,
        "comment_trace": comment_trace,
        "trajectory_trace": trajectory,
        "audience_tokens": lexical_tokens,
        "support_level": support_level,
        "support_score": support_score,
        "support_flags": support_flags,
        "raw_payload": payload,
        "retrieval_branch_scores": {
            "semantic": 0.0,
            "hashtag_topic": 0.0,
            "structured_compatibility": 0.0,
            "fused_retrieval": 0.0,
        },
        "retrieval_branches": [],
    }
