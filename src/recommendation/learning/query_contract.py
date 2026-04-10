from __future__ import annotations

from typing import Any, Dict, List, Optional

from .baseline_common import (
    CONTENT_TYPES,
    PRIMARY_CTAS,
    TOPIC_LEXICON,
    derive_language,
    normalize_locale,
    normalize_text,
    safe_text,
    uniq,
)
from ..semantic_processor import process_text


def build_audience_profile(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        label = safe_text(value.get("label"))
        segments = uniq(
            [
                normalize_text(item).replace(" ", "_")
                for item in list(value.get("segments") or [])
                if normalize_text(item)
            ]
        )
        expertise = safe_text(value.get("expertise_level")).lower() or "mixed"
        if expertise not in {"beginner", "intermediate", "advanced", "mixed"}:
            expertise = "mixed"
        normalized_label = normalize_text(label)
        return {
            "label": label,
            "normalized_label": normalized_label,
            "segments": segments,
            "expertise_level": expertise,
        }
    label = safe_text(value)
    return {
        "label": label,
        "normalized_label": normalize_text(label),
        "segments": [],
        "expertise_level": "mixed",
    }


def normalize_content_type(value: Any) -> str:
    normalized = normalize_text(value).replace(" ", "_")
    if normalized in CONTENT_TYPES:
        return normalized
    return "other"


def normalize_primary_cta(value: Any) -> str:
    normalized = normalize_text(value).replace(" ", "_")
    if normalized in PRIMARY_CTAS:
        return normalized
    return "none"


def infer_topic_key(payload: Dict[str, Any]) -> str:
    explicit = safe_text(payload.get("topic_key")).lower()
    if explicit:
        return explicit
    processed = process_text(
        text=" ".join(
            [
                safe_text(payload.get("text")),
                safe_text(payload.get("description")),
                " ".join(str(item) for item in list(payload.get("keywords") or [])),
            ]
        ).strip(),
        explicit_hashtags=list(payload.get("hashtags") or []),
        explicit_mentions=list(payload.get("mentions") or []),
    )
    tokens = uniq(processed.semantic_tokens + processed.hashtags)
    for topic, terms in TOPIC_LEXICON.items():
        if any(term in tokens for term in terms):
            return topic
    if processed.hashtags:
        return processed.hashtags[0]
    return "general"


def build_query_profile(
    *,
    objective: str,
    query: Dict[str, Any],
    fallback_language: Optional[str],
    fallback_locale: Optional[str],
    fallback_content_type: Optional[str],
) -> Dict[str, Any]:
    raw_text = safe_text(query.get("text")) or " ".join(
        [
            safe_text(query.get("description")),
        ]
    ).strip()
    processed = process_text(
        text=raw_text,
        explicit_hashtags=list(query.get("hashtags") or []),
        explicit_mentions=list(query.get("mentions") or []),
    )
    keywords = uniq(
        normalize_text(item)
        for item in list(query.get("keywords") or [])
        if normalize_text(item)
    )
    hashtags = list(processed.hashtags)
    mentions = list(processed.mentions)
    semantic_tokens = uniq(processed.semantic_tokens + keywords)
    lexical_tokens = uniq(processed.lexical_tokens + keywords + hashtags)
    topic_key = infer_topic_key(query)
    audience = build_audience_profile(query.get("audience"))
    locale = normalize_locale(query.get("locale") or fallback_locale)
    return {
        "query_id": str(query.get("query_id") or "query"),
        "objective": objective,
        "text": processed.semantic_text or raw_text,
        "raw_text": raw_text,
        "semantic_text": processed.semantic_text or raw_text,
        "lexical_text": processed.lexical_text,
        "tokens": semantic_tokens,
        "semantic_tokens": semantic_tokens,
        "lexical_tokens": lexical_tokens,
        "hashtags": hashtags,
        "mentions": mentions,
        "emoji_tokens": list(processed.emoji_tokens),
        "keywords": keywords,
        "topic_key": topic_key,
        "audience": audience,
        "content_type": normalize_content_type(
            query.get("content_type") or fallback_content_type
        ),
        "primary_cta": normalize_primary_cta(query.get("primary_cta")),
        "language": derive_language(query.get("language") or fallback_language, locale),
        "locale": locale,
        "query_text": processed.semantic_text or raw_text,
    }
