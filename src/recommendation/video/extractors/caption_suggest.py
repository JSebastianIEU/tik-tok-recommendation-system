"""Generate suggested caption and hashtags from video analysis results.

Combines visual topics, transcript keywords, and scene captions
to produce a suggested caption the user can accept or modify.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models import KeywordResult


def suggest_caption(
    visual_topics: List[str],
    transcript_text: str,
    scene_captions: List[str],
    keywords: List[KeywordResult],
    user_description: str = "",
) -> str:
    """Generate a suggested TikTok caption from video analysis."""
    parts = []

    # Use transcript as primary source if available
    if transcript_text and len(transcript_text.strip()) > 10:
        # Take first ~100 chars of transcript as base
        base = transcript_text.strip()
        if len(base) > 120:
            # Find a good break point
            cut = base[:120].rfind(" ")
            base = base[:cut] if cut > 50 else base[:120]
        parts.append(base)

    # Add visual context if transcript is thin
    if not parts and scene_captions:
        unique_captions = list(dict.fromkeys(scene_captions))[:3]
        parts.append(" | ".join(unique_captions))

    # Add topic context
    if visual_topics:
        topic_str = ", ".join(visual_topics[:3])
        if parts:
            parts.append(f"({topic_str})")
        else:
            parts.append(topic_str)

    # Add top keywords as hashtag-style suffix
    top_kw = [kw.keyword for kw in keywords[:5] if len(kw.keyword) > 2]
    if top_kw:
        hashtag_str = " ".join(f"#{kw.replace(' ', '')}" for kw in top_kw)
        parts.append(hashtag_str)

    if not parts:
        return user_description if user_description else "Check out this video!"

    return " ".join(parts)


def suggest_hashtags(
    visual_topics: List[str],
    keywords: List[KeywordResult],
    transcript_text: str,
    scene_captions: List[str],
    hashtag_recommender: Optional[Any] = None,
    max_hashtags: int = 10,
) -> List[str]:
    """Generate suggested hashtags from video analysis."""
    hashtags: List[str] = []
    seen: set = set()

    def _add(tag: str) -> None:
        normalized = tag.lower().strip()
        if not normalized.startswith("#"):
            normalized = f"#{normalized}"
        normalized = normalized.replace(" ", "")
        if normalized not in seen and len(normalized) > 2:
            seen.add(normalized)
            hashtags.append(normalized)

    # From visual topics (most relevant)
    for topic in visual_topics:
        for word in topic.split():
            if len(word) > 2:
                _add(word)

    # From YAKE keywords
    for kw in keywords[:10]:
        _add(kw.keyword.replace(" ", ""))

    # If hashtag recommender available, use enriched description
    if hashtag_recommender is not None:
        enriched = " ".join([transcript_text] + scene_captions)
        if len(enriched.strip()) > 10:
            recs = hashtag_recommender.recommend(
                caption=enriched, k=10, top_n=max_hashtags,
            )
            for rec in recs:
                _add(rec["hashtag"])

    # Always add generic high-reach tags
    for generic in ["#fyp", "#foryou", "#viral"]:
        if generic not in seen:
            _add(generic)

    return hashtags[:max_hashtags]


def build_enriched_description(
    user_description: str,
    transcript_text: str,
    scene_captions: List[str],
    keywords: List[KeywordResult],
) -> str:
    """Merge all text signals into one rich description for downstream matching."""
    parts = []

    if user_description.strip():
        parts.append(user_description.strip())

    if transcript_text.strip():
        parts.append(f"[transcript] {transcript_text.strip()}")

    if scene_captions:
        unique = list(dict.fromkeys(scene_captions))[:5]
        parts.append(f"[visual] {' | '.join(unique)}")

    top_kw = [kw.keyword for kw in keywords[:10]]
    if top_kw:
        parts.append(f"[keywords] {', '.join(top_kw)}")

    return " ".join(parts) if parts else user_description
