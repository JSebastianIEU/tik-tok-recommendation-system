from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

BASELINE_RETRIEVER_VERSION = "recommender.retrieval.baseline.v1"
BASELINE_RANKER_VERSION = "recommender.ranker.baseline.v1"
BASELINE_CALIBRATION_VERSION = "calibration.baseline.v1"
BASELINE_POLICY_VERSION = "policy.baseline.v1"
BASELINE_COMMENT_VERSION = "comment_intelligence.v2"
BASELINE_GRAPH_VERSION = "graph.baseline.v1"
BASELINE_TRAJECTORY_VERSION = "trajectory.baseline.v1"

CONTENT_TYPES = {
    "tutorial",
    "story",
    "reaction",
    "showcase",
    "opinion",
    "commentary",
    "trend_participation",
    "educational",
    "behind_the_scenes",
    "other",
}
PRIMARY_CTAS = {
    "follow",
    "comment",
    "save",
    "share",
    "link_click",
    "profile_visit",
    "dm",
    "none",
}

SEMANTIC_BRANCH_TOP_K = 100
HASHTAG_BRANCH_TOP_K = 75
STRUCTURED_BRANCH_TOP_K = 75
SHORTLIST_TOP_K = 100

SUPPORT_FULL_THRESHOLD = 0.75
SUPPORT_PARTIAL_THRESHOLD = 0.45

DEFAULT_RANKING_WEIGHTS = {
    "semantic_relevance": 0.40,
    "intent_alignment": 0.30,
    "reference_usefulness": 0.20,
    "support_confidence": 0.10,
}
OBJECTIVE_RANKING_WEIGHTS = {
    "reach": {
        "semantic_relevance": 0.42,
        "intent_alignment": 0.26,
        "reference_usefulness": 0.22,
        "support_confidence": 0.10,
    },
    "engagement": {
        "semantic_relevance": 0.38,
        "intent_alignment": 0.32,
        "reference_usefulness": 0.20,
        "support_confidence": 0.10,
    },
    "conversion": {
        "semantic_relevance": 0.34,
        "intent_alignment": 0.36,
        "reference_usefulness": 0.20,
        "support_confidence": 0.10,
    },
}

CTA_TERMS = {
    "follow": ("follow", "sigue", "subscribe"),
    "comment": ("comment", "comenta", "reply", "tell me"),
    "save": ("save", "guarda", "bookmark"),
    "share": ("share", "comparte", "send this"),
    "link_click": ("link", "bio", "shop", "visit"),
    "profile_visit": ("profile", "page", "check my page"),
    "dm": ("dm", "message", "inbox"),
    "none": (),
}
CONVERSION_TERMS = ("buy", "shop", "product", "price", "discount", "link", "bio")
COMMUNITY_TERMS = ("comment", "community", "join", "reply", "friends", "together")
ENGAGEMENT_TERMS = ("viral", "save", "share", "follow", "comment", "react")
TOPIC_LEXICON: Dict[str, Tuple[str, ...]] = {
    "food": ("food", "recipe", "cook", "meal", "kitchen", "comida", "receta"),
    "fitness": ("fitness", "workout", "gym", "exercise", "health", "salud"),
    "finance": ("finance", "invest", "money", "crypto", "budget", "stock", "trading"),
    "beauty": ("beauty", "makeup", "skincare", "hair", "cosmetic"),
    "fashion": ("fashion", "outfit", "style", "lookbook", "streetwear"),
    "tech": ("tech", "ai", "software", "app", "coding", "gadget"),
    "travel": ("travel", "trip", "vacation", "city", "journey", "viaje"),
    "education": ("learn", "lesson", "tutorial", "guide", "tips", "teach", "study"),
    "entertainment": ("funny", "meme", "dance", "music", "viral", "trend", "reaction"),
    "lifestyle": ("lifestyle", "daily", "routine", "mindset", "productivity", "selfcare"),
}


def to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def as_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isfinite(out):
        return out
    return fallback


def as_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def round_score(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def sanitize_probability(value: Any, fallback: float = 0.0) -> float:
    return round_score(clamp(as_float(value, fallback), 0.0, 1.0), 6)


def normalize_text(value: Any) -> str:
    text = safe_text(value).lower()
    text = re.sub(r"[\u0300-\u036f]", "", text)
    text = re.sub(r"[^\w\s#@-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def uniq(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def tokenize(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        normalized = normalize_text(value)
        if not normalized:
            continue
        for token in normalized.split(" "):
            cleaned = token.lstrip("#@").strip()
            if len(cleaned) >= 2:
                out.append(cleaned)
    return uniq(out)


def jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    a = set(left)
    b = set(right)
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def normalize_language(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    return cleaned[:8]


def normalize_locale(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().replace("_", "-").lower()
    if not cleaned:
        return None
    return cleaned[:24]


def derive_language(language: Any, locale: Any) -> Optional[str]:
    normalized = normalize_language(language)
    if normalized:
        return normalized
    normalized_locale = normalize_locale(locale)
    if not normalized_locale:
        return None
    return normalized_locale.split("-", 1)[0] or None


def candidate_video_id(row_id: str) -> str:
    if "::" in row_id:
        return row_id.split("::", 1)[0]
    return row_id
