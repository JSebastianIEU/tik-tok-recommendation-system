from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.common.config import settings

from .artifacts import ArtifactRegistry
from .baseline_common import (
    BASELINE_CALIBRATION_VERSION,
    BASELINE_COMMENT_VERSION,
    BASELINE_GRAPH_VERSION,
    BASELINE_POLICY_VERSION,
    BASELINE_RANKER_VERSION,
    BASELINE_RETRIEVER_VERSION,
    BASELINE_TRAJECTORY_VERSION,
    DEFAULT_RANKING_WEIGHTS,
    HASHTAG_BRANCH_TOP_K,
    OBJECTIVE_RANKING_WEIGHTS,
    SEMANTIC_BRANCH_TOP_K,
    SHORTLIST_TOP_K,
    STRUCTURED_BRANCH_TOP_K,
    SUPPORT_FULL_THRESHOLD,
    SUPPORT_PARTIAL_THRESHOLD,
    as_float as _as_float,
    as_int as _as_int,
    candidate_video_id as _candidate_video_id,
    clamp as _clamp,
    derive_language as _derive_language,
    jaccard as _jaccard,
    normalize_locale as _normalize_locale,
    normalize_text as _normalize_text,
    round_score as _round,
    safe_text as _safe_text,
    sanitize_probability as _sanitize_probability,
    to_utc_iso as _to_utc_iso,
    tokenize as _tokenize,
    uniq as _uniq,
)
from .candidate_support import (
    coerce_manifest_comment_intelligence as _coerce_manifest_comment_intelligence,
    prepare_candidate,
)
from .explainability_baseline import apply_explainability
from .learned_reranker import (
    LEARNED_RERANKER_ID,
    LEARNED_RERANKER_VERSION,
    LearnedPairwiseReranker,
)
from .objectives import map_objective
from .policy import PolicyReranker, PolicyRerankerConfig
from .query_contract import build_query_profile
from .ranking_baseline import rank_shortlist
from .retrieval_baseline import (
    hashtag_topic_score,
    retrieve_shortlist,
    retrieval_blend_score,
    structured_compatibility_score,
)
from .retriever import HybridRetriever
from .temporal import parse_dt
from .user_affinity import (
    USER_AFFINITY_VERSION,
    apply_user_affinity_blend,
    build_user_affinity_context,
)
from ..comment_intelligence import load_comment_intelligence_snapshot_manifest
from ..fabric import FeatureFabric


class RoutingContractError(ValueError):
    pass


class ArtifactCompatibilityError(ValueError):
    pass


class RecommenderStageTimeoutError(TimeoutError):
    def __init__(self, stage: str, elapsed_ms: float, budget_ms: float) -> None:
        self.stage = str(stage)
        self.elapsed_ms = float(elapsed_ms)
        self.budget_ms = float(budget_ms)
        super().__init__(
            f"{self.stage}_timeout elapsed_ms={self.elapsed_ms:.3f} budget_ms={self.budget_ms:.3f}"
        )


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
    "semantic_relevance": 0.30,
    "intent_alignment": 0.25,
    "performance_quality": 0.20,
    "reference_usefulness": 0.15,
    "support_confidence": 0.10,
}
OBJECTIVE_RANKING_WEIGHTS = {
    "reach": {
        "semantic_relevance": 0.28,
        "intent_alignment": 0.22,
        "performance_quality": 0.25,
        "reference_usefulness": 0.15,
        "support_confidence": 0.10,
    },
    "engagement": {
        "semantic_relevance": 0.28,
        "intent_alignment": 0.25,
        "performance_quality": 0.22,
        "reference_usefulness": 0.15,
        "support_confidence": 0.10,
    },
    "conversion": {
        "semantic_relevance": 0.26,
        "intent_alignment": 0.30,
        "performance_quality": 0.20,
        "reference_usefulness": 0.14,
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


@dataclass
class RecommenderRuntimeConfig:
    feature_schema_hash: Optional[str] = None
    contract_version: Optional[str] = None
    datamart_version: Optional[str] = None
    component: Optional[str] = "recommender-learning-v1"


@dataclass
class _IdentityCalibrator:
    objective: str

    def calibrate(self, score_raw: float, segment_id: str = "baseline_weighted") -> Dict[str, Any]:
        score = _sanitize_probability(score_raw, 0.0)
        return {
            "score_raw": score,
            "score_calibrated": score,
            "calibrator_segment_id": "baseline_identity",
            "requested_segment_id": str(segment_id or "baseline_weighted"),
            "calibrator_method": "identity",
            "calibrator_support_count": 0,
            "calibration_fallback_used": False,
            "target_definition": "baseline_weighted_score",
            "objective": self.objective,
            "version": BASELINE_CALIBRATION_VERSION,
        }


class _CommentManifestIndex:
    def __init__(self, rows: Sequence[Dict[str, Any]]) -> None:
        by_video: Dict[str, List[Tuple[datetime, Dict[str, Any]]]] = {}
        for row in rows:
            video_id = str(row.get("video_id") or "").strip()
            if not video_id:
                continue
            parsed = parse_dt(row.get("as_of_time"))
            if parsed is None:
                continue
            by_video.setdefault(video_id, []).append((parsed, row))
        for values in by_video.values():
            values.sort(key=lambda item: item[0])
        self.by_video = by_video

    def lookup(self, *, video_id: str, as_of: datetime) -> Optional[Dict[str, Any]]:
        rows = self.by_video.get(video_id)
        if not rows:
            return None
        selected: Optional[Dict[str, Any]] = None
        for ts, row in rows:
            if ts <= as_of:
                selected = row
            else:
                break
        return selected


def _to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isfinite(out):
        return out
    return fallback


def _as_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _sanitize_probability(value: Any, fallback: float = 0.0) -> float:
    return _round(_clamp(_as_float(value, fallback), 0.0, 1.0), 6)


def _normalize_text(value: Any) -> str:
    text = _safe_text(value).lower()
    text = re.sub(r"[\u0300-\u036f]", "", text)
    text = re.sub(r"[^\w\s#@-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _uniq(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _tokenize(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        normalized = _normalize_text(value)
        if not normalized:
            continue
        for token in normalized.split(" "):
            cleaned = token.lstrip("#@").strip()
            if len(cleaned) >= 2:
                out.append(cleaned)
    return _uniq(out)


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    a = set(left)
    b = set(right)
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def _normalize_language(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    return cleaned[:8]


def _normalize_locale(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().replace("_", "-").lower()
    if not cleaned:
        return None
    return cleaned[:24]


def _derive_language(language: Any, locale: Any) -> Optional[str]:
    normalized = _normalize_language(language)
    if normalized:
        return normalized
    normalized_locale = _normalize_locale(locale)
    if not normalized_locale:
        return None
    return normalized_locale.split("-", 1)[0] or None


def _manifest_ref_candidates(
    *,
    bundle_dir: Path,
    manifest_path: Optional[str],
    manifest_id: Optional[str],
) -> List[Path]:
    out: List[Path] = []
    if isinstance(manifest_path, str) and manifest_path.strip():
        path = Path(manifest_path.strip())
        out.append(path / "manifest.json" if path.is_dir() else path)
    if isinstance(manifest_id, str) and manifest_id.strip():
        out.extend(
            [
                bundle_dir.parent.parent
                / "comment_intelligence"
                / "features"
                / manifest_id.strip()
                / "manifest.json",
                Path("artifacts")
                / "comment_intelligence"
                / "features"
                / manifest_id.strip()
                / "manifest.json",
            ]
        )
    return [Path(ref) for ref in _uniq([str(item) for item in out])]


def _candidate_video_id(row_id: str) -> str:
    if "::" in row_id:
        return row_id.split("::", 1)[0]
    return row_id


def _load_fabric_from_env() -> FeatureFabric:
    calibration_path = settings.fabric_calibration_path
    if not calibration_path:
        return FeatureFabric()
    path = Path(calibration_path)
    if not path.exists():
        return FeatureFabric()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return FeatureFabric()
    return FeatureFabric(calibration_artifacts=payload)


def _build_audience_profile(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        label = _safe_text(value.get("label"))
        segments = _uniq(
            [
                _normalize_text(item).replace(" ", "_")
                for item in list(value.get("segments") or [])
                if _normalize_text(item)
            ]
        )
        expertise = _safe_text(value.get("expertise_level")).lower() or "mixed"
        if expertise not in {"beginner", "intermediate", "advanced", "mixed"}:
            expertise = "mixed"
        normalized_label = _normalize_text(label)
        return {
            "label": label,
            "normalized_label": normalized_label,
            "segments": segments,
            "expertise_level": expertise,
        }
    label = _safe_text(value)
    return {
        "label": label,
        "normalized_label": _normalize_text(label),
        "segments": [],
        "expertise_level": "mixed",
    }


def _normalize_content_type(value: Any) -> str:
    normalized = _normalize_text(value).replace(" ", "_")
    if normalized in CONTENT_TYPES:
        return normalized
    return "other"


def _normalize_primary_cta(value: Any) -> str:
    normalized = _normalize_text(value).replace(" ", "_")
    if normalized in PRIMARY_CTAS:
        return normalized
    return "none"


def _infer_topic_key(payload: Dict[str, Any]) -> str:
    explicit = _safe_text(payload.get("topic_key")).lower()
    if explicit:
        return explicit
    tokens = _tokenize(
        [
            payload.get("text"),
            payload.get("description"),
            *list(payload.get("hashtags") or []),
            *list(payload.get("keywords") or []),
        ]
    )
    for topic, terms in TOPIC_LEXICON.items():
        if any(term in tokens for term in terms):
            return topic
    hashtags = [
        _normalize_text(tag).lstrip("#")
        for tag in list(payload.get("hashtags") or [])
        if _normalize_text(tag).lstrip("#")
    ]
    if hashtags:
        return hashtags[0]
    return "general"


def _infer_candidate_objective(payload: Dict[str, Any]) -> str:
    text = _normalize_text(
        " ".join(
            [
                _safe_text(payload.get("text")),
                _safe_text(payload.get("caption")),
                " ".join(str(item) for item in list(payload.get("hashtags") or [])),
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


def _objective_compatibility(query_objective: str, candidate_objective: str) -> float:
    if query_objective == candidate_objective:
        return 1.0
    if (
        (query_objective == "reach" and candidate_objective == "engagement")
        or (query_objective == "engagement" and candidate_objective == "reach")
    ):
        return 0.65
    if (
        (query_objective == "community" and candidate_objective == "engagement")
        or (query_objective == "engagement" and candidate_objective == "community")
    ):
        return 0.78
    if (
        (query_objective == "conversion" and candidate_objective == "engagement")
        or (query_objective == "engagement" and candidate_objective == "conversion")
    ):
        return 0.55
    return 0.35


def _content_type_compatibility(query_type: str, candidate_type: str) -> float:
    if query_type == candidate_type:
        return 1.0
    if (
        (query_type == "tutorial" and candidate_type == "educational")
        or (query_type == "educational" and candidate_type == "tutorial")
    ):
        return 0.85
    if (
        (query_type == "story" and candidate_type == "behind_the_scenes")
        or (query_type == "commentary" and candidate_type == "opinion")
    ):
        return 0.70
    return 0.25


def _cta_alignment(primary_cta: str, text: str) -> float:
    terms = CTA_TERMS.get(primary_cta, ())
    if not terms:
        return 0.60
    normalized = _normalize_text(text)
    return 1.0 if any(_normalize_text(term) in normalized for term in terms) else 0.25


def _audience_compatibility(audience: Dict[str, Any], candidate_tokens: Sequence[str]) -> float:
    tokens = _uniq(
        list(audience.get("segments") or [])
        + _tokenize([audience.get("normalized_label") or ""])
    )
    if not tokens:
        return 0.55
    return _round(_clamp(_jaccard(tokens, candidate_tokens), 0.0, 1.0), 6)


def _locale_compatibility(
    query_locale: Optional[str],
    query_language: Optional[str],
    candidate_locale: Optional[str],
    candidate_language: Optional[str],
) -> float:
    if query_locale and candidate_locale and query_locale == candidate_locale:
        return 1.0
    if query_language and candidate_language and query_language == candidate_language:
        return 0.75
    if not candidate_locale and not candidate_language:
        return 0.50
    return 0.20


def _support_confidence_score(level: str, score: float) -> float:
    tier_floor = 0.82 if level == "full" else 0.52 if level == "partial" else 0.0
    return _round(_clamp((tier_floor * 0.55) + (score * 0.45), 0.0, 1.0), 6)


FRESHNESS_HALF_LIFE_DAYS = 18.0


def _freshness_score(posted_at: Optional[datetime], reference_date: datetime) -> float:
    if posted_at is None:
        return 0.55
    age_days = max(0.0, (reference_date - posted_at).total_seconds() / 86400.0)
    return _round(_clamp(math.exp((-math.log(2.0) * age_days) / FRESHNESS_HALF_LIFE_DAYS), 0.0, 1.0), 6)


def _performance_quality_score(candidate: Dict[str, Any]) -> float:
    metrics = candidate.get("engagement_metrics") or {}
    views = _as_float(metrics.get("views"), 0.0)
    engagement_rate = _as_float(metrics.get("engagement_rate"), 0.0)
    view_signal = math.log1p(views) / math.log1p(10_000_000) if views > 0 else 0.0
    er_signal = min(engagement_rate / 0.10, 1.0)
    return _round(_clamp(view_signal * 0.55 + er_signal * 0.45, 0.0, 1.0), 6)


def _reference_usefulness(candidate: Dict[str, Any], reference_date: datetime) -> float:
    comment_trace = candidate["comment_trace"]
    metadata_quality = _sanitize_probability(candidate["support_score"], 0.0)
    freshness = _freshness_score(candidate.get("posted_at"), reference_date)
    comment_richness = _sanitize_probability(comment_trace.get("value_prop_coverage"), 0.0)
    share_signal = _sanitize_probability(comment_trace.get("on_topic_ratio"), 0.0)
    fabric = candidate.get("fabric_signals") or {}
    content_quality = _clamp(
        (_as_float(fabric.get("clarity_score"), 0.5) * 0.5)
        + (_as_float(fabric.get("pacing_score"), 0.5) * 0.3)
        + (min(_as_float(fabric.get("cta_keyword_count"), 0), 3) / 3.0 * 0.2),
        0.0,
        1.0,
    )
    return _round(
        _clamp(
            (metadata_quality * 0.25)
            + (freshness * 0.20)
            + (content_quality * 0.20)
            + (comment_richness * 0.15)
            + (share_signal * 0.10)
            + (candidate["support_score"] * 0.10),
            0.0,
            1.0,
        ),
        6,
    )


def _retrieval_blend_score(scores: Dict[str, float], branch_count: int, support_score: float) -> float:
    agreement = 0.08 if branch_count >= 3 else 0.04 if branch_count == 2 else 0.0
    return _round(
        _clamp(
            (scores["semantic"] * 0.45)
            + (scores["hashtag_topic"] * 0.30)
            + (scores["structured_compatibility"] * 0.25)
            + agreement
            + (support_score * 0.04),
            0.0,
            1.0,
        ),
        6,
    )


def _score_components_for_candidate(
    *,
    query_profile: Dict[str, Any],
    candidate: Dict[str, Any],
    reference_date: datetime,
) -> Dict[str, float]:
    semantic_relevance = _round(
        _clamp(
            (candidate["retrieval_branch_scores"]["semantic"] * 0.70)
            + (candidate["retrieval_branch_scores"]["hashtag_topic"] * 0.30),
            0.0,
            1.0,
        ),
        6,
    )
    intent_alignment = _round(
        _clamp(
            (_objective_compatibility(query_profile["objective"], candidate["objective_guess"]) * 0.25)
            + (_content_type_compatibility(query_profile["content_type"], candidate["content_type"]) * 0.30)
            + (_cta_alignment(query_profile["primary_cta"], candidate["text"]) * 0.20)
            + (_audience_compatibility(query_profile["audience"], candidate["audience_tokens"]) * 0.15)
            + (
                _locale_compatibility(
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
    performance_quality = _performance_quality_score(candidate)
    reference_usefulness = _reference_usefulness(candidate, reference_date)
    support_confidence = _support_confidence_score(
        candidate["support_level"], candidate["support_score"]
    )
    return {
        "semantic_relevance": semantic_relevance,
        "intent_alignment": intent_alignment,
        "performance_quality": performance_quality,
        "reference_usefulness": reference_usefulness,
        "support_confidence": support_confidence,
    }


def _ranking_reasons(candidate: Dict[str, Any], score_components: Dict[str, float]) -> List[str]:
    ordered = sorted(score_components.items(), key=lambda item: item[1], reverse=True)
    reasons = [f"strong_{name}" for name, _ in ordered[:2]]
    if len(candidate["retrieval_branches"]) >= 2:
        reasons.append("multi_branch_retrieval_match")
    if candidate["support_level"] == "full":
        reasons.append("fully_supported_reference")
    return reasons


def _default_comment_trace(text: str, query_tokens: Sequence[str]) -> Dict[str, Any]:
    candidate_tokens = _tokenize([text])
    overlap = _jaccard(query_tokens, candidate_tokens)
    return {
        "source": "baseline_inferred",
        "available": False,
        "taxonomy_version": BASELINE_COMMENT_VERSION,
        "dominant_intents": [],
        "confusion_index": _round(1.0 - overlap, 6),
        "help_seeking_index": _round(min(1.0, overlap + 0.15), 6),
        "sentiment_volatility": 0.0,
        "sentiment_shift_early_late": 0.0,
        "reply_depth_max": 0.0,
        "reply_branch_factor": 0.0,
        "reply_ratio": 0.0,
        "root_thread_concentration": 0.0,
        "alignment_score": _round(overlap, 6),
        "value_prop_coverage": _round(min(1.0, overlap + 0.10), 6),
        "on_topic_ratio": _round(min(1.0, overlap + 0.05), 6),
        "artifact_drift_ratio": _round(max(0.0, 0.30 - overlap), 6),
        "alignment_shift_early_late": 0.0,
        "alignment_confidence": 0.35,
        "alignment_method_version": "baseline_overlap.v1",
        "confidence": 0.35,
        "missingness_flags": ["comment_intelligence_unavailable"],
    }


def _coerce_comment_intelligence(payload: Dict[str, Any]) -> Dict[str, Any]:
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
        "confusion_index": _sanitize_probability(raw.get("confusion_index"), 0.0),
        "help_seeking_index": _sanitize_probability(raw.get("help_seeking_index"), 0.0),
        "sentiment_volatility": _sanitize_probability(raw.get("sentiment_volatility"), 0.0),
        "sentiment_shift_early_late": _as_float(raw.get("sentiment_shift_early_late"), 0.0),
        "reply_depth_max": _as_float(raw.get("reply_depth_max"), 0.0),
        "reply_branch_factor": _sanitize_probability(raw.get("reply_branch_factor"), 0.0),
        "reply_ratio": _sanitize_probability(raw.get("reply_ratio"), 0.0),
        "root_thread_concentration": _sanitize_probability(raw.get("root_thread_concentration"), 0.0),
        "alignment_score": _sanitize_probability(raw.get("alignment_score"), 0.0),
        "value_prop_coverage": _sanitize_probability(raw.get("value_prop_coverage"), 0.0),
        "on_topic_ratio": _sanitize_probability(raw.get("on_topic_ratio"), 0.0),
        "artifact_drift_ratio": _sanitize_probability(raw.get("artifact_drift_ratio"), 0.0),
        "alignment_shift_early_late": _as_float(raw.get("alignment_shift_early_late"), 0.0),
        "alignment_confidence": _sanitize_probability(raw.get("alignment_confidence"), _as_float(raw.get("confidence"), 0.35)),
        "alignment_method_version": str(raw.get("alignment_method_version") or "request_hint"),
        "confidence": _sanitize_probability(raw.get("confidence"), 0.35),
        "missingness_flags": [str(item) for item in list(raw.get("missingness_flags") or [])],
    }


def _coerce_manifest_comment_intelligence(row: Dict[str, Any]) -> Dict[str, Any]:
    features = row.get("features")
    if not isinstance(features, dict):
        return {}
    missingness = row.get("missingness")
    missingness_flags = (
        sorted(str(key) for key in missingness.keys())
        if isinstance(missingness, dict)
        else []
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
        "confusion_index": _sanitize_probability(features.get("confusion_index"), 0.0),
        "help_seeking_index": _sanitize_probability(features.get("help_seeking_index"), 0.0),
        "sentiment_volatility": _sanitize_probability(features.get("sentiment_volatility"), 0.0),
        "sentiment_shift_early_late": _as_float(features.get("sentiment_shift_early_late"), 0.0),
        "reply_depth_max": _as_float(features.get("reply_depth_max"), 0.0),
        "reply_branch_factor": _sanitize_probability(features.get("reply_branch_factor"), 0.0),
        "reply_ratio": _sanitize_probability(features.get("reply_ratio"), 0.0),
        "root_thread_concentration": _sanitize_probability(features.get("root_thread_concentration"), 0.0),
        "alignment_score": _sanitize_probability(features.get("alignment_score"), 0.0),
        "value_prop_coverage": _sanitize_probability(features.get("value_prop_coverage"), 0.0),
        "on_topic_ratio": _sanitize_probability(features.get("on_topic_ratio"), 0.0),
        "artifact_drift_ratio": _sanitize_probability(features.get("artifact_drift_ratio"), 0.0),
        "alignment_shift_early_late": _as_float(features.get("alignment_shift_early_late"), 0.0),
        "alignment_confidence": _sanitize_probability(features.get("alignment_confidence"), _sanitize_probability(features.get("confidence"), 0.35)),
        "alignment_method_version": str(features.get("alignment_method_version") or "manifest"),
        "confidence": _sanitize_probability(features.get("confidence"), 0.35),
        "missingness_flags": missingness_flags,
    }


def _coerce_trajectory_features(payload: Dict[str, Any]) -> Dict[str, Any]:
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
            "spike": _sanitize_probability(regime_probs.get("spike"), 0.0),
            "balanced": _sanitize_probability(regime_probs.get("balanced"), 1.0),
            "durable": _sanitize_probability(regime_probs.get("durable"), 0.0),
        },
        "regime_confidence": _sanitize_probability(raw.get("regime_confidence"), 0.0),
        "durability_ratio": _sanitize_probability(raw.get("durability_ratio"), 0.0),
        "available_ratio": _sanitize_probability(raw.get("available_ratio"), 0.0),
    }


class RecommenderRuntime:
    def __init__(
        self,
        bundle_dir: Path,
        config: Optional[RecommenderRuntimeConfig] = None,
    ) -> None:
        self.bundle_dir = bundle_dir
        self.config = config or RecommenderRuntimeConfig()

        registry = ArtifactRegistry(bundle_dir.parent)
        manifest = registry.load_manifest(bundle_dir)
        required_fields = (
            "component",
            "contract_version",
            "datamart_version",
            "feature_schema_hash",
            "objectives",
        )
        missing = [field for field in required_fields if field not in manifest]
        if missing:
            raise ValueError(
                f"Invalid recommender artifact manifest. Missing fields: {', '.join(missing)}"
            )

        expected: Dict[str, Any] = {}
        if self.config.component:
            expected["component"] = self.config.component
        if self.config.feature_schema_hash:
            expected["feature_schema_hash"] = self.config.feature_schema_hash
        if self.config.contract_version:
            expected["contract_version"] = self.config.contract_version
        if self.config.datamart_version:
            expected["datamart_version"] = self.config.datamart_version
        if expected:
            try:
                registry.assert_compatible(bundle_dir, expected)
            except ValueError as error:
                raise ArtifactCompatibilityError(str(error)) from error

        self.manifest = manifest
        self.bundle_id = str(
            manifest.get("bundle_id") or manifest.get("run_name") or bundle_dir.name
        )

        self.fabric = _load_fabric_from_env()
        expected_fabric_signature = manifest.get("fabric_registry_signature")
        if (
            isinstance(expected_fabric_signature, str)
            and expected_fabric_signature
            and self.fabric.registry.signature() != expected_fabric_signature
        ):
            raise ValueError(
                "Fabric registry signature mismatch between runtime and artifact manifest."
            )
        expected_fabric_schema_hashes = manifest.get("fabric_schema_hashes")
        if isinstance(expected_fabric_schema_hashes, dict) and expected_fabric_schema_hashes:
            self.fabric.registry.assert_compatible(
                {str(key): str(value) for key, value in expected_fabric_schema_hashes.items()}
            )

        self.graph_bundle_id = str(
            (
                (manifest.get("graph") or {}).get("graph_bundle_id")
                if isinstance(manifest.get("graph"), dict)
                else manifest.get("graph_bundle_id")
            )
            or ""
        )
        self.graph_version = str(
            (
                (manifest.get("graph") or {}).get("graph_version")
                if isinstance(manifest.get("graph"), dict)
                else manifest.get("graph_version")
            )
            or BASELINE_GRAPH_VERSION
        )
        self.trajectory_manifest_id = str(
            (
                (manifest.get("trajectory") or {}).get("trajectory_manifest_id")
                if isinstance(manifest.get("trajectory"), dict)
                else manifest.get("trajectory_manifest_id")
            )
            or ""
        )
        self.trajectory_version = str(
            (
                (manifest.get("trajectory") or {}).get("trajectory_version")
                if isinstance(manifest.get("trajectory"), dict)
                else manifest.get("trajectory_version")
            )
            or BASELINE_TRAJECTORY_VERSION
        )

        self.rankers = {
            str(objective): {"ranker_id": "baseline_weighted"}
            for objective in list(manifest.get("objectives") or [])
        }
        self.learned_rerankers: Dict[str, LearnedPairwiseReranker] = {}
        self.learned_reranker_load_warnings: Dict[str, str] = {}
        self.calibrators: Dict[str, _IdentityCalibrator] = {}
        self.calibration_load_warnings: Dict[str, str] = {}
        for objective in self.rankers.keys():
            learned_dir = bundle_dir / "rankers" / objective / "learned_reranker"
            manifest_path = learned_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    learned_reranker = LearnedPairwiseReranker.load(learned_dir)
                    self.learned_rerankers[objective] = learned_reranker
                    self.rankers[objective] = {
                        "ranker_id": LEARNED_RERANKER_ID,
                        "version": LEARNED_RERANKER_VERSION,
                    }
                except Exception as error:
                    self.learned_reranker_load_warnings[objective] = (
                        f"learned_reranker_load_failed: {error}"
                    )
        for objective in self.rankers.keys():
            calibration_path = bundle_dir / "rankers" / objective / "calibration.json"
            if calibration_path.exists():
                try:
                    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
                    compatibility = payload.get("compatibility") if isinstance(payload.get("compatibility"), dict) else {}
                    feature_schema_hash = str(compatibility.get("feature_schema_hash") or "")
                    if feature_schema_hash and feature_schema_hash != str(self.manifest.get("feature_schema_hash") or ""):
                        self.calibration_load_warnings[objective] = (
                            "calibration_incompatible: feature_schema_hash mismatch"
                        )
                        continue
                except Exception as error:
                    self.calibration_load_warnings[objective] = f"calibration_load_failed: {error}"
                    continue
            self.calibrators[objective] = _IdentityCalibrator(objective=objective)

        self.retriever: Optional[HybridRetriever] = None
        self.retriever_load_warning: Optional[str] = None
        self._load_retriever()
        self.comment_index: Optional[_CommentManifestIndex] = None
        self.comment_index_source_path: Optional[str] = None
        self.comment_index_load_error: Optional[str] = None
        self._load_comment_feature_index()

    def _load_retriever(self) -> None:
        retriever_dir = self.bundle_dir / "retriever"
        manifest_path = retriever_dir / "manifest.json"
        if not manifest_path.exists():
            self.retriever = None
            self.retriever_load_warning = "retriever_artifact_missing"
            return
        try:
            self.retriever = HybridRetriever.load(retriever_dir)
            self.retriever_load_warning = None
        except Exception as error:
            self.retriever = None
            self.retriever_load_warning = f"retriever_load_failed: {error}"

        # Reconcile graph/trajectory metadata with actual retriever blend
        if self.retriever is not None and manifest_path.exists():
            try:
                ret_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                blend = ret_manifest.get("objective_blend") or {}
                graph_used = any(
                    float((weights or {}).get("graph_dense") or 0) > 0
                    for weights in blend.values()
                )
                trajectory_used = any(
                    float((weights or {}).get("trajectory_dense") or 0) > 0
                    for weights in blend.values()
                )
                if not graph_used:
                    self.graph_bundle_id = ""
                if not trajectory_used:
                    self.trajectory_manifest_id = ""
            except Exception:
                pass

    def _load_comment_feature_index(self) -> None:
        manifest_path = self.manifest.get("comment_feature_manifest_path")
        manifest_id = self.manifest.get("comment_feature_manifest_id")
        refs = _manifest_ref_candidates(
            bundle_dir=self.bundle_dir,
            manifest_path=str(manifest_path) if manifest_path is not None else None,
            manifest_id=str(manifest_id) if manifest_id is not None else None,
        )
        for ref in refs:
            if not ref.exists():
                continue
            try:
                _, rows = load_comment_intelligence_snapshot_manifest(ref)
                self.comment_index = _CommentManifestIndex(rows)
                self.comment_index_source_path = str(ref)
                self.comment_index_load_error = None
                return
            except Exception as error:
                self.comment_index = None
                self.comment_index_source_path = None
                self.comment_index_load_error = str(error)

        # Auto-discover comment intelligence from well-known directories
        import os as _os
        discovery_dirs: List[Path] = []
        env_dir = _os.environ.get("COMMENT_INTELLIGENCE_DIR", "").strip()
        if env_dir:
            discovery_dirs.append(Path(env_dir))
        discovery_dirs.extend([
            self.bundle_dir.parent.parent / "comment_intelligence" / "features",
            Path("artifacts") / "comment_intelligence" / "features",
        ])
        best_manifest: Optional[Path] = None
        best_generated_at: Optional[str] = None
        for search_dir in discovery_dirs:
            if not search_dir.is_dir():
                continue
            for child in search_dir.iterdir():
                candidate = child / "manifest.json"
                if not candidate.exists():
                    continue
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                    gen_at = str(payload.get("generated_at") or "")
                    if best_generated_at is None or gen_at > best_generated_at:
                        best_manifest = candidate
                        best_generated_at = gen_at
                except Exception:
                    continue
        if best_manifest is not None:
            try:
                _, rows = load_comment_intelligence_snapshot_manifest(best_manifest)
                self.comment_index = _CommentManifestIndex(rows)
                self.comment_index_source_path = str(best_manifest)
                self.comment_index_load_error = None
            except Exception as error:
                self.comment_index_load_error = f"auto_discover_failed: {error}"

    def _manifest_comment_for_row_id(self, *, row_id: str, as_of: datetime) -> Optional[Dict[str, Any]]:
        if self.comment_index is None:
            return None
        candidate_video_ids = [row_id]
        row_video_id = _candidate_video_id(row_id)
        if row_video_id not in candidate_video_ids:
            candidate_video_ids.append(row_video_id)
        for video_id in candidate_video_ids:
            found = self.comment_index.lookup(video_id=video_id, as_of=as_of)
            if isinstance(found, dict):
                return _coerce_manifest_comment_intelligence(found)
        return None

    def _retriever_row_payload(self, row_id: str) -> Dict[str, Any]:
        if self.retriever is None:
            return {"row_id": row_id, "video_id": row_id, "candidate_id": row_id}
        metadata = self.retriever.row_payload(row_id)
        candidate_id = str(
            metadata.get("candidate_id") or metadata.get("video_id") or row_id
        ).strip() or row_id
        return {
            "candidate_id": candidate_id,
            "row_id": str(metadata.get("row_id") or row_id),
            "video_id": str(metadata.get("video_id") or candidate_id),
            "text": str(metadata.get("semantic_text") or metadata.get("text") or "").strip(),
            "caption": str(metadata.get("caption") or metadata.get("text") or "").strip(),
            "hashtags": list(metadata.get("hashtags") or []),
            "keywords": list(metadata.get("keywords") or []),
            "search_query": metadata.get("search_query"),
            "author_id": metadata.get("author_id"),
            "as_of_time": metadata.get("as_of_time"),
            "posted_at": metadata.get("posted_at") or metadata.get("as_of_time"),
            "language": metadata.get("language"),
            "locale": metadata.get("locale"),
            "content_type": metadata.get("content_type"),
        }

    def _merge_candidate_payload(
        self,
        *,
        request_payload: Optional[Dict[str, Any]],
        retriever_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(request_payload, dict):
            return dict(retriever_payload)
        merged = dict(retriever_payload)
        for key, value in request_payload.items():
            if isinstance(value, str):
                if value.strip():
                    merged[key] = value
                continue
            if isinstance(value, list):
                if value:
                    merged[key] = value
                continue
            if isinstance(value, dict):
                if value:
                    existing = merged.get(key)
                    if isinstance(existing, dict):
                        merged[key] = {**existing, **value}
                    else:
                        merged[key] = value
                continue
            if value is not None:
                merged[key] = value
        return merged

    def _prepare_retrieved_candidates(
        self,
        *,
        retrieved_items: Sequence[Dict[str, Any]],
        as_of: datetime,
        query_profile: Dict[str, Any],
        request_candidates_by_id: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        for item in retrieved_items:
            candidate_id = str(item.get("candidate_id") or "").strip()
            row_id = str(item.get("candidate_row_id") or item.get("feature_row_id") or candidate_id).strip()
            retriever_payload = self._retriever_row_payload(row_id)
            merged_payload = self._merge_candidate_payload(
                request_payload=request_candidates_by_id.get(candidate_id)
                or request_candidates_by_id.get(row_id),
                retriever_payload=retriever_payload,
            )
            prepared_row = self._prepare_candidate(
                payload=merged_payload,
                as_of=as_of,
                query_profile=query_profile,
            )
            if prepared_row is None:
                continue
            semantic_score = _sanitize_probability(
                (
                    ((item.get("retrieval_branch_scores") or {}).get("dense_text"))
                    if isinstance(item.get("retrieval_branch_scores"), dict)
                    else item.get("dense_text_score", item.get("dense_score"))
                ),
                0.0,
            )
            lexical_score = _sanitize_probability(
                (
                    ((item.get("retrieval_branch_scores") or {}).get("lexical"))
                    if isinstance(item.get("retrieval_branch_scores"), dict)
                    else item.get("lexical_score", item.get("sparse_score"))
                ),
                0.0,
            )
            hashtag_score = hashtag_topic_score(query_profile, prepared_row)
            structured_score = structured_compatibility_score(query_profile, prepared_row)
            fused_score = retrieval_blend_score(
                {
                    "semantic": semantic_score,
                    "hashtag_topic": hashtag_score,
                    "structured_compatibility": structured_score,
                },
                sum(
                    1
                    for value in (semantic_score, hashtag_score, structured_score)
                    if float(value) > 0.0
                ),
                0.0,
            )
            retrieval_branch_scores = {
                "semantic": semantic_score,
                "hashtag_topic": hashtag_score,
                "structured_compatibility": structured_score,
                "fused_retrieval": fused_score,
                "lexical": lexical_score,
                "dense_text": semantic_score,
                "multimodal": _sanitize_probability(
                    ((item.get("retrieval_branch_scores") or {}).get("multimodal"))
                    if isinstance(item.get("retrieval_branch_scores"), dict)
                    else item.get("multimodal_score"),
                    0.0,
                ),
                "graph_dense": _sanitize_probability(
                    ((item.get("retrieval_branch_scores") or {}).get("graph_dense"))
                    if isinstance(item.get("retrieval_branch_scores"), dict)
                    else item.get("graph_dense_score"),
                    0.0,
                ),
                "trajectory_dense": _sanitize_probability(
                    ((item.get("retrieval_branch_scores") or {}).get("trajectory_dense"))
                    if isinstance(item.get("retrieval_branch_scores"), dict)
                    else item.get("trajectory_dense_score"),
                    0.0,
                ),
                "fused": _sanitize_probability(
                    ((item.get("retrieval_branch_scores") or {}).get("fused"))
                    if isinstance(item.get("retrieval_branch_scores"), dict)
                    else item.get("fused_score"),
                    fused_score,
                ),
            }
            prepared_row["candidate_id"] = candidate_id or prepared_row["candidate_id"]
            prepared_row["candidate_row_id"] = row_id or prepared_row["candidate_row_id"]
            prepared_row["retrieval_branch_scores"] = retrieval_branch_scores
            prepared_row["retrieval_branches"] = [
                key
                for key in ("semantic", "hashtag_topic", "structured_compatibility")
                if float(retrieval_branch_scores.get(key) or 0.0) > 0.0
            ]
            if isinstance(item.get("graph_trace"), dict):
                prepared_row["graph_trace"] = dict(item["graph_trace"])
            if isinstance(item.get("trajectory_trace"), dict):
                prepared_row["trajectory_trace"] = {
                    **prepared_row.get("trajectory_trace", {}),
                    **item["trajectory_trace"],
                }
            prepared_row["creator_retrieval_score"] = _sanitize_probability(
                item.get("creator_retrieval_score"),
                0.0,
            )
            if isinstance(item.get("creator_retrieval_trace"), dict):
                prepared_row["creator_retrieval_trace"] = dict(item["creator_retrieval_trace"])
            prepared.append(prepared_row)
        return prepared

    def compatibility_payload(
        self,
        required_compat: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        required = {
            str(key): str(value)
            for key, value in (required_compat or {}).items()
            if str(key).strip()
        }
        fingerprints = {
            "bundle_id": self.bundle_id,
            "component": str(self.manifest.get("component") or ""),
            "contract_version": str(self.manifest.get("contract_version") or ""),
            "datamart_version": str(self.manifest.get("datamart_version") or ""),
            "feature_schema_hash": str(self.manifest.get("feature_schema_hash") or ""),
            "ranker_family_schema_hash": str(self.manifest.get("ranker_family_schema_hash") or ""),
            "ranker_family_version": BASELINE_RANKER_VERSION,
            "graph_bundle_id": self.graph_bundle_id,
            "trajectory_manifest_id": self.trajectory_manifest_id,
            "trajectory_version": self.trajectory_version,
        }
        mismatches: List[Dict[str, str]] = []
        for key, expected_value in required.items():
            actual = str(fingerprints.get(key, ""))
            if actual != str(expected_value):
                mismatches.append({"key": key, "expected": str(expected_value), "actual": actual})
        return {
            "ok": len(mismatches) == 0,
            "bundle_dir": str(self.bundle_dir),
            "bundle_id": self.bundle_id,
            "fingerprints": fingerprints,
            "required_compat": required,
            "mismatches": mismatches,
        }

    def _validate_routing_contract(
        self,
        *,
        routing: Dict[str, Any],
        requested_objective: str,
        effective_objective: str,
    ) -> Dict[str, Any]:
        objective_requested = str(
            routing.get("objective_requested") or requested_objective
        ).strip()
        objective_effective = str(
            routing.get("objective_effective") or effective_objective
        ).strip()
        if objective_effective != effective_objective:
            raise RoutingContractError(
                f"routing.objective_effective={objective_effective!r} does not match runtime effective objective={effective_objective!r}"
            )
        if objective_requested and objective_requested != requested_objective:
            raise RoutingContractError(
                f"routing.objective_requested={objective_requested!r} does not match request objective={requested_objective!r}"
            )
        track = str(routing.get("track") or "post_publication").strip().lower()
        if track not in {"pre_publication", "post_publication"}:
            raise RoutingContractError(
                "routing.track must be one of: pre_publication, post_publication."
            )
        allow_fallback = bool(routing.get("allow_fallback", True))
        required_compat_raw = (
            routing.get("required_compat")
            if isinstance(routing.get("required_compat"), dict)
            else {}
        )
        required_compat = {
            str(key): str(value)
            for key, value in required_compat_raw.items()
            if str(key).strip()
        }
        request_id = str(routing.get("request_id") or "").strip()
        if request_id:
            try:
                uuid.UUID(request_id)
            except ValueError as error:
                raise RoutingContractError(
                    "routing.request_id must be a valid UUID string."
                ) from error
        experiment_payload = (
            routing.get("experiment")
            if isinstance(routing.get("experiment"), dict)
            else {}
        )
        experiment_variant = str(experiment_payload.get("variant") or "").strip() or None
        if experiment_variant and experiment_variant not in {"control", "treatment"}:
            raise RoutingContractError(
                "routing.experiment.variant must be one of: control, treatment."
            )
        compatibility = self.compatibility_payload(required_compat=required_compat)
        if not bool(compatibility.get("ok", False)):
            raise ArtifactCompatibilityError(
                "incompatible_artifact: "
                + " | ".join(
                    f"{item['key']} expected={item['expected']!r} actual={item['actual']!r}"
                    for item in compatibility.get("mismatches", [])
                    if isinstance(item, dict)
                )
            )
        return {
            "objective_requested": requested_objective,
            "objective_effective": effective_objective,
            "track": track,
            "allow_fallback": allow_fallback,
            "required_compat": required_compat,
            "selected_bundle_id": self.bundle_id,
            "route_reason": "objective_route_match",
            "request_id": request_id,
            "experiment": {
                "id": str(experiment_payload.get("id") or "").strip() or None,
                "variant": experiment_variant,
                "unit_hash": str(experiment_payload.get("unit_hash") or "").strip() or None,
            },
        }

    def _build_query_profile(
        self,
        *,
        objective: str,
        query: Dict[str, Any],
        fallback_language: Optional[str],
        fallback_locale: Optional[str],
        fallback_content_type: Optional[str],
    ) -> Dict[str, Any]:
        return build_query_profile(
            objective=objective,
            query=query,
            fallback_language=fallback_language,
            fallback_locale=fallback_locale,
            fallback_content_type=fallback_content_type,
        )

    def _prepare_candidate(
        self,
        *,
        payload: Dict[str, Any],
        as_of: datetime,
        query_profile: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        candidate = prepare_candidate(
            payload=payload,
            as_of=as_of,
            query_profile=query_profile,
            manifest_comment_lookup=lambda row_id, point_in_time: self._manifest_comment_for_row_id(
                row_id=row_id,
                as_of=point_in_time,
            ),
        )
        if candidate is not None:
            try:
                fabric_out = self.fabric.extract({
                    "video_id": candidate["candidate_id"],
                    "as_of_time": as_of.isoformat(),
                    "caption": candidate.get("text", ""),
                    "hashtags": candidate.get("hashtags", []),
                    "keywords": candidate.get("keywords", []),
                    "content_type": candidate.get("content_type"),
                })
                candidate["fabric_signals"] = {
                    "clarity_score": fabric_out.text.clarity_score,
                    "pacing_score": fabric_out.structure.pacing_score,
                    "cta_keyword_count": fabric_out.text.cta_keyword_count,
                    "hook_timing_seconds": fabric_out.structure.hook_timing_seconds,
                }
            except Exception:
                candidate["fabric_signals"] = {}
        return candidate

    def _apply_explainability(
        self,
        *,
        query_profile: Dict[str, Any],
        items: List[Dict[str, Any]],
        run_counterfactuals: bool,
    ) -> None:
        apply_explainability(items=items, run_counterfactuals=run_counterfactuals)

    def recommend(
        self,
        objective: str,
        as_of_time: Any,
        query: Dict[str, Any],
        candidates: Sequence[Dict[str, Any]],
        user_context: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        retrieve_k: int = 200,
        language: Optional[str] = None,
        locale: Optional[str] = None,
        content_type: Optional[str] = None,
        candidate_ids: Optional[Sequence[str]] = None,
        policy_overrides: Optional[Dict[str, Any]] = None,
        portfolio: Optional[Dict[str, Any]] = None,
        graph_controls: Optional[Dict[str, Any]] = None,
        trajectory_controls: Optional[Dict[str, Any]] = None,
        explainability: Optional[Dict[str, Any]] = None,
        routing: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        started_total = time.perf_counter()
        requested_objective, effective_objective = map_objective(objective)
        configured_objectives = {str(item) for item in self.manifest.get("objectives", [])}
        if effective_objective not in configured_objectives:
            raise ValueError(
                f"Objective '{effective_objective}' is not available in loaded artifact bundle."
            )
        as_of = parse_dt(as_of_time)
        if as_of is None:
            raise ValueError("as_of_time must be a valid ISO-8601 timestamp.")

        explainability_payload = explainability if isinstance(explainability, dict) else {}
        routing_payload = routing if isinstance(routing, dict) else {}
        routing_decision = self._validate_routing_contract(
            routing=routing_payload,
            requested_objective=requested_objective,
            effective_objective=effective_objective,
        )
        query_profile = self._build_query_profile(
            objective=requested_objective,
            query=query,
            fallback_language=language,
            fallback_locale=locale,
            fallback_content_type=content_type,
        )
        user_affinity_context = build_user_affinity_context(
            user_context=user_context if isinstance(user_context, dict) else None,
            effective_objective=effective_objective,
            as_of=as_of,
        )

        graph_controls_payload = graph_controls if isinstance(graph_controls, dict) else {}
        trajectory_controls_payload = trajectory_controls if isinstance(trajectory_controls, dict) else {}
        graph_enabled = bool(graph_controls_payload.get("enable_graph_branch", True))
        trajectory_enabled = bool(trajectory_controls_payload.get("enabled", True))
        graph_fallback_mode = (
            "graph_branch_disabled"
            if not graph_enabled
            else "active" if self.graph_bundle_id else "graph_bundle_unavailable"
        )
        trajectory_mode = (
            "trajectory_branch_disabled"
            if not trajectory_enabled
            else "active" if self.trajectory_manifest_id else "trajectory_manifest_unavailable"
        )

        candidate_scope = {
            str(item).strip()
            for item in list(candidate_ids or [])
            if isinstance(item, str) and str(item).strip()
        }
        request_candidates_by_id: Dict[str, Dict[str, Any]] = {}
        for item in candidates:
            if not isinstance(item, dict):
                continue
            for key in ("candidate_id", "video_id", "row_id"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    request_candidates_by_id[value.strip()] = item

        prepared: List[Dict[str, Any]] = []
        usable: List[Dict[str, Any]] = []
        retrieval_mode = "intersected" if candidate_scope else "global"
        retrieval_stats: Dict[str, Any] = {}
        retrieval_personalization_metadata: Dict[str, Any] = {
            "enabled": False,
            "applied": False,
            "reason": "bundle_retriever_unavailable",
            "version": "creator_retrieval.v1",
        }
        retriever_artifact_version = BASELINE_RETRIEVER_VERSION
        constraint_tier_used = 1
        retrieval_started = time.perf_counter()
        shortlist: List[Dict[str, Any]] = []
        if self.retriever is not None:
            retriever_query = {
                "row_id": str(query_profile.get("query_id") or "query"),
                "query_id": str(query_profile.get("query_id") or "query"),
                "text": str(query_profile.get("semantic_text") or query_profile.get("text") or ""),
                "caption": str(query.get("description") or query_profile.get("raw_text") or ""),
                "hashtags": list(query_profile.get("hashtags") or []),
                "keywords": list(query_profile.get("keywords") or []),
                "search_query": query_profile.get("topic_key"),
                "topic_key": query_profile.get("topic_key"),
                "content_type": query_profile.get("content_type"),
                "language": query_profile.get("language"),
                "locale": query_profile.get("locale"),
                "author_id": query.get("author_id"),
                "as_of_time": as_of,
            }
            retrieved_items, retrieval_meta = self.retriever.retrieve(
                query_row=retriever_query,
                candidate_rows=list(candidates) if candidates else None,
                top_k=max(1, int(retrieve_k)),
                index_cutoff_time=as_of,
                objective=effective_objective,
                candidate_ids=list(candidate_scope) if candidate_scope else None,
                retrieval_constraints={
                    "max_age_days": int(self.manifest.get("max_age_days") or 0),
                    "language": query_profile.get("language"),
                    "locale": query_profile.get("locale"),
                    "content_type": query_profile.get("content_type"),
                },
                user_context=user_context if isinstance(user_context, dict) else None,
                return_metadata=True,
            )
            prepared = self._prepare_retrieved_candidates(
                retrieved_items=retrieved_items,
                as_of=as_of,
                query_profile=query_profile,
                request_candidates_by_id=request_candidates_by_id,
            )
            usable = [item for item in prepared if item["support_level"] != "low"]
            shortlist = usable[: max(1, int(retrieve_k))]
            retrieval_mode = str(retrieval_meta.get("retrieval_mode") or retrieval_mode)
            constraint_tier_used = int(retrieval_meta.get("constraint_tier_used") or 1)
            retriever_artifact_version = str(
                retrieval_meta.get("retriever_artifact_version")
                or self.retriever.artifact_version
            )
            retrieval_stats = dict(retrieval_meta)
            creator_retrieval_meta = retrieval_meta.get("creator_retrieval")
            if isinstance(creator_retrieval_meta, dict):
                retrieval_personalization_metadata = dict(creator_retrieval_meta)

        if not shortlist:
            prepared = []
            for item in candidates:
                row = self._prepare_candidate(payload=item, as_of=as_of, query_profile=query_profile)
                if row is None:
                    continue
                if candidate_scope and row["candidate_id"] not in candidate_scope:
                    continue
                prepared.append(row)
            usable = [item for item in prepared if item["support_level"] != "low"]
            shortlist, retrieval_stats = retrieve_shortlist(
                usable_candidates=usable,
                query_profile=query_profile,
                retrieve_k=retrieve_k,
            )
            retrieval_personalization_metadata = {
                "enabled": False,
                "applied": False,
                "reason": "fallback_shortlist_retrieval",
                "version": "creator_retrieval.v1",
            }

        retrieval_elapsed_ms = (time.perf_counter() - retrieval_started) * 1000.0
        retrieval_budget_ms = _as_float(
            ((routing_payload.get("stage_budgets_ms") or {}).get("retrieval") if isinstance(routing_payload.get("stage_budgets_ms"), dict) else 260.0),
            260.0,
        )
        if retrieval_elapsed_ms > retrieval_budget_ms:
            raise RecommenderStageTimeoutError("retrieval", retrieval_elapsed_ms, retrieval_budget_ms)

        ranking_started = time.perf_counter()
        ranked, ranking_meta = rank_shortlist(
            shortlist=shortlist,
            query_profile=query_profile,
            effective_objective=effective_objective,
            portfolio=portfolio,
            rankers_available=self.rankers.keys(),
        )
        learned_reranker_metadata: Dict[str, Any] = {
            "enabled": effective_objective in self.learned_rerankers,
            "applied": False,
            "reason": None,
            "ranker_id": LEARNED_RERANKER_ID,
            "version": LEARNED_RERANKER_VERSION,
            "load_warning": self.learned_reranker_load_warnings.get(effective_objective),
        }
        if effective_objective in self.learned_rerankers:
            if bool((portfolio or {}).get("enabled", False)):
                learned_reranker_metadata["reason"] = "portfolio_mode_uses_baseline"
            else:
                ranked, learned_meta = self.learned_rerankers[effective_objective].rerank_items(
                    items=ranked
                )
                learned_reranker_metadata.update(learned_meta)

        user_adaptation_metadata: Dict[str, Any] = {
            "enabled": bool(user_affinity_context.get("enabled")),
            "applied": False,
            "reason": None,
            "version": USER_AFFINITY_VERSION,
            "creator_id": user_affinity_context.get("creator_id"),
        }
        if not bool((portfolio or {}).get("enabled", False)):
            ranked, user_meta = apply_user_affinity_blend(
                items=ranked,
                affinity_context=user_affinity_context,
            )
            user_adaptation_metadata.update(user_meta)
        else:
            user_adaptation_metadata["reason"] = "portfolio_mode_uses_global_order"

        ranking_elapsed_ms = (time.perf_counter() - ranking_started) * 1000.0
        ranking_budget_ms = _as_float(
            ((routing_payload.get("stage_budgets_ms") or {}).get("ranking") if isinstance(routing_payload.get("stage_budgets_ms"), dict) else 260.0),
            260.0,
        )
        if ranking_elapsed_ms > ranking_budget_ms:
            raise RecommenderStageTimeoutError("ranking", ranking_elapsed_ms, ranking_budget_ms)

        policy_payload = policy_overrides if isinstance(policy_overrides, dict) else {}
        manifest_policy = self.manifest.get("policy_reranker")
        manifest_policy = manifest_policy if isinstance(manifest_policy, dict) else {}

        portfolio_supported = bool(ranking_meta["portfolio_supported"])
        portfolio_requested = bool(ranking_meta["portfolio_requested"])
        portfolio_fallback_reason = ranking_meta["portfolio_fallback_reason"]
        portfolio_weights = ranking_meta["portfolio_weights"]
        risk_aversion = float(ranking_meta["risk_aversion"])
        ranking_weights = ranking_meta["weights"]

        policy_config = PolicyRerankerConfig.from_payload({
            **manifest_policy,
            **policy_payload,
        })
        policy_reranker = PolicyReranker(config=policy_config)
        for item in ranked:
            item["_author_id"] = item.get("author_id", "unknown")
            item["_topic_key"] = item.get("topic_key", "unknown")
            item["_language"] = item.get("language")
            item["_locale"] = item.get("locale")
            item["_candidate_as_of_time"] = item.get("posted_at")
        selected, policy_meta = policy_reranker.rerank(
            ranked_items=ranked,
            query_context={
                "as_of_time": as_of,
                "language": query_profile.get("language"),
                "locale": query_profile.get("locale"),
            },
            top_k=top_k,
            overrides=policy_payload,
            portfolio=portfolio if portfolio_supported else None,
        )
        dropped_by_rule = policy_meta.get("dropped_by_rule", {})

        fallback_mode = False
        fallback_reason: Optional[str] = None
        explainability_enabled = bool(explainability_payload.get("enabled", False))
        explainability_metadata: Optional[Dict[str, Any]] = None
        if explainability_enabled and selected:
            explain_started = time.perf_counter()
            explain_budget_ms = _as_float(
                ((routing_payload.get("stage_budgets_ms") or {}).get("explainability") if isinstance(routing_payload.get("stage_budgets_ms"), dict) else 300.0),
                300.0,
            )
            if explain_budget_ms <= 0.001:
                fallback_mode = True
                fallback_reason = "explainability_timeout"
            else:
                self._apply_explainability(
                    query_profile=query_profile,
                    items=selected,
                    run_counterfactuals=bool(explainability_payload.get("run_counterfactuals", False)),
                )
                explain_elapsed_ms = (time.perf_counter() - explain_started) * 1000.0
                if explain_elapsed_ms > explain_budget_ms:
                    fallback_mode = True
                    fallback_reason = "explainability_timeout"
                    for item in selected:
                        item.pop("evidence_cards", None)
                        item.pop("temporal_confidence_band", None)
                        item.pop("counterfactual_scenarios", None)
                else:
                    explainability_metadata = {
                        "enabled": True,
                        "engine": "deterministic_rag.v1",
                        "counterfactuals_enabled": bool(explainability_payload.get("run_counterfactuals", False)),
                    }

        latency_breakdown_ms = {
            "retrieval": _round(retrieval_elapsed_ms, 6),
            "ranking": _round(ranking_elapsed_ms, 6),
            "total": _round((time.perf_counter() - started_total) * 1000.0, 6),
        }

        top_trajectory = selected[0].get("trajectory_trace") if selected else {}
        trajectory_prediction = {
            "regime_pred": str((top_trajectory or {}).get("regime_pred") or "balanced"),
            "regime_confidence": _sanitize_probability((top_trajectory or {}).get("regime_confidence"), 0.0),
        }

        items_payload: List[Dict[str, Any]] = []
        for rank, item in enumerate(selected, start=1):
            raw_branch_scores = (
                item.get("retrieval_branch_scores")
                if isinstance(item.get("retrieval_branch_scores"), dict)
                else {}
            )
            fused_retrieval = float(
                raw_branch_scores.get("fused_retrieval", raw_branch_scores.get("fused", 0.0))
            )
            graph_dense_score = (
                _sanitize_probability(raw_branch_scores.get("graph_dense"), 0.0)
                if graph_enabled
                else 0.0
            )
            trajectory_dense_score = (
                _sanitize_probability(raw_branch_scores.get("trajectory_dense"), 0.0)
                if trajectory_enabled
                else 0.0
            )
            branch_scores = {
                "semantic": _round(raw_branch_scores.get("semantic", 0.0), 6),
                "hashtag_topic": _round(raw_branch_scores.get("hashtag_topic", 0.0), 6),
                "structured_compatibility": _round(
                    raw_branch_scores.get("structured_compatibility", 0.0), 6
                ),
                "fused_retrieval": _round(fused_retrieval, 6),
                "lexical": _round(
                    raw_branch_scores.get("lexical", raw_branch_scores.get("hashtag_topic", 0.0)),
                    6,
                ),
                "dense_text": _round(
                    raw_branch_scores.get("dense_text", raw_branch_scores.get("semantic", 0.0)),
                    6,
                ),
                "multimodal": _round(raw_branch_scores.get("multimodal", 0.0), 6),
                "graph_dense": _round(graph_dense_score, 6),
                "trajectory_dense": _round(trajectory_dense_score, 6),
                "fused": _round(raw_branch_scores.get("fused", fused_retrieval), 6),
            }
            eng = item.get("engagement_metrics") or {}
            payload = {
                "candidate_id": item["candidate_id"],
                "candidate_row_id": item["candidate_row_id"],
                "author_id": item.get("author_id"),
                "caption": str(item.get("raw_text") or item.get("text") or ""),
                "topic_key": item.get("topic_key"),
                "content_type": item.get("content_type"),
                "language": item.get("language"),
                "locale": item.get("locale"),
                "hashtags": list(item.get("hashtags") or []),
                "keywords": list(item.get("keywords") or []),
                "views": int(eng.get("views", 0)),
                "likes": int(eng.get("likes", 0)),
                "comments_count": int(eng.get("comments", 0)),
                "shares": int(eng.get("shares", 0)),
                "engagement_rate": float(eng.get("engagement_rate", 0.0)),
                "rank": rank,
                "score": _round(item["score"], 6),
                "score_raw": _round(item["score_raw"], 6),
                "score_calibrated": _round(item["score_calibrated"], 6),
                "score_mean": _round(item["score_mean"], 6),
                "score_std": _round(item["score_std"], 6),
                "confidence": _round(item["confidence"], 6),
                "selected_ranker_id": str(item["selected_ranker_id"]),
                "baseline_score": _round(
                    item.get("baseline_score", item.get("score", 0.0)), 6
                ),
                "learned_score": _round(item.get("learned_score", item.get("score", 0.0)), 6),
                "user_affinity_score": _round(
                    item.get("user_affinity_score", 0.5), 6
                ),
                "creator_retrieval_score": _round(
                    item.get("creator_retrieval_score", 0.0), 6
                ),
                "global_score_mean": _round(item["global_score_mean"], 6),
                "segment_blend_weight": _round(item["segment_blend_weight"], 6),
                "policy_penalty": _round(item["policy_penalty"], 6),
                "policy_bonus": _round(item["policy_bonus"], 6),
                "policy_adjusted_score": _round(item["policy_adjusted_score"], 6),
                "calibration_trace": item["calibration_trace"],
                "policy_trace": item["policy_trace"],
                "portfolio_trace": item.get("portfolio_trace"),
                "similarity": {
                    "sparse": _round(item["score_components"]["semantic_relevance"], 6),
                    "dense": _round(item["score_components"]["intent_alignment"], 6),
                    "fused": _round(item["score"], 6),
                },
                "retrieval_branch_scores": branch_scores,
                "comment_trace": item["comment_trace"],
                "trajectory_trace": {
                    "trajectory_manifest_id": self.trajectory_manifest_id or None,
                    "trajectory_version": self.trajectory_version,
                    "trajectory_mode": trajectory_mode,
                    "available": bool(item.get("trajectory_trace")),
                    "similarity": _round(trajectory_dense_score, 6),
                    "regime_pred": str((item.get("trajectory_trace") or {}).get("regime_pred") or "balanced"),
                    "regime_probabilities": dict((item.get("trajectory_trace") or {}).get("regime_probabilities") or {}),
                    "regime_confidence": _sanitize_probability((item.get("trajectory_trace") or {}).get("regime_confidence"), 0.0),
                },
                "trajectory_score": _round(trajectory_dense_score, 6),
                "trajectory_similarity": _round(trajectory_dense_score, 6),
                "trajectory_regime_pred": str((item.get("trajectory_trace") or {}).get("regime_pred") or "balanced"),
                "trajectory_regime_probabilities": dict((item.get("trajectory_trace") or {}).get("regime_probabilities") or {}),
                "trajectory_regime_confidence": _sanitize_probability((item.get("trajectory_trace") or {}).get("regime_confidence"), 0.0),
                "support_level": item["support_level"],
                "support_score": _round(item["support_score"], 6),
                "score_components": item["score_components"],
                "ranking_reasons": list(item["ranking_reasons"]),
            }
            if "learned_trace" in item:
                payload["learned_trace"] = item["learned_trace"]
            if "user_affinity_trace" in item:
                payload["user_affinity_trace"] = item["user_affinity_trace"]
            if "creator_retrieval_trace" in item:
                payload["creator_retrieval_trace"] = item["creator_retrieval_trace"]
            if "evidence_cards" in item:
                payload["evidence_cards"] = item["evidence_cards"]
            if "temporal_confidence_band" in item:
                payload["temporal_confidence_band"] = item["temporal_confidence_band"]
            if "counterfactual_scenarios" in item:
                payload["counterfactual_scenarios"] = item["counterfactual_scenarios"]
            items_payload.append(payload)

        calibration_warning = self.calibration_load_warnings.get(effective_objective)
        response: Dict[str, Any] = {
            "objective": requested_objective,
            "objective_effective": effective_objective,
            "generated_at": _to_utc_iso(datetime.now(timezone.utc)),
            "bundle_id": self.bundle_id,
            "bundle_dir": str(self.bundle_dir),
            "retrieval_mode": retrieval_mode,
            "constraint_tier_used": constraint_tier_used,
            "retriever_artifact_version": retriever_artifact_version,
            "graph_bundle_id": self.graph_bundle_id,
            "graph_version": self.graph_version,
            "graph_coverage": 0.0,
            "graph_fallback_mode": graph_fallback_mode,
            "trajectory_manifest_id": self.trajectory_manifest_id,
            "trajectory_version": self.trajectory_version,
            "trajectory_mode": trajectory_mode,
            "trajectory_prediction": trajectory_prediction,
            "portfolio_mode": bool(portfolio_supported),
            "portfolio_metadata": {
                "enabled_requested": bool(portfolio_requested),
                "enabled": bool(portfolio_supported),
                "weights": {key: _round(value, 6) for key, value in portfolio_weights.items()},
                "risk_aversion": _round(risk_aversion, 6),
                "candidate_pool_cap": _as_int(ranking_meta.get("candidate_pool_cap"), 120),
                "fallback_reason": portfolio_fallback_reason,
            },
            "comment_intelligence_version": BASELINE_COMMENT_VERSION,
            "calibration_version": BASELINE_CALIBRATION_VERSION,
            "calibration_metadata": {
                "calibrator_available": effective_objective in self.calibrators,
                "load_warning": calibration_warning,
                "engine": "identity",
            },
            "retrieval_personalization_metadata": retrieval_personalization_metadata,
            "learned_reranker_metadata": learned_reranker_metadata,
            "user_adaptation_metadata": user_adaptation_metadata,
            "policy_version": BASELINE_POLICY_VERSION,
            "policy_metadata": {
                "strict_language": bool(policy_meta.get("strict_language", False)),
                "strict_locale": bool(policy_meta.get("strict_locale", False)),
                "max_items_per_author": _as_int(policy_meta.get("max_items_per_author"), 2),
                "dropped_by_rule": dropped_by_rule,
            },
            "routing_decision": routing_decision,
            "compatibility_status": self.compatibility_payload(
                required_compat=routing_decision.get("required_compat")
                if isinstance(routing_decision.get("required_compat"), dict)
                else None
            ),
            "fallback_mode": fallback_mode,
            "fallback_reason": fallback_reason,
            "latency_breakdown_ms": latency_breakdown_ms,
            "request_id": routing_decision.get("request_id"),
            "experiment_id": (routing_decision.get("experiment") or {}).get("id"),
            "variant": (routing_decision.get("experiment") or {}).get("variant"),
            "items": items_payload,
        }
        if explainability_metadata is not None:
            response["explainability_metadata"] = explainability_metadata
        if debug:
            response["debug"] = {
                "retrieved_universe": [item["candidate_id"] for item in shortlist],
                "ranking_universe": [item["candidate_id"] for item in ranked],
                "comment_index_loaded": self.comment_index is not None,
                "comment_index_source_path": self.comment_index_source_path,
                "comment_index_load_error": self.comment_index_load_error,
                "config": {
                    "retrieval": {
                        "semantic_branch_top_k": SEMANTIC_BRANCH_TOP_K,
                        "hashtag_topic_branch_top_k": HASHTAG_BRANCH_TOP_K,
                        "structured_branch_top_k": STRUCTURED_BRANCH_TOP_K,
                        "shortlist_top_k": SHORTLIST_TOP_K,
                        "stats": retrieval_stats,
                        "personalization": retrieval_personalization_metadata,
                        "retriever_loaded": self.retriever is not None,
                        "retriever_load_warning": self.retriever_load_warning,
                    },
                    "ranking": {
                        "default_weights": DEFAULT_RANKING_WEIGHTS,
                        "objective_weights": OBJECTIVE_RANKING_WEIGHTS,
                        "learned_reranker": learned_reranker_metadata,
                        "user_adaptation": user_adaptation_metadata,
                    },
                    "support": {
                        "low_excluded": True,
                        "full_threshold": SUPPORT_FULL_THRESHOLD,
                        "partial_threshold": SUPPORT_PARTIAL_THRESHOLD,
                    },
                },
            }
        return response
