from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field, model_validator

from ..contracts import CanonicalComment, CanonicalCommentSnapshot, CanonicalDatasetBundle

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


COMMENT_INTELLIGENCE_VERSION = "comment_intelligence.v2"
COMMENT_TAXONOMY_VERSION = "comment_taxonomy.v2.0.0"
ALIGNMENT_METHOD_VERSION = "alignment.v2.hybrid.lexical+embedding"
INTENT_LABELS = (
    "confusion",
    "help_seeking",
    "purchase_intent",
    "save_intent",
    "praise",
    "complaint",
    "skepticism",
    "off_topic",
)

MissingReason = Literal[
    "not_available",
    "low_quality",
    "extraction_failed",
    "not_applicable",
]
QualityTier = Literal["gold", "silver", "bronze", "quarantine"]

POSITIVE_WORDS = {
    "amazing",
    "awesome",
    "best",
    "clean",
    "cool",
    "excellent",
    "fire",
    "good",
    "great",
    "helpful",
    "impressive",
    "love",
    "nice",
    "perfect",
    "solid",
    "useful",
    "wow",
}

NEGATIVE_WORDS = {
    "bad",
    "broken",
    "confusing",
    "dislike",
    "frustrating",
    "hard",
    "hate",
    "issue",
    "messy",
    "problem",
    "slow",
    "stupid",
    "terrible",
    "unclear",
    "useless",
    "wrong",
}

CONFUSION_PATTERNS = (
    r"\bconfus(ed|ing)\b",
    r"\b(not sure|dont understand|don't understand)\b",
    r"\b(what do you mean|which one)\b",
    r"\b(unclear|lost)\b",
    r"\?",
)

HELP_PATTERNS = (
    r"\bhow\b",
    r"\bcan you\b",
    r"\bplease\b",
    r"\bneed help\b",
    r"\bshow (me|us)\b",
    r"\bsteps?\b",
)

PURCHASE_PATTERNS = (
    r"\b(price|cost|buy|link|where to get|checkout)\b",
    r"\bworth it\b",
    r"\bdiscount\b",
)

SAVE_PATTERNS = (
    r"\bsaved\b",
    r"\bsaving\b",
    r"\bbookmarked\b",
    r"\bcome back\b",
)

PRAISE_PATTERNS = (
    r"\b(love|great|awesome|amazing|helpful|fire|clean)\b",
    r"\bthanks\b",
)

COMPLAINT_PATTERNS = (
    r"\b(doesn't work|doesnt work|broken|bad|problem|issue)\b",
    r"\bwaste\b",
    r"\bunhelpful\b",
)

SKEPTICISM_PATTERNS = (
    r"\b(fake|cap|doubt|really\?|sure\?)\b",
    r"\btoo good to be true\b",
)

OFFTOPIC_PATTERNS = (
    r"\b(first|early)\b",
    r"\bwho else\b",
    r"\balgorithm\b",
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "this",
    "to",
    "too",
    "up",
    "we",
    "what",
    "when",
    "where",
    "why",
    "with",
    "you",
    "your",
}

ARTIFACT_TOKENS = {
    "algorithm",
    "camera",
    "edit",
    "editing",
    "fyp",
    "lighting",
    "mic",
    "music",
    "quality",
    "sound",
    "thumbnail",
    "voice",
}

_EMBEDDING_ENCODER_CACHE: Dict[str, Any] = {}


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return _to_utc(dt).isoformat().replace("+00:00", "Z")


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9#@? ]+", " ", value.lower())).strip()


def _tokens(value: str) -> List[str]:
    return [token for token in _norm_text(value).split(" ") if token]


def _content_tokens(value: str) -> List[str]:
    out: List[str] = []
    for token in _tokens(value):
        clean = token.strip("#").strip().lower()
        if len(clean) < 3:
            continue
        if clean in STOPWORDS:
            continue
        out.append(clean)
    return out


def _match_any(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text) is not None for pattern in patterns)


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isfinite(out):
        return out
    return fallback


def _vector_norm(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.linalg.norm(values))


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    norm = _vector_norm(values)
    if norm <= 0.0:
        return values.astype(np.float32)
    return (values / norm).astype(np.float32)


def _hash_embedding(text: str, dim: int) -> np.ndarray:
    vec = np.zeros((max(4, int(dim)),), dtype=np.float32)
    toks = _content_tokens(text)
    if not toks:
        return vec
    for token in toks:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], byteorder="big", signed=False) % len(vec)
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    return _l2_normalize(vec)


def _load_sentence_encoder(model_name: str) -> Optional[Any]:
    cached = _EMBEDDING_ENCODER_CACHE.get(model_name)
    if cached is not None:
        return cached
    if SentenceTransformer is None:  # pragma: no cover - optional dependency
        _EMBEDDING_ENCODER_CACHE[model_name] = None
        return None
    try:
        encoder = SentenceTransformer(  # pragma: no cover - external model
            model_name,
            device="cpu",
            local_files_only=True,
        )
    except Exception:
        encoder = None
    _EMBEDDING_ENCODER_CACHE[model_name] = encoder
    return encoder


def _embed_texts(
    texts: Sequence[str],
    *,
    model_name: str,
    hash_dim: int,
) -> Tuple[List[np.ndarray], str]:
    if not texts:
        return [], "none"
    encoder = _load_sentence_encoder(model_name)
    if encoder is not None:
        try:  # pragma: no cover - model runtime branch
            raw = encoder.encode(
                [str(item) for item in texts],
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            matrix = np.asarray(raw, dtype=np.float32)
            return [matrix[idx] for idx in range(matrix.shape[0])], "sentence_transformers_local"
        except Exception:
            pass
    return [_hash_embedding(str(item), hash_dim) for item in texts], "hash_embedding"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = _vector_norm(a) * _vector_norm(b)
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _infer_content_type_bucket(caption: str, hashtags: Sequence[str]) -> str:
    text = f"{caption} {' '.join(hashtags)}".lower()
    if any(token in text for token in ("tutorial", "how to", "#tutorial", "step")):
        return "tutorial"
    if any(token in text for token in ("story", "pov", "#storytime")):
        return "story"
    if any(token in text for token in ("review", "unboxing", "#review")):
        return "review"
    return "general"


def _author_size_bucket(followers_count: int) -> str:
    if followers_count < 10_000:
        return "nano"
    if followers_count < 100_000:
        return "micro"
    if followers_count < 1_000_000:
        return "macro"
    return "mega"


def _infer_topic_key(search_query: Optional[str], hashtags: Sequence[str], caption: str) -> str:
    if isinstance(search_query, str) and search_query.strip():
        return search_query.strip().lower()
    if hashtags:
        return hashtags[0].replace("#", "").strip().lower() or "general"
    words = _tokens(caption)
    return words[0] if words else "general"


def _intent_for_comment(text: str) -> str:
    normalized = _norm_text(text)
    if _match_any(normalized, CONFUSION_PATTERNS):
        return "confusion"
    if _match_any(normalized, HELP_PATTERNS):
        return "help_seeking"
    if _match_any(normalized, PURCHASE_PATTERNS):
        return "purchase_intent"
    if _match_any(normalized, SAVE_PATTERNS):
        return "save_intent"
    if _match_any(normalized, PRAISE_PATTERNS):
        return "praise"
    if _match_any(normalized, COMPLAINT_PATTERNS):
        return "complaint"
    if _match_any(normalized, SKEPTICISM_PATTERNS):
        return "skepticism"
    if _match_any(normalized, OFFTOPIC_PATTERNS):
        return "off_topic"
    return "off_topic"


def _sentiment_polarity(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    positive = sum(1 for token in toks if token in POSITIVE_WORDS)
    negative = sum(1 for token in toks if token in NEGATIVE_WORDS)
    score = _safe_div(float(positive - negative), float(max(1, len(toks))))
    return _clamp(score, -1.0, 1.0)


class CommentIntelligenceConfig(BaseModel):
    early_window_hours: int = Field(default=24, ge=1, le=96)
    late_window_hours: int = Field(default=96, ge=2, le=240)
    taxonomy_version: str = COMMENT_TAXONOMY_VERSION
    min_comments_for_stable: int = Field(default=3, ge=1)
    confidence_floor: float = Field(default=0.2, ge=0.0, le=1.0)
    alignment_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    alignment_embedding_hash_dim: int = Field(default=64, ge=8, le=512)
    alignment_semantic_threshold: float = Field(default=0.55, ge=0.1, le=0.95)
    alignment_lexical_weight: float = Field(default=0.55, ge=0.0, le=1.0)
    alignment_semantic_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    alignment_drift_penalty_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_alignment_weights(self) -> "CommentIntelligenceConfig":
        if (self.alignment_lexical_weight + self.alignment_semantic_weight) <= 0.0:
            raise ValueError("alignment lexical+semantic weights must sum to > 0.")
        return self


class CommentFeatureQuality(BaseModel):
    quality_score: float = Field(ge=0.0, le=1.0)
    quality_tier: QualityTier
    quality_reasons: List[str] = Field(default_factory=list)
    required_fields_ok: bool = True


class CommentMissingness(BaseModel):
    reason: MissingReason
    detail: Optional[str] = None


class CommentSignalFeatures(BaseModel):
    comment_count_total: int = Field(ge=0)
    comment_count_early: int = Field(ge=0)
    comment_count_late: int = Field(ge=0)
    intent_rates: Dict[str, float] = Field(default_factory=dict)
    dominant_intents: List[str] = Field(default_factory=list)
    confusion_index: float = Field(ge=0.0, le=1.0)
    help_seeking_index: float = Field(ge=0.0, le=1.0)
    sentiment_mean: float = Field(ge=-1.0, le=1.0)
    sentiment_volatility: float = Field(ge=0.0)
    sentiment_shift_early_late: float = Field(ge=-2.0, le=2.0)
    reply_depth_max: float = Field(ge=0.0)
    reply_branch_factor: float = Field(ge=0.0)
    reply_ratio: float = Field(ge=0.0, le=1.0)
    root_thread_concentration: float = Field(ge=0.0, le=1.0)
    alignment_score: float = Field(ge=0.0, le=1.0)
    value_prop_coverage: float = Field(ge=0.0, le=1.0)
    on_topic_ratio: float = Field(ge=0.0, le=1.0)
    artifact_drift_ratio: float = Field(ge=0.0, le=1.0)
    alignment_shift_early_late: float = Field(ge=-1.0, le=1.0)
    alignment_confidence: float = Field(ge=0.0, le=1.0)
    alignment_method_version: str = ALIGNMENT_METHOD_VERSION
    confidence: float = Field(ge=0.0, le=1.0)


class CommentIntelligenceSnapshot(BaseModel):
    comment_intelligence_version: str = COMMENT_INTELLIGENCE_VERSION
    taxonomy_version: str = COMMENT_TAXONOMY_VERSION
    video_id: str
    as_of_time: str
    window_spec: str
    topic_key: str
    content_type_bucket: str
    author_size_bucket: str
    features: CommentSignalFeatures
    missingness: Dict[str, CommentMissingness] = Field(default_factory=dict)
    quality: CommentFeatureQuality
    trace_id: str


class CommentTransferPriorEntry(BaseModel):
    topic_key: str
    content_type_bucket: str
    author_size_bucket: str
    support_count: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    dominant_intents: List[str] = Field(default_factory=list)
    confusion_index: float = Field(ge=0.0, le=1.0)
    help_seeking_index: float = Field(ge=0.0, le=1.0)
    sentiment_volatility: float = Field(ge=0.0)
    sentiment_shift_early_late: float = Field(ge=-2.0, le=2.0)
    reply_depth_max: float = Field(ge=0.0)
    reply_branch_factor: float = Field(ge=0.0)
    reply_ratio: float = Field(ge=0.0, le=1.0)
    root_thread_concentration: float = Field(ge=0.0, le=1.0)
    prior_alignment_score: float = Field(ge=0.0, le=1.0)
    prior_value_prop_coverage: float = Field(ge=0.0, le=1.0)
    prior_artifact_drift_ratio: float = Field(ge=0.0, le=1.0)
    prior_alignment_confidence: float = Field(ge=0.0, le=1.0)


class CommentTransferPriors(BaseModel):
    comment_intelligence_version: str = COMMENT_INTELLIGENCE_VERSION
    taxonomy_version: str = COMMENT_TAXONOMY_VERSION
    as_of_time: str
    entries: List[CommentTransferPriorEntry]


@dataclass
class _PreparedComment:
    comment_id: str
    root_id: str
    level: int
    created_at: datetime
    text: str
    intent: str
    sentiment: float
    tokens: Tuple[str, ...]
    is_early: bool
    is_late: bool
    is_reply: bool
    reply_count: int


def _prepare_comments_for_video(
    *,
    comments: Sequence[CanonicalComment],
    comment_snapshots: Sequence[CanonicalCommentSnapshot],
    posted_at: datetime,
    as_of: datetime,
    cfg: CommentIntelligenceConfig,
) -> List[_PreparedComment]:
    reply_count_by_comment: Dict[str, int] = {}
    for snapshot in comment_snapshots:
        snap_time = _to_utc(snapshot.scraped_at)
        ingest = _to_utc(snapshot.ingested_at or snapshot.scraped_at)
        if snap_time > as_of or ingest > as_of:
            continue
        prior = reply_count_by_comment.get(snapshot.comment_id, 0)
        reply_count_by_comment[snapshot.comment_id] = max(prior, int(snapshot.reply_count))

    out: List[_PreparedComment] = []
    early_end = posted_at + timedelta(hours=cfg.early_window_hours)
    late_end = posted_at + timedelta(hours=cfg.late_window_hours)
    for comment in comments:
        created = _to_utc(comment.created_at)
        ingested = _to_utc(comment.ingested_at or comment.created_at)
        if created > as_of or ingested > as_of:
            continue
        age_hours = (created - posted_at).total_seconds() / 3600.0
        if age_hours < 0 or age_hours > cfg.late_window_hours:
            continue
        text = str(comment.text or "")
        content_tokens = tuple(_content_tokens(text))
        out.append(
            _PreparedComment(
                comment_id=comment.comment_id,
                root_id=comment.root_comment_id or comment.comment_id,
                level=int(comment.comment_level or 0),
                created_at=created,
                text=text,
                intent=_intent_for_comment(text),
                sentiment=_sentiment_polarity(text),
                tokens=content_tokens,
                is_early=created <= early_end,
                is_late=created > early_end and created <= late_end,
                is_reply=bool(comment.parent_comment_id),
                reply_count=int(reply_count_by_comment.get(comment.comment_id, 0)),
            )
        )
    return out


def _quality_for_snapshot(
    comments_total: int,
    missing: Dict[str, CommentMissingness],
    confidence: float,
) -> CommentFeatureQuality:
    reasons: List[str] = []
    score = confidence
    if comments_total == 0:
        reasons.append("no_comments")
        score -= 0.45
    elif comments_total < 3:
        reasons.append("low_comment_volume")
        score -= 0.2
    if missing:
        reasons.extend(sorted(missing.keys()))
        score -= min(0.3, 0.05 * len(missing))
    score = _clamp(score, 0.0, 1.0)
    if score >= 0.85:
        tier: QualityTier = "gold"
    elif score >= 0.65:
        tier = "silver"
    elif score >= 0.4:
        tier = "bronze"
    else:
        tier = "quarantine"
    return CommentFeatureQuality(
        quality_score=round(score, 6),
        quality_tier=tier,
        quality_reasons=reasons,
        required_fields_ok=comments_total > 0,
    )


def _build_intended_value_props(
    *,
    caption: str,
    hashtags: Sequence[str],
    keywords: Sequence[str],
    search_query: Optional[str],
    transcript_text: Optional[str],
    ocr_text: Optional[str],
) -> List[str]:
    source_parts: List[str] = [caption]
    source_parts.extend(str(item).replace("#", " ") for item in list(hashtags or []))
    source_parts.extend(str(item) for item in list(keywords or []))
    if isinstance(search_query, str):
        source_parts.append(search_query)
    if isinstance(transcript_text, str):
        source_parts.append(transcript_text)
    if isinstance(ocr_text, str):
        source_parts.append(ocr_text)
    ranked: Dict[str, int] = {}
    for token in _content_tokens(" ".join(source_parts)):
        ranked[token] = ranked.get(token, 0) + 1
    ordered = sorted(ranked.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ordered[:24]]


@dataclass
class _AlignmentContext:
    value_props: Tuple[str, ...]
    embedding_backend: str
    method_version: str
    anchor_embedding: np.ndarray
    comment_embeddings: Dict[str, np.ndarray]


@dataclass
class _AlignmentWindow:
    alignment_score: float
    value_prop_coverage: float
    on_topic_ratio: float
    artifact_drift_ratio: float


def _compute_alignment_window(
    *,
    comments: Sequence[_PreparedComment],
    cfg: CommentIntelligenceConfig,
    context: _AlignmentContext,
) -> _AlignmentWindow:
    if not comments or not context.value_props:
        return _AlignmentWindow(0.0, 0.0, 0.0, 0.0)
    props = set(context.value_props)
    covered: set[str] = set()
    on_topic = 0
    artifact_drift = 0
    semantic_values: List[float] = []
    semantic_on_topic = 0
    for item in comments:
        toks = set(item.tokens)
        overlap = toks & props
        if overlap:
            covered.update(overlap)
            on_topic += 1
        if toks and (len(toks & ARTIFACT_TOKENS) > 0) and not overlap:
            artifact_drift += 1
        emb = context.comment_embeddings.get(item.comment_id)
        if emb is not None and emb.size > 0 and context.anchor_embedding.size > 0:
            sim = _cosine_similarity(context.anchor_embedding, emb)
            sim01 = _clamp((sim + 1.0) * 0.5, 0.0, 1.0)
            semantic_values.append(sim01)
            if sim01 >= cfg.alignment_semantic_threshold:
                semantic_on_topic += 1
    coverage = _safe_div(float(len(covered)), float(max(1, len(props))))
    on_topic_ratio = _safe_div(float(on_topic), float(max(1, len(comments))))
    drift_ratio = _safe_div(float(artifact_drift), float(max(1, len(comments))))
    semantic_mean = (
        _safe_div(sum(semantic_values), float(max(1, len(semantic_values))))
        if semantic_values
        else 0.0
    )
    semantic_ratio = _safe_div(float(semantic_on_topic), float(max(1, len(comments))))
    lexical_component = (0.6 * coverage) + (0.4 * on_topic_ratio)
    semantic_component = (0.7 * semantic_mean) + (0.3 * semantic_ratio)
    blend_denominator = max(
        1e-6,
        cfg.alignment_lexical_weight + cfg.alignment_semantic_weight,
    )
    alignment_score = (
        ((cfg.alignment_lexical_weight * lexical_component) + (cfg.alignment_semantic_weight * semantic_component))
        / blend_denominator
    ) - (cfg.alignment_drift_penalty_weight * drift_ratio)
    return _AlignmentWindow(
        alignment_score=_clamp(float(alignment_score), 0.0, 1.0),
        value_prop_coverage=_clamp(float(coverage), 0.0, 1.0),
        on_topic_ratio=_clamp(float(on_topic_ratio), 0.0, 1.0),
        artifact_drift_ratio=_clamp(float(drift_ratio), 0.0, 1.0),
    )


def _build_alignment_context(
    *,
    value_props: Sequence[str],
    comments: Sequence[_PreparedComment],
    cfg: CommentIntelligenceConfig,
) -> _AlignmentContext:
    if not value_props:
        return _AlignmentContext(
            value_props=tuple(),
            embedding_backend="none",
            method_version=f"{ALIGNMENT_METHOD_VERSION}.none",
            anchor_embedding=np.zeros((0,), dtype=np.float32),
            comment_embeddings={},
        )
    anchor_text = " ".join(value_props)
    payload = [anchor_text] + [item.text for item in comments]
    vectors, backend = _embed_texts(
        payload,
        model_name=cfg.alignment_embedding_model,
        hash_dim=cfg.alignment_embedding_hash_dim,
    )
    if not vectors:
        return _AlignmentContext(
            value_props=tuple(value_props),
            embedding_backend=backend,
            method_version=f"{ALIGNMENT_METHOD_VERSION}.{backend}",
            anchor_embedding=np.zeros((0,), dtype=np.float32),
            comment_embeddings={},
        )
    anchor = np.asarray(vectors[0], dtype=np.float32)
    comment_embeddings: Dict[str, np.ndarray] = {}
    for item, vector in zip(comments, vectors[1:]):
        comment_embeddings[item.comment_id] = np.asarray(vector, dtype=np.float32)
    return _AlignmentContext(
        value_props=tuple(value_props),
        embedding_backend=backend,
        method_version=f"{ALIGNMENT_METHOD_VERSION}.{backend}",
        anchor_embedding=anchor,
        comment_embeddings=comment_embeddings,
    )


def _alignment_confidence(
    *,
    comments_total: int,
    value_props: Sequence[str],
    alignment_score: float,
    context: _AlignmentContext,
    cfg: CommentIntelligenceConfig,
) -> float:
    confidence = cfg.confidence_floor
    confidence += min(0.45, 0.04 * float(comments_total))
    if value_props:
        confidence += 0.2
    if context.embedding_backend == "sentence_transformers_local":
        confidence += 0.2
    elif context.embedding_backend == "hash_embedding":
        confidence += 0.08
    confidence += max(0.0, 0.12 * alignment_score)
    return _clamp(confidence, cfg.confidence_floor, 1.0)


def extract_comment_intelligence_for_video(
    *,
    video_id: str,
    posted_at: datetime,
    as_of_time: datetime,
    topic_key: str,
    content_type_bucket: str,
    author_size_bucket: str,
    comments: Sequence[CanonicalComment] = (),
    comment_snapshots: Sequence[CanonicalCommentSnapshot] = (),
    caption: str = "",
    hashtags: Sequence[str] = (),
    keywords: Sequence[str] = (),
    search_query: Optional[str] = None,
    transcript_text: Optional[str] = None,
    ocr_text: Optional[str] = None,
    config: Optional[CommentIntelligenceConfig] = None,
) -> CommentIntelligenceSnapshot:
    cfg = config or CommentIntelligenceConfig()
    posted_at_utc = _to_utc(posted_at)
    as_of_utc = _to_utc(as_of_time)

    started = time.perf_counter()
    prepared = _prepare_comments_for_video(
        comments=comments,
        comment_snapshots=comment_snapshots,
        posted_at=posted_at_utc,
        as_of=as_of_utc,
        cfg=cfg,
    )
    missing: Dict[str, CommentMissingness] = {}

    counts_by_intent = {label: 0 for label in INTENT_LABELS}
    sentiments: List[float] = []
    early_sentiments: List[float] = []
    late_sentiments: List[float] = []
    root_counts: Dict[str, int] = {}
    depth_max = 0
    reply_count = 0
    reply_edges = 0

    for item in prepared:
        counts_by_intent[item.intent] = counts_by_intent.get(item.intent, 0) + 1
        sentiments.append(item.sentiment)
        if item.is_early:
            early_sentiments.append(item.sentiment)
        if item.is_late:
            late_sentiments.append(item.sentiment)
        root_counts[item.root_id] = root_counts.get(item.root_id, 0) + 1
        depth_max = max(depth_max, int(item.level))
        reply_edges += max(0, int(item.reply_count))
        if item.is_reply:
            reply_count += 1

    total = len(prepared)
    if total == 0:
        missing["comments"] = CommentMissingness(
            reason="not_available",
            detail="No comments available in early/late windows at as_of_time.",
        )

    sentiments_mean = 0.0 if not sentiments else sum(sentiments) / len(sentiments)
    sentiment_volatility = 0.0
    if len(sentiments) >= 2:
        mu = sentiments_mean
        sentiment_volatility = math.sqrt(
            sum((value - mu) ** 2 for value in sentiments) / max(1, len(sentiments) - 1)
        )

    early_mean = 0.0 if not early_sentiments else sum(early_sentiments) / len(early_sentiments)
    late_mean = 0.0 if not late_sentiments else sum(late_sentiments) / len(late_sentiments)

    intent_rates = {
        label: round(_safe_div(float(counts_by_intent.get(label, 0)), float(max(1, total))), 6)
        for label in INTENT_LABELS
    }
    ranked_intents = sorted(
        intent_rates.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    dominant_intents = [label for label, rate in ranked_intents if rate > 0][:3]
    if not dominant_intents:
        dominant_intents = ["off_topic"]

    confusion_index = _clamp(
        intent_rates.get("confusion", 0.0) + 0.5 * intent_rates.get("help_seeking", 0.0),
        0.0,
        1.0,
    )
    help_index = _clamp(
        intent_rates.get("help_seeking", 0.0),
        0.0,
        1.0,
    )

    root_top = max(root_counts.values()) if root_counts else 0
    root_concentration = _safe_div(float(root_top), float(max(1, total)))
    reply_ratio = _safe_div(float(reply_count), float(max(1, total)))
    reply_branch = _safe_div(float(reply_edges), float(max(1, len(root_counts))))

    value_props = _build_intended_value_props(
        caption=caption,
        hashtags=hashtags,
        keywords=keywords,
        search_query=search_query,
        transcript_text=transcript_text,
        ocr_text=ocr_text,
    )
    if not value_props:
        missing["alignment_no_content_signals"] = CommentMissingness(
            reason="not_available",
            detail="no_content_signals",
        )
    low_quality_comment_ratio = _safe_div(
        float(sum(1 for item in prepared if len(item.tokens) < 2)),
        float(max(1, total)),
    )
    if total > 0 and low_quality_comment_ratio >= 0.6:
        missing["alignment_low_quality_text"] = CommentMissingness(
            reason="low_quality",
            detail="low_quality_text",
        )
    alignment_context = _build_alignment_context(
        value_props=value_props,
        comments=prepared,
        cfg=cfg,
    )
    if total > 0 and alignment_context.embedding_backend != "sentence_transformers_local":
        missing["alignment_embedding_unavailable"] = CommentMissingness(
            reason="not_available",
            detail="embedding_unavailable",
        )
    early_comments = [item for item in prepared if item.is_early]
    late_comments = [item for item in prepared if item.is_late]
    if not early_comments or not late_comments:
        missing["alignment_window_gap"] = CommentMissingness(
            reason="not_available",
            detail="no_comments_in_window",
        )
    alignment_total = _compute_alignment_window(
        comments=prepared,
        cfg=cfg,
        context=alignment_context,
    )
    alignment_early = _compute_alignment_window(
        comments=early_comments,
        cfg=cfg,
        context=alignment_context,
    )
    alignment_late = _compute_alignment_window(
        comments=late_comments,
        cfg=cfg,
        context=alignment_context,
    )
    alignment_shift_early_late = _clamp(
        alignment_late.alignment_score - alignment_early.alignment_score,
        -1.0,
        1.0,
    )
    alignment_confidence = _alignment_confidence(
        comments_total=total,
        value_props=value_props,
        alignment_score=alignment_total.alignment_score,
        context=alignment_context,
        cfg=cfg,
    )
    confidence = _clamp(
        cfg.confidence_floor + min(0.7, 0.05 * float(total)) - min(0.25, sentiment_volatility * 0.5),
        cfg.confidence_floor,
        1.0,
    )
    confidence = _clamp((0.75 * confidence) + (0.25 * alignment_confidence), cfg.confidence_floor, 1.0)
    if total < cfg.min_comments_for_stable:
        missing["volume"] = CommentMissingness(
            reason="low_quality",
            detail=f"Only {total} comments available (< {cfg.min_comments_for_stable}).",
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    features = CommentSignalFeatures(
        comment_count_total=total,
        comment_count_early=sum(1 for item in prepared if item.is_early),
        comment_count_late=sum(1 for item in prepared if item.is_late),
        intent_rates=intent_rates,
        dominant_intents=dominant_intents,
        confusion_index=round(confusion_index, 6),
        help_seeking_index=round(help_index, 6),
        sentiment_mean=round(sentiments_mean, 6),
        sentiment_volatility=round(sentiment_volatility, 6),
        sentiment_shift_early_late=round(late_mean - early_mean, 6),
        reply_depth_max=round(float(depth_max), 6),
        reply_branch_factor=round(reply_branch, 6),
        reply_ratio=round(reply_ratio, 6),
        root_thread_concentration=round(root_concentration, 6),
        alignment_score=round(alignment_total.alignment_score, 6),
        value_prop_coverage=round(alignment_total.value_prop_coverage, 6),
        on_topic_ratio=round(alignment_total.on_topic_ratio, 6),
        artifact_drift_ratio=round(alignment_total.artifact_drift_ratio, 6),
        alignment_shift_early_late=round(alignment_shift_early_late, 6),
        alignment_confidence=round(alignment_confidence, 6),
        alignment_method_version=alignment_context.method_version,
        confidence=round(confidence, 6),
    )
    quality = _quality_for_snapshot(total, missing, confidence)
    trace_id = _sha256_text(
        _canonical_json(
            {
                "video_id": video_id,
                "as_of_time": _to_iso(as_of_utc),
                "topic_key": topic_key,
                "content_type_bucket": content_type_bucket,
                "author_size_bucket": author_size_bucket,
                "dominant_intents": dominant_intents,
                "elapsed_ms": round(elapsed_ms, 4),
            }
        )
    )
    return CommentIntelligenceSnapshot(
        taxonomy_version=cfg.taxonomy_version,
        video_id=video_id,
        as_of_time=_to_iso(as_of_utc),
        window_spec=f"early=0-{cfg.early_window_hours}h;late={cfg.early_window_hours}-{cfg.late_window_hours}h",
        topic_key=topic_key,
        content_type_bucket=content_type_bucket,
        author_size_bucket=author_size_bucket,
        features=features,
        missingness=missing,
        quality=quality,
        trace_id=trace_id,
    )


def extract_comment_intelligence_snapshots(
    *,
    bundle: CanonicalDatasetBundle,
    as_of_time: Optional[datetime] = None,
    config: Optional[CommentIntelligenceConfig] = None,
) -> List[CommentIntelligenceSnapshot]:
    cfg = config or CommentIntelligenceConfig()
    as_of_utc = _to_utc(as_of_time or bundle.generated_at)
    comments_by_video: Dict[str, List[CanonicalComment]] = {}
    for item in bundle.comments:
        comments_by_video.setdefault(item.video_id, []).append(item)
    snapshots_by_video: Dict[str, List[CanonicalCommentSnapshot]] = {}
    for item in bundle.comment_snapshots:
        snapshots_by_video.setdefault(item.video_id, []).append(item)
    authors_by_id = {item.author_id: item for item in bundle.authors}

    out: List[CommentIntelligenceSnapshot] = []
    for video in bundle.videos:
        author = authors_by_id.get(video.author_id)
        followers = int(author.followers_count if author else 0)
        out.append(
            extract_comment_intelligence_for_video(
                video_id=video.video_id,
                posted_at=video.posted_at,
                as_of_time=as_of_utc,
                topic_key=_infer_topic_key(video.search_query, video.hashtags, video.caption),
                content_type_bucket=_infer_content_type_bucket(video.caption, video.hashtags),
                author_size_bucket=_author_size_bucket(followers),
                caption=video.caption,
                hashtags=video.hashtags,
                keywords=video.keywords,
                search_query=video.search_query,
                comments=comments_by_video.get(video.video_id, []),
                comment_snapshots=snapshots_by_video.get(video.video_id, []),
                config=cfg,
            )
        )
    return out


def _aggregate_mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return float(sum(values_list) / len(values_list))


def build_transfer_priors_from_snapshots(
    *,
    snapshots: Sequence[CommentIntelligenceSnapshot],
    as_of_time: datetime,
    taxonomy_version: str,
    min_support: int = 3,
    shrinkage_alpha: float = 8.0,
) -> CommentTransferPriors:
    global_confusion = _aggregate_mean(item.features.confusion_index for item in snapshots)
    global_help = _aggregate_mean(item.features.help_seeking_index for item in snapshots)
    global_vol = _aggregate_mean(item.features.sentiment_volatility for item in snapshots)
    global_shift = _aggregate_mean(item.features.sentiment_shift_early_late for item in snapshots)
    global_depth = _aggregate_mean(item.features.reply_depth_max for item in snapshots)
    global_branch = _aggregate_mean(item.features.reply_branch_factor for item in snapshots)
    global_reply = _aggregate_mean(item.features.reply_ratio for item in snapshots)
    global_root_conc = _aggregate_mean(item.features.root_thread_concentration for item in snapshots)
    global_alignment = _aggregate_mean(item.features.alignment_score for item in snapshots)
    global_alignment_cov = _aggregate_mean(item.features.value_prop_coverage for item in snapshots)
    global_alignment_drift = _aggregate_mean(item.features.artifact_drift_ratio for item in snapshots)
    global_alignment_conf = _aggregate_mean(item.features.alignment_confidence for item in snapshots)
    global_conf = _aggregate_mean(item.features.confidence for item in snapshots)

    grouped: Dict[Tuple[str, str, str], List[CommentIntelligenceSnapshot]] = {}
    for item in snapshots:
        grouped.setdefault(
            (item.topic_key, item.content_type_bucket, item.author_size_bucket),
            [],
        ).append(item)

    entries: List[CommentTransferPriorEntry] = []
    for key, items in grouped.items():
        support = len(items)
        if support < min_support:
            continue
        ratio = float(support) / float(support + max(0.0, shrinkage_alpha))

        def shrunk(local: float, global_value: float) -> float:
            return ratio * local + (1.0 - ratio) * global_value

        intent_pool: Dict[str, float] = {label: 0.0 for label in INTENT_LABELS}
        for item in items:
            for label in INTENT_LABELS:
                intent_pool[label] += float(item.features.intent_rates.get(label, 0.0))
        for label in intent_pool:
            intent_pool[label] = _safe_div(intent_pool[label], float(max(1, support)))
        dominant = [
            label
            for label, _ in sorted(intent_pool.items(), key=lambda x: x[1], reverse=True)
            if intent_pool[label] > 0
        ][:3]
        if not dominant:
            dominant = ["off_topic"]

        entries.append(
            CommentTransferPriorEntry(
                topic_key=key[0],
                content_type_bucket=key[1],
                author_size_bucket=key[2],
                support_count=support,
                confidence=round(_clamp(shrunk(_aggregate_mean(it.features.confidence for it in items), global_conf), 0.0, 1.0), 6),
                dominant_intents=dominant,
                confusion_index=round(_clamp(shrunk(_aggregate_mean(it.features.confusion_index for it in items), global_confusion), 0.0, 1.0), 6),
                help_seeking_index=round(_clamp(shrunk(_aggregate_mean(it.features.help_seeking_index for it in items), global_help), 0.0, 1.0), 6),
                sentiment_volatility=round(max(0.0, shrunk(_aggregate_mean(it.features.sentiment_volatility for it in items), global_vol)), 6),
                sentiment_shift_early_late=round(_clamp(shrunk(_aggregate_mean(it.features.sentiment_shift_early_late for it in items), global_shift), -2.0, 2.0), 6),
                reply_depth_max=round(max(0.0, shrunk(_aggregate_mean(it.features.reply_depth_max for it in items), global_depth)), 6),
                reply_branch_factor=round(max(0.0, shrunk(_aggregate_mean(it.features.reply_branch_factor for it in items), global_branch)), 6),
                reply_ratio=round(_clamp(shrunk(_aggregate_mean(it.features.reply_ratio for it in items), global_reply), 0.0, 1.0), 6),
                root_thread_concentration=round(_clamp(shrunk(_aggregate_mean(it.features.root_thread_concentration for it in items), global_root_conc), 0.0, 1.0), 6),
                prior_alignment_score=round(
                    _clamp(
                        shrunk(
                            _aggregate_mean(it.features.alignment_score for it in items),
                            global_alignment,
                        ),
                        0.0,
                        1.0,
                    ),
                    6,
                ),
                prior_value_prop_coverage=round(
                    _clamp(
                        shrunk(
                            _aggregate_mean(it.features.value_prop_coverage for it in items),
                            global_alignment_cov,
                        ),
                        0.0,
                        1.0,
                    ),
                    6,
                ),
                prior_artifact_drift_ratio=round(
                    _clamp(
                        shrunk(
                            _aggregate_mean(it.features.artifact_drift_ratio for it in items),
                            global_alignment_drift,
                        ),
                        0.0,
                        1.0,
                    ),
                    6,
                ),
                prior_alignment_confidence=round(
                    _clamp(
                        shrunk(
                            _aggregate_mean(it.features.alignment_confidence for it in items),
                            global_alignment_conf,
                        ),
                        0.0,
                        1.0,
                    ),
                    6,
                ),
            )
        )

    entries.sort(key=lambda item: (item.topic_key, item.content_type_bucket, item.author_size_bucket))
    return CommentTransferPriors(
        taxonomy_version=taxonomy_version,
        as_of_time=_to_iso(as_of_time),
        entries=entries,
    )


__all__ = [
    "COMMENT_INTELLIGENCE_VERSION",
    "COMMENT_TAXONOMY_VERSION",
    "INTENT_LABELS",
    "CommentFeatureQuality",
    "CommentIntelligenceConfig",
    "CommentIntelligenceSnapshot",
    "CommentMissingness",
    "CommentSignalFeatures",
    "CommentTransferPriorEntry",
    "CommentTransferPriors",
    "build_transfer_priors_from_snapshots",
    "extract_comment_intelligence_for_video",
    "extract_comment_intelligence_snapshots",
]
