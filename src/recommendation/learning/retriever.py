from __future__ import annotations

import json
import hashlib
import logging
import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .temporal import parse_dt, row_as_of_time, row_lexical_text, row_text
from .user_affinity import build_user_affinity_context

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BM25Okapi = None

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


logger = logging.getLogger(__name__)

RUNTIME_OBJECTIVES = ("reach", "engagement", "conversion")
CREATOR_RETRIEVAL_VERSION = "creator_retrieval.v1"
CREATOR_RETRIEVAL_MAX_BLEND_WEIGHT = 0.16
CREATOR_RETRIEVAL_MIN_QUERY_GUARD = 0.20
CREATOR_RETRIEVAL_MAX_MEMORY = 24


@dataclass
class HybridRetrieverTrainerConfig:
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_weight: float = 0.45
    dense_weight: float = 0.25
    multimodal_weight: float = 0.20
    graph_weight: float = 0.10
    trajectory_weight: float = 0.0


def _to_tokens(text: str) -> List[str]:
    return [
        token
        for token in text.lower().replace("#", " ").split()
        if len(token.strip()) >= 2
    ]


def _to_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _normalize_scores(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return np.array([], dtype=np.float32)
    min_v = float(np.min(array))
    max_v = float(np.max(array))
    if math.isclose(min_v, max_v):
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_v) / (max_v - min_v)


def _normalize_weight_map(weights: Dict[str, float]) -> Dict[str, float]:
    normalized = {
        "lexical": max(0.0, float(weights.get("lexical", 0.0))),
        "dense_text": max(0.0, float(weights.get("dense_text", 0.0))),
        "multimodal": max(0.0, float(weights.get("multimodal", 0.0))),
        "graph_dense": max(0.0, float(weights.get("graph_dense", 0.0))),
        "trajectory_dense": max(0.0, float(weights.get("trajectory_dense", 0.0))),
    }
    denom = sum(normalized.values())
    if denom <= 0:
        return {
            "lexical": 0.35,
            "dense_text": 0.30,
            "multimodal": 0.20,
            "graph_dense": 0.10,
            "trajectory_dense": 0.05,
        }
    return {key: value / denom for key, value in normalized.items()}


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ensure_2d(vectors: Sequence[Sequence[float]], dim: int) -> np.ndarray:
    out = np.zeros((len(vectors), dim), dtype=np.float32)
    for idx, raw in enumerate(vectors):
        values = [float(item) for item in raw]
        limit = min(dim, len(values))
        if limit > 0:
            out[idx, :limit] = np.asarray(values[:limit], dtype=np.float32)
    return out


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def _month_shard(value: Optional[str]) -> str:
    dt = parse_dt(value)
    if dt is None:
        return "unknown"
    return dt.astimezone(timezone.utc).strftime("%Y-%m")


def _row_multimodal_fallback(row: Dict[str, Any]) -> List[float]:
    features = row.get("features", {})
    if not isinstance(features, dict):
        features = {}
    comment = features.get("comment_intelligence", {})
    if not isinstance(comment, dict):
        comment = {}
    return [
        float(features.get("caption_word_count") or 0.0),
        float(features.get("hashtag_count") or 0.0),
        float(features.get("keyword_count") or 0.0),
        float(comment.get("confusion_index") or 0.0),
        float(comment.get("help_seeking_index") or 0.0),
        float(comment.get("sentiment_volatility") or 0.0),
    ]


def _query_multimodal_fallback(query_row: Dict[str, Any], dim: int) -> np.ndarray:
    source: List[float] = []
    fabric_output = query_row.get("_fabric_output")
    if isinstance(fabric_output, dict):
        text = fabric_output.get("text", {})
        structure = fabric_output.get("structure", {})
        audio = fabric_output.get("audio", {})
        visual = fabric_output.get("visual", {})
        if isinstance(text, dict) and isinstance(structure, dict):
            source = [
                float(text.get("clarity_score") or 0.0),
                float(text.get("token_count") or 0.0),
                float(text.get("hashtag_count") or 0.0),
                float(structure.get("hook_timing_seconds") or 0.0),
                float(structure.get("payoff_timing_seconds") or 0.0),
                float(visual.get("visual_motion_score") or 0.0)
                if isinstance(visual, dict)
                else 0.0,
                float(audio.get("speech_ratio") or 0.0) if isinstance(audio, dict) else 0.0,
            ]
    if not source:
        source = _row_multimodal_fallback(query_row)
    out = np.zeros((1, dim), dtype=np.float32)
    limit = min(dim, len(source))
    if limit > 0:
        out[0, :limit] = np.asarray(source[:limit], dtype=np.float32)
    return _l2_normalize_rows(out)[0]


def _row_trajectory_fallback(row: Dict[str, Any]) -> List[float]:
    features = row.get("features", {})
    if not isinstance(features, dict):
        features = {}
    trajectory = features.get("trajectory_features", {})
    if not isinstance(trajectory, dict):
        trajectory = {}
    regime_probs = trajectory.get("regime_probabilities", {})
    if not isinstance(regime_probs, dict):
        regime_probs = {}
    return [
        float(trajectory.get("early_velocity") or 0.0),
        float(trajectory.get("core_velocity") or 0.0),
        float(trajectory.get("late_lift") or 0.0),
        float(trajectory.get("stability") or 0.0),
        float(trajectory.get("durability_ratio") or 0.0),
        float(trajectory.get("acceleration_proxy") or 0.0),
        float(trajectory.get("curvature_proxy") or 0.0),
        float(trajectory.get("peak_lag_hours") or 0.0) / 96.0,
        float(regime_probs.get("spike") or 0.0),
        float(regime_probs.get("balanced") or 0.0),
        float(regime_probs.get("durable") or 0.0),
        float(trajectory.get("regime_confidence") or 0.0),
    ]


def _query_trajectory_fallback(
    query_row: Dict[str, Any],
    dim: int,
    payload: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    source: List[float] = []
    features = query_row.get("features")
    if isinstance(features, dict):
        trajectory = features.get("trajectory_features")
        if isinstance(trajectory, dict):
            probe = dict(features)
            probe["trajectory_features"] = trajectory
            source = _row_trajectory_fallback({"features": probe})
    if not source:
        source = _row_trajectory_fallback(query_row)
    if not source and isinstance(payload, dict):
        video_lookup = payload.get("video_lookup")
        query_id = str(query_row.get("video_id") or query_row.get("row_id") or "").split("::", 1)[0]
        if isinstance(video_lookup, dict):
            raw = video_lookup.get(query_id)
            if isinstance(raw, list):
                source = [float(item) for item in raw]
    out = np.zeros((1, dim), dtype=np.float32)
    limit = min(dim, len(source))
    if limit > 0:
        out[0, :limit] = np.asarray(source[:limit], dtype=np.float32)
    return _l2_normalize_rows(out)[0]


def _query_hashtags(query_row: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    explicit = query_row.get("hashtags")
    if isinstance(explicit, list):
        out.extend(str(item) for item in explicit if str(item).strip())
    text = row_text(query_row)
    if text:
        out.extend(token for token in text.split() if token.strip().startswith("#"))
    cleaned: List[str] = []
    seen: set[str] = set()
    for value in out:
        item = str(value).strip().lower()
        if not item:
            continue
        if not item.startswith("#"):
            item = f"#{item}"
        if item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned[:12]


def _query_style_tags(query_row: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    fabric = query_row.get("_fabric_output")
    if isinstance(fabric, dict):
        visual = fabric.get("visual")
        if isinstance(visual, dict):
            raw = visual.get("style_tags")
            if isinstance(raw, list):
                out.extend(str(item) for item in raw if str(item).strip())
    content_type = str(query_row.get("content_type") or "").strip().lower()
    if content_type:
        out.append(content_type)
    cleaned: List[str] = []
    seen: set[str] = set()
    for value in out:
        item = str(value).strip().lower().replace(" ", "_")
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned[:8]


def _query_audio_motif(query_row: Dict[str, Any]) -> Optional[str]:
    fabric = query_row.get("_fabric_output")
    audio = fabric.get("audio") if isinstance(fabric, dict) else None
    if not isinstance(audio, dict):
        return None
    speech = float(audio.get("speech_ratio") or 0.0)
    tempo = float(audio.get("tempo") or 0.0)
    energy = float(audio.get("energy") or 0.0)
    music = bool(audio.get("music_presence", False))

    def _bucket(value: float, lo: float, hi: float) -> str:
        if value < lo:
            return "low"
        if value > hi:
            return "high"
        return "mid"

    return (
        f"speech:{_bucket(speech, 0.35, 0.70)}|"
        f"tempo:{_bucket(tempo, 90.0, 140.0)}|"
        f"energy:{_bucket(energy, 0.35, 0.70)}|"
        f"music:{'yes' if music else 'no'}"
    )


def _normalize_candidate_ids(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for item in value:
        candidate_id = str(item or "").strip().lower()
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        out.append(candidate_id)
        if len(out) >= CREATOR_RETRIEVAL_MAX_MEMORY:
            break
    return out


def _candidate_memory_block(user_context: Dict[str, Any], key: str) -> Dict[str, List[str]]:
    block = user_context.get(key)
    if not isinstance(block, dict):
        return {"positive_candidate_ids": [], "negative_candidate_ids": []}
    return {
        "positive_candidate_ids": _normalize_candidate_ids(block.get("positive_candidate_ids")),
        "negative_candidate_ids": _normalize_candidate_ids(block.get("negative_candidate_ids")),
    }


def _merge_candidate_memory(
    primary: Dict[str, List[str]],
    fallback: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for key in ("positive_candidate_ids", "negative_candidate_ids"):
        merged: List[str] = []
        seen: set[str] = set()
        for source in (primary.get(key, []), fallback.get(key, [])):
            for item in source:
                if item in seen:
                    continue
                seen.add(item)
                merged.append(item)
                if len(merged) >= CREATOR_RETRIEVAL_MAX_MEMORY:
                    break
            if len(merged) >= CREATOR_RETRIEVAL_MAX_MEMORY:
                break
        out[key] = merged
    return out


def _lookup_vector(payload: Dict[str, Any], key: str, value: str) -> Optional[np.ndarray]:
    lookup = payload.get(key)
    if not isinstance(lookup, dict):
        return None
    raw = lookup.get(value)
    if not isinstance(raw, list):
        return None
    if not raw:
        return None
    return np.asarray([float(item) for item in raw], dtype=np.float32)


def _query_graph_vector(query_row: Dict[str, Any], payload: Dict[str, Any], dim: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    vectors: List[np.ndarray] = []
    query_id = str(query_row.get("row_id") or query_row.get("video_id") or "").strip()
    creator_id = str(query_row.get("author_id") or "").strip()
    hashtags = _query_hashtags(query_row)
    style_tags = _query_style_tags(query_row)
    motif = _query_audio_motif(query_row)
    sources: List[str] = []

    if query_id:
        vec = _lookup_vector(payload, "video_lookup", query_id)
        if vec is not None:
            vectors.append(vec)
            sources.append("video")
    if creator_id:
        vec = _lookup_vector(payload, "creator_lookup", creator_id)
        if vec is not None:
            vectors.append(vec)
            sources.append("creator")
    for tag in hashtags:
        vec = _lookup_vector(payload, "hashtag_lookup", tag)
        if vec is not None:
            vectors.append(vec)
            sources.append("hashtag")
    for style in style_tags:
        vec = _lookup_vector(payload, "style_lookup", style)
        if vec is not None:
            vectors.append(vec)
            sources.append("style")
    if motif:
        vec = _lookup_vector(payload, "audio_lookup", motif)
        if vec is not None:
            vectors.append(vec)
            sources.append("audio_motif")

    trace = {
        "query_id": query_id,
        "creator_id": creator_id,
        "hashtags": hashtags[:6],
        "style_tags": style_tags[:6],
        "audio_motif": motif,
        "sources": sources,
    }
    if not vectors:
        return np.zeros((max(1, dim),), dtype=np.float32), trace
    matrix = np.vstack(vectors).astype(np.float32)
    out = np.mean(matrix, axis=0)
    norm = float(np.linalg.norm(out))
    if norm > 0.0:
        out = out / norm
    return out.astype(np.float32), trace


def _payload_embedding_dim(payload: Dict[str, Any]) -> int:
    embeddings = payload.get("embeddings")
    if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
        return int(embeddings.shape[1])
    return int(_to_float(payload.get("dimension"), 0.0))


class HybridRetriever:
    def __init__(
        self,
        row_ids: List[str],
        row_texts: List[str],
        row_times: Dict[str, str],
        row_metadata: Dict[str, Dict[str, Any]],
        row_shards: Dict[str, str],
        sparse_backend: str,
        sparse_payload: Dict[str, Any],
        dense_backend: str,
        dense_payload: Dict[str, Any],
        multimodal_backend: str,
        multimodal_payload: Dict[str, Any],
        graph_backend: str,
        graph_payload: Dict[str, Any],
        trajectory_backend: str,
        trajectory_payload: Dict[str, Any],
        objective_blend: Dict[str, Dict[str, float]],
        config: HybridRetrieverTrainerConfig,
        artifact_version: str = "retriever.v2.0",
    ) -> None:
        self.row_ids = row_ids
        self.row_texts = row_texts
        self.row_times = row_times
        self.row_metadata = row_metadata
        self.row_shards = row_shards
        self.sparse_backend = sparse_backend
        self.sparse_payload = sparse_payload
        self.dense_backend = dense_backend
        self.dense_payload = dense_payload
        self.multimodal_backend = multimodal_backend
        self.multimodal_payload = multimodal_payload
        self.graph_backend = graph_backend
        self.graph_payload = graph_payload
        self.trajectory_backend = trajectory_backend
        self.trajectory_payload = trajectory_payload
        self.objective_blend = {
            objective: _normalize_weight_map(weights)
            for objective, weights in objective_blend.items()
        }
        self.config = config
        self.artifact_version = artifact_version
        self._dense_encoder: Any = None
        self._dense_faiss: Any = None
        self._multimodal_faiss: Any = None
        self._graph_faiss: Any = None
        self._trajectory_faiss: Any = None
        self.graph_bundle_id = str(self.graph_payload.get("graph_bundle_id") or "")
        self.graph_version = str(self.graph_payload.get("graph_version") or "")
        self.trajectory_manifest_id = str(
            self.trajectory_payload.get("trajectory_manifest_id") or ""
        )
        self.trajectory_version = str(
            self.trajectory_payload.get("trajectory_version") or ""
        )
        self.row_candidate_keys: Dict[str, str] = {
            row_id: str((self.row_metadata.get(row_id, {}) or {}).get("video_id") or row_id.split("::", 1)[0])
            for row_id in self.row_ids
        }
        self.row_index_by_id: Dict[str, int] = {
            row_id: idx for idx, row_id in enumerate(self.row_ids)
        }
        self.candidate_latest_row_ids: Dict[str, str] = {}
        for row_id in self.row_ids:
            candidate_key = self.row_candidate_keys.get(row_id, row_id.split("::", 1)[0])
            normalized_candidate_key = str(candidate_key).strip().lower()
            current = self.candidate_latest_row_ids.get(normalized_candidate_key)
            if current is None:
                self.candidate_latest_row_ids[normalized_candidate_key] = row_id
                continue
            current_dt = parse_dt(self.row_times.get(current))
            row_dt = parse_dt(self.row_times.get(row_id))
            if row_dt is not None and (current_dt is None or row_dt > current_dt):
                self.candidate_latest_row_ids[normalized_candidate_key] = row_id

    @classmethod
    def train(
        cls,
        rows: Iterable[Dict[str, Any]],
        config: Optional[HybridRetrieverTrainerConfig] = None,
        multimodal_vectors: Optional[Dict[str, Sequence[float]]] = None,
        graph_vectors: Optional[Dict[str, Sequence[float]]] = None,
        graph_lookup: Optional[Dict[str, Dict[str, Sequence[float]]]] = None,
        graph_metadata: Optional[Dict[str, Any]] = None,
        trajectory_vectors: Optional[Dict[str, Sequence[float]]] = None,
        trajectory_lookup: Optional[Dict[str, Dict[str, Sequence[float]]]] = None,
        trajectory_metadata: Optional[Dict[str, Any]] = None,
        objective_blend: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> "HybridRetriever":
        cfg = config or HybridRetrieverTrainerConfig()
        ordered_rows = list(rows)
        row_ids = [str(row.get("row_id")) for row in ordered_rows]
        row_texts = [row_text(row) for row in ordered_rows]
        row_lexical_texts = [row_lexical_text(row) for row in ordered_rows]
        row_times = {
            str(row.get("row_id")): (
                row_as_of_time(row).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
                if row_as_of_time(row)
                else ""
            )
            for row in ordered_rows
        }
        row_metadata: Dict[str, Dict[str, Any]] = {}
        row_shards: Dict[str, str] = {}
        for idx, (row, row_id, text_value) in enumerate(zip(ordered_rows, row_ids, row_texts)):
            features = row.get("features", {})
            if not isinstance(features, dict):
                features = {}
            language = (
                row.get("language")
                if isinstance(row.get("language"), str)
                else features.get("language")
                if isinstance(features.get("language"), str)
                else None
            )
            content_type = (
                row.get("content_type")
                if isinstance(row.get("content_type"), str)
                else "other"
            )
            row_metadata[row_id] = {
                "row_id": row_id,
                "video_id": str(row.get("video_id") or row_id.split("::", 1)[0]),
                "candidate_id": str(row.get("video_id") or row_id.split("::", 1)[0]),
                "text": text_value,
                "semantic_text": text_value,
                "lexical_text": row_lexical_texts[idx],
                "caption": str(row.get("caption") or ""),
                "topic_key": str(row.get("topic_key") or "general"),
                "author_id": str(row.get("author_id") or "unknown"),
                "as_of_time": row_times.get(row_id, ""),
                "posted_at": (
                    row.get("posted_at").astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
                    if isinstance(row.get("posted_at"), datetime)
                    else str(row.get("posted_at") or "")
                ),
                "language": language,
                "locale": row.get("locale") if isinstance(row.get("locale"), str) else None,
                "content_type": content_type,
                "hashtags": list(row.get("hashtags") or _query_hashtags(row)),
                "keywords": list(row.get("keywords") or []),
                "search_query": str(row.get("search_query") or "").strip() or None,
                "style_tags": _query_style_tags(row),
                "audio_motif": _query_audio_motif(row),
            }
            row_shards[row_id] = _month_shard(row_times.get(row_id, ""))

        sparse_backend = "tfidf"
        sparse_payload: Dict[str, Any]
        if BM25Okapi is not None:
            tokenized = [_to_tokens(text) for text in row_lexical_texts]
            sparse_backend = "bm25"
            sparse_payload = {"model": BM25Okapi(tokenized), "tokenized": tokenized}
        else:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                token_pattern=r"\b\w+\b",
            )
            matrix = vectorizer.fit_transform(row_lexical_texts)
            sparse_payload = {"vectorizer": vectorizer, "matrix": matrix}

        dense_backend = "tfidf-char"
        dense_payload: Dict[str, Any]
        _sbert_encoder = None  # stored for reuse in multimodal branch
        if SentenceTransformer is not None:
            try:
                _sbert_encoder = SentenceTransformer(cfg.dense_model_name)
                embeddings = _sbert_encoder.encode(
                    row_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype(np.float32)
                dense_backend = (
                    "sentence-transformers-faiss"
                    if faiss is not None
                    else "sentence-transformers"
                )
                dense_payload = {
                    "model_name": cfg.dense_model_name,
                    "embeddings": embeddings,
                }
                logger.info(
                    "SentenceTransformer loaded (%s), dense embeddings shape=%s",
                    cfg.dense_model_name, embeddings.shape,
                )
            except Exception as exc:
                logger.warning(
                    "SentenceTransformer failed (%s), falling back to char-TF-IDF: %s",
                    cfg.dense_model_name, exc,
                )
                dense_vectorizer = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    lowercase=True,
                )
                dense_matrix = dense_vectorizer.fit_transform(row_lexical_texts)
                dense_payload = {"vectorizer": dense_vectorizer, "matrix": dense_matrix}
        else:
            logger.warning("sentence-transformers not installed, using char-TF-IDF for dense branch")
            dense_vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                lowercase=True,
            )
            dense_matrix = dense_vectorizer.fit_transform(row_lexical_texts)
            dense_payload = {"vectorizer": dense_vectorizer, "matrix": dense_matrix}

        # --- Multimodal branch ---
        # When SBERT is available and no precomputed vectors, use caption-only
        # embeddings (different signal from dense_text which uses full row text).
        mm_map = multimodal_vectors or {}
        if not mm_map and _sbert_encoder is not None:
            caption_texts = [str(row.get("caption") or "") for row in ordered_rows]
            mm_embeddings = _sbert_encoder.encode(
                caption_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            multimodal_matrix = mm_embeddings
            multimodal_backend = (
                "sbert-caption-faiss" if faiss is not None else "sbert-caption"
            )
            multimodal_payload: Dict[str, Any] = {
                "embeddings": multimodal_matrix,
                "dimension": mm_embeddings.shape[1],
                "source": "sbert_caption_auto",
            }
            logger.info(
                "Multimodal branch: SBERT caption embeddings, shape=%s",
                mm_embeddings.shape,
            )
        else:
            default_vectors: List[Sequence[float]] = []
            vector_dim = 0
            for row, row_id in zip(ordered_rows, row_ids):
                raw = mm_map.get(row_id)
                if raw is None:
                    raw = mm_map.get(str(row.get("video_id") or ""))
                values = list(raw) if raw is not None else _row_multimodal_fallback(row)
                vector_dim = max(vector_dim, len(values))
                default_vectors.append(values)
            vector_dim = max(4, vector_dim)
            multimodal_matrix = _l2_normalize_rows(_ensure_2d(default_vectors, vector_dim))
            multimodal_backend = "fabric-precomputed-faiss" if faiss is not None else "fabric-precomputed"
            multimodal_payload = {
                "embeddings": multimodal_matrix,
                "dimension": vector_dim,
                "source": "feature_snapshot" if multimodal_vectors else "row_fallback",
            }

        graph_vectors_map = graph_vectors or {}
        graph_rows: List[Sequence[float]] = []
        graph_dim = 0
        for row, row_id in zip(ordered_rows, row_ids):
            raw = graph_vectors_map.get(row_id)
            if raw is None:
                raw = graph_vectors_map.get(str(row.get("video_id") or ""))
            values = list(raw) if raw is not None else []
            graph_dim = max(graph_dim, len(values))
            graph_rows.append(values)
        graph_dim = max(4, graph_dim)
        graph_matrix = _l2_normalize_rows(_ensure_2d(graph_rows, graph_dim))
        graph_payload = {
            "embeddings": graph_matrix,
            "dimension": graph_dim,
            "source": "creator_video_dna_graph" if graph_vectors else "disabled",
            "video_lookup": {
                str(key): [float(item) for item in value]
                for key, value in (graph_lookup or {}).get("video", {}).items()
                if isinstance(value, (list, tuple))
            },
            "creator_lookup": {
                str(key): [float(item) for item in value]
                for key, value in (graph_lookup or {}).get("creator", {}).items()
                if isinstance(value, (list, tuple))
            },
            "hashtag_lookup": {
                str(key): [float(item) for item in value]
                for key, value in (graph_lookup or {}).get("hashtag", {}).items()
                if isinstance(value, (list, tuple))
            },
            "audio_lookup": {
                str(key): [float(item) for item in value]
                for key, value in (graph_lookup or {}).get("audio_motif", {}).items()
                if isinstance(value, (list, tuple))
            },
            "style_lookup": {
                str(key): [float(item) for item in value]
                for key, value in (graph_lookup or {}).get("style_signature", {}).items()
                if isinstance(value, (list, tuple))
            },
            "graph_bundle_id": str((graph_metadata or {}).get("graph_bundle_id") or ""),
            "graph_version": str((graph_metadata or {}).get("graph_version") or ""),
            "graph_schema_hash": str((graph_metadata or {}).get("graph_schema_hash") or ""),
        }
        graph_backend = "graph-dna-faiss" if faiss is not None and graph_vectors else "graph-dna"

        trajectory_vectors_map = trajectory_vectors or {}
        trajectory_rows: List[Sequence[float]] = []
        trajectory_dim = 0
        for row, row_id in zip(ordered_rows, row_ids):
            raw = trajectory_vectors_map.get(row_id)
            if raw is None:
                raw = trajectory_vectors_map.get(str(row.get("video_id") or ""))
            values = list(raw) if raw is not None else _row_trajectory_fallback(row)
            trajectory_dim = max(trajectory_dim, len(values))
            trajectory_rows.append(values)
        trajectory_dim = max(4, trajectory_dim)
        trajectory_matrix = _l2_normalize_rows(
            _ensure_2d(trajectory_rows, trajectory_dim)
        )
        trajectory_payload = {
            "embeddings": trajectory_matrix,
            "dimension": trajectory_dim,
            "source": "trajectory_artifact" if trajectory_vectors else "row_fallback",
            "video_lookup": {
                str(key): [float(item) for item in value]
                for key, value in (trajectory_lookup or {}).get("video", {}).items()
                if isinstance(value, (list, tuple))
            },
            "trajectory_manifest_id": str(
                (trajectory_metadata or {}).get("trajectory_manifest_id") or ""
            ),
            "trajectory_version": str(
                (trajectory_metadata or {}).get("trajectory_version") or ""
            ),
            "trajectory_schema_hash": str(
                (trajectory_metadata or {}).get("trajectory_schema_hash") or ""
            ),
        }
        trajectory_backend = (
            "trajectory-precomputed-faiss"
            if faiss is not None
            else "trajectory-precomputed"
        )

        default_blend = _normalize_weight_map(
            {
                "lexical": cfg.sparse_weight,
                "dense_text": cfg.dense_weight,
                "multimodal": cfg.multimodal_weight,
                "graph_dense": cfg.graph_weight,
                "trajectory_dense": cfg.trajectory_weight,
            }
        )
        objective_weights = {
            objective: _normalize_weight_map(default_blend)
            for objective in RUNTIME_OBJECTIVES
        }
        if objective_blend:
            for objective, weights in objective_blend.items():
                if objective in RUNTIME_OBJECTIVES:
                    objective_weights[objective] = _normalize_weight_map(weights)

        return cls(
            row_ids=row_ids,
            row_texts=row_texts,
            row_times=row_times,
            row_metadata=row_metadata,
            row_shards=row_shards,
            sparse_backend=sparse_backend,
            sparse_payload=sparse_payload,
            dense_backend=dense_backend,
            dense_payload=dense_payload,
            multimodal_backend=multimodal_backend,
            multimodal_payload=multimodal_payload,
            graph_backend=graph_backend,
            graph_payload=graph_payload,
            trajectory_backend=trajectory_backend,
            trajectory_payload=trajectory_payload,
            objective_blend=objective_weights,
            config=cfg,
            artifact_version="retriever.v2.0",
        )

    def _weights_for_objective(
        self,
        objective: Optional[str],
        override: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        if override is not None:
            return _normalize_weight_map(override)
        effective = (objective or "engagement").strip().lower()
        if effective not in self.objective_blend:
            effective = "engagement"
        return _normalize_weight_map(self.objective_blend.get(effective, {}))

    def set_objective_blend(self, objective_blend: Dict[str, Dict[str, float]]) -> None:
        for objective, weights in objective_blend.items():
            if objective in RUNTIME_OBJECTIVES:
                self.objective_blend[objective] = _normalize_weight_map(weights)

    def branch_weights(self, objective: str) -> Dict[str, float]:
        return dict(self._weights_for_objective(objective))

    def row_payload(self, row_id: str) -> Dict[str, Any]:
        return dict(self.row_metadata.get(row_id, {"row_id": row_id, "video_id": row_id}))

    def _branch_embeddings(self, branch: str) -> Optional[np.ndarray]:
        if branch == "dense_text":
            raw = self.dense_payload.get("embeddings")
        elif branch == "multimodal":
            raw = self.multimodal_payload.get("embeddings")
        elif branch == "graph_dense":
            raw = self.graph_payload.get("embeddings")
        elif branch == "trajectory_dense":
            raw = self.trajectory_payload.get("embeddings")
        else:
            raw = None
        if isinstance(raw, np.ndarray) and raw.ndim == 2:
            return np.asarray(raw, dtype=np.float32)
        return None

    def _candidate_centroid(
        self,
        *,
        candidate_ids: Sequence[str],
        branch: str,
    ) -> Tuple[Optional[np.ndarray], int]:
        embeddings = self._branch_embeddings(branch)
        if embeddings is None or embeddings.shape[0] != len(self.row_ids):
            return None, 0
        row_indexes: List[int] = []
        seen_rows: set[str] = set()
        for candidate_id in candidate_ids:
            lookup_id = str(candidate_id or "").strip().lower()
            if not lookup_id:
                continue
            row_id = self.candidate_latest_row_ids.get(lookup_id)
            if row_id is None and lookup_id in self.row_index_by_id:
                row_id = lookup_id
            if row_id is None or row_id in seen_rows:
                continue
            row_idx = self.row_index_by_id.get(row_id)
            if row_idx is None:
                continue
            seen_rows.add(row_id)
            row_indexes.append(int(row_idx))
            if len(row_indexes) >= CREATOR_RETRIEVAL_MAX_MEMORY:
                break
        if not row_indexes:
            return None, 0
        centroid = np.mean(embeddings[row_indexes], axis=0).astype(np.float32)
        norm = float(np.linalg.norm(centroid))
        if norm <= 0.0:
            return None, 0
        centroid = centroid / norm
        return centroid.astype(np.float32), len(row_indexes)

    def _creator_branch_scores(
        self,
        *,
        branch: str,
        positive_candidate_ids: Sequence[str],
        negative_candidate_ids: Sequence[str],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        embeddings = self._branch_embeddings(branch)
        if embeddings is None or embeddings.shape[0] != len(self.row_ids):
            return np.zeros(len(self.row_ids), dtype=np.float32), {
                "available": False,
                "reason": "embeddings_unavailable",
                "positive_memory_count": 0,
                "negative_memory_count": 0,
            }
        positive_centroid, positive_count = self._candidate_centroid(
            candidate_ids=positive_candidate_ids,
            branch=branch,
        )
        negative_centroid, negative_count = self._candidate_centroid(
            candidate_ids=negative_candidate_ids,
            branch=branch,
        )
        if positive_centroid is None and negative_centroid is None:
            return np.zeros(len(self.row_ids), dtype=np.float32), {
                "available": False,
                "reason": "candidate_memory_unavailable",
                "positive_memory_count": positive_count,
                "negative_memory_count": negative_count,
            }
        positive_scores = (
            np.dot(embeddings, positive_centroid).astype(np.float32)
            if positive_centroid is not None
            else np.zeros(len(self.row_ids), dtype=np.float32)
        )
        negative_scores = (
            np.dot(embeddings, negative_centroid).astype(np.float32)
            if negative_centroid is not None
            else np.zeros(len(self.row_ids), dtype=np.float32)
        )
        raw = positive_scores - (0.7 * negative_scores)
        return _normalize_scores(raw), {
            "available": True,
            "reason": None,
            "positive_memory_count": positive_count,
            "negative_memory_count": negative_count,
        }

    def _creator_retrieval_personalization(
        self,
        *,
        user_context: Optional[Dict[str, Any]],
        objective: Optional[str],
        query_as_of: Optional[datetime],
        lexical_scores: np.ndarray,
        dense_scores: np.ndarray,
        branch_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        payload = user_context if isinstance(user_context, dict) else {}
        affinity_context = build_user_affinity_context(
            user_context=payload if payload else None,
            effective_objective=(objective or "engagement").strip().lower(),
            as_of=query_as_of or datetime.now(timezone.utc),
        )
        if not bool(affinity_context.get("enabled")):
            return {
                "applied": False,
                "reason": "missing_creator_context",
                "version": CREATOR_RETRIEVAL_VERSION,
                "scores": np.zeros(len(self.row_ids), dtype=np.float32),
                "score_shifts": np.zeros(len(self.row_ids), dtype=np.float32),
                "query_guards": np.zeros(len(self.row_ids), dtype=np.float32),
                "branch_scores": {},
                "branch_meta": {},
            }
        confidence = float(affinity_context.get("confidence") or 0.0)
        if confidence <= 0.01:
            return {
                "applied": False,
                "reason": "insufficient_profile_confidence",
                "version": CREATOR_RETRIEVAL_VERSION,
                "creator_id": affinity_context.get("creator_id"),
                "confidence": confidence,
                "scores": np.zeros(len(self.row_ids), dtype=np.float32),
                "score_shifts": np.zeros(len(self.row_ids), dtype=np.float32),
                "query_guards": np.zeros(len(self.row_ids), dtype=np.float32),
                "branch_scores": {},
                "branch_meta": {},
            }

        memory = _merge_candidate_memory(
            _candidate_memory_block(payload, "objective_candidate_memory"),
            _candidate_memory_block(payload, "global_candidate_memory"),
        )
        positive_candidate_ids = memory["positive_candidate_ids"]
        negative_candidate_ids = memory["negative_candidate_ids"]
        if not positive_candidate_ids and not negative_candidate_ids:
            return {
                "applied": False,
                "reason": "missing_candidate_memory",
                "version": CREATOR_RETRIEVAL_VERSION,
                "creator_id": affinity_context.get("creator_id"),
                "confidence": confidence,
                "scores": np.zeros(len(self.row_ids), dtype=np.float32),
                "score_shifts": np.zeros(len(self.row_ids), dtype=np.float32),
                "query_guards": np.zeros(len(self.row_ids), dtype=np.float32),
                "branch_scores": {},
                "branch_meta": {},
            }

        creator_branch_scores: Dict[str, np.ndarray] = {}
        creator_branch_meta: Dict[str, Any] = {}
        available_weight = 0.0
        fused_scores = np.zeros(len(self.row_ids), dtype=np.float32)
        for branch in ("dense_text", "multimodal", "graph_dense", "trajectory_dense"):
            branch_score, branch_meta = self._creator_branch_scores(
                branch=branch,
                positive_candidate_ids=positive_candidate_ids,
                negative_candidate_ids=negative_candidate_ids,
            )
            creator_branch_meta[branch] = branch_meta
            if not branch_meta.get("available"):
                continue
            branch_weight = float(branch_weights.get(branch, 0.0))
            if branch_weight <= 0.0:
                continue
            creator_branch_scores[branch] = branch_score
            fused_scores += branch_weight * branch_score
            available_weight += branch_weight
        if available_weight <= 0.0:
            return {
                "applied": False,
                "reason": "no_personalized_branches_available",
                "version": CREATOR_RETRIEVAL_VERSION,
                "creator_id": affinity_context.get("creator_id"),
                "confidence": confidence,
                "scores": np.zeros(len(self.row_ids), dtype=np.float32),
                "score_shifts": np.zeros(len(self.row_ids), dtype=np.float32),
                "query_guards": np.zeros(len(self.row_ids), dtype=np.float32),
                "branch_scores": creator_branch_scores,
                "branch_meta": creator_branch_meta,
            }
        fused_scores = fused_scores / available_weight
        query_signal = np.asarray(
            (0.65 * dense_scores) + (0.35 * lexical_scores),
            dtype=np.float32,
        )
        query_guards = np.clip(
            (query_signal - CREATOR_RETRIEVAL_MIN_QUERY_GUARD)
            / max(1e-6, 1.0 - CREATOR_RETRIEVAL_MIN_QUERY_GUARD),
            0.0,
            1.0,
        ).astype(np.float32)
        blend_weight = CREATOR_RETRIEVAL_MAX_BLEND_WEIGHT * confidence
        score_shifts = (
            blend_weight * query_guards * ((fused_scores - 0.5) * 2.0)
        ).astype(np.float32)
        return {
            "applied": True,
            "reason": None,
            "version": CREATOR_RETRIEVAL_VERSION,
            "creator_id": affinity_context.get("creator_id"),
            "confidence": confidence,
            "blend_weight": blend_weight,
            "scores": fused_scores,
            "score_shifts": score_shifts,
            "query_guards": query_guards,
            "branch_scores": creator_branch_scores,
            "branch_meta": creator_branch_meta,
            "positive_memory_count": len(positive_candidate_ids),
            "negative_memory_count": len(negative_candidate_ids),
        }

    def _sparse_scores(self, query_text: str) -> np.ndarray:
        if self.sparse_backend == "bm25":
            model = self.sparse_payload["model"]
            query_tokens = _to_tokens(query_text)
            return np.asarray(model.get_scores(query_tokens), dtype=np.float32)
        vectorizer = self.sparse_payload["vectorizer"]
        matrix = self.sparse_payload["matrix"]
        q = vectorizer.transform([query_text])
        return cosine_similarity(q, matrix)[0].astype(np.float32)

    def _dense_scores(self, query_text: str) -> np.ndarray:
        if self.dense_backend.startswith("sentence-transformers"):
            if SentenceTransformer is None:  # pragma: no cover
                return np.zeros(len(self.row_ids), dtype=np.float32)
            model_name = self.dense_payload["model_name"]
            try:
                if self._dense_encoder is None:
                    self._dense_encoder = SentenceTransformer(model_name)
                query_embedding = self._dense_encoder.encode(
                    [query_text],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )[0].astype(np.float32)
            except Exception:  # pragma: no cover
                return np.zeros(len(self.row_ids), dtype=np.float32)
            embeddings = self.dense_payload["embeddings"]
            if faiss is not None and self.dense_backend.endswith("faiss"):
                if self._dense_faiss is None:
                    self._dense_faiss = faiss.IndexFlatIP(int(embeddings.shape[1]))
                    self._dense_faiss.add(np.asarray(embeddings, dtype=np.float32))
                scores, indices = self._dense_faiss.search(
                    np.expand_dims(query_embedding, axis=0), len(self.row_ids)
                )
                out = np.zeros(len(self.row_ids), dtype=np.float32)
                for idx, score in zip(indices[0], scores[0]):
                    if idx >= 0:
                        out[int(idx)] = float(score)
                return out
            return np.dot(embeddings, query_embedding).astype(np.float32)

        vectorizer = self.dense_payload["vectorizer"]
        matrix = self.dense_payload["matrix"]
        q = vectorizer.transform([query_text])
        return cosine_similarity(q, matrix)[0].astype(np.float32)

    def _multimodal_scores(self, query_row: Dict[str, Any]) -> np.ndarray:
        embeddings = np.asarray(
            self.multimodal_payload.get("embeddings", np.zeros((len(self.row_ids), 4))),
            dtype=np.float32,
        )
        if embeddings.shape[0] != len(self.row_ids):
            return np.zeros(len(self.row_ids), dtype=np.float32)
        dim = int(self.multimodal_payload.get("dimension", embeddings.shape[1] if embeddings.ndim == 2 else 4))
        query_vec = _query_multimodal_fallback(query_row, max(1, dim))
        if faiss is not None and self.multimodal_backend.endswith("faiss"):
            if self._multimodal_faiss is None:
                self._multimodal_faiss = faiss.IndexFlatIP(int(embeddings.shape[1]))
                self._multimodal_faiss.add(np.asarray(embeddings, dtype=np.float32))
            scores, indices = self._multimodal_faiss.search(
                np.expand_dims(query_vec, axis=0),
                len(self.row_ids),
            )
            out = np.zeros(len(self.row_ids), dtype=np.float32)
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0:
                    out[int(idx)] = float(score)
            return out
        return np.dot(embeddings, query_vec).astype(np.float32)

    def _graph_scores(self, query_row: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        embeddings = np.asarray(
            self.graph_payload.get("embeddings", np.zeros((len(self.row_ids), 4))),
            dtype=np.float32,
        )
        if embeddings.shape[0] != len(self.row_ids):
            return np.zeros(len(self.row_ids), dtype=np.float32), {"sources": []}
        dim = int(
            self.graph_payload.get(
                "dimension",
                embeddings.shape[1] if embeddings.ndim == 2 else 4,
            )
        )
        query_vec, trace = _query_graph_vector(query_row, self.graph_payload, max(1, dim))
        if faiss is not None and self.graph_backend.endswith("faiss"):
            if self._graph_faiss is None:
                self._graph_faiss = faiss.IndexFlatIP(int(embeddings.shape[1]))
                self._graph_faiss.add(np.asarray(embeddings, dtype=np.float32))
            scores, indices = self._graph_faiss.search(
                np.expand_dims(query_vec, axis=0),
                len(self.row_ids),
            )
            out = np.zeros(len(self.row_ids), dtype=np.float32)
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0:
                    out[int(idx)] = float(score)
            return out, trace
        return np.dot(embeddings, query_vec).astype(np.float32), trace

    def _trajectory_scores(self, query_row: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        embeddings = np.asarray(
            self.trajectory_payload.get("embeddings", np.zeros((len(self.row_ids), 4))),
            dtype=np.float32,
        )
        if embeddings.shape[0] != len(self.row_ids):
            return np.zeros(len(self.row_ids), dtype=np.float32), {"source": "unavailable"}
        dim = int(
            self.trajectory_payload.get(
                "dimension",
                embeddings.shape[1] if embeddings.ndim == 2 else 4,
            )
        )
        query_vec = _query_trajectory_fallback(
            query_row,
            max(1, dim),
            payload=self.trajectory_payload,
        )
        if faiss is not None and self.trajectory_backend.endswith("faiss"):
            if self._trajectory_faiss is None:
                self._trajectory_faiss = faiss.IndexFlatIP(int(embeddings.shape[1]))
                self._trajectory_faiss.add(np.asarray(embeddings, dtype=np.float32))
            scores, indices = self._trajectory_faiss.search(
                np.expand_dims(query_vec, axis=0),
                len(self.row_ids),
            )
            out = np.zeros(len(self.row_ids), dtype=np.float32)
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0:
                    out[int(idx)] = float(score)
            return out, {
                "source": "trajectory_features",
                "trajectory_manifest_id": self.trajectory_manifest_id,
                "trajectory_version": self.trajectory_version,
            }
        return np.dot(embeddings, query_vec).astype(np.float32), {
            "source": "trajectory_features",
            "trajectory_manifest_id": self.trajectory_manifest_id,
            "trajectory_version": self.trajectory_version,
        }

    def _passes_constraints(
        self,
        row_id: str,
        constraints: Dict[str, Optional[str]],
        tier: int,
    ) -> bool:
        metadata = self.row_metadata.get(row_id, {})
        language = str(metadata.get("language") or "").strip().lower()
        locale = str(metadata.get("locale") or "").strip().lower()
        content_type = str(metadata.get("content_type") or "").strip().lower()
        wanted_language = str(constraints.get("language") or "").strip().lower()
        wanted_locale = str(constraints.get("locale") or "").strip().lower()
        wanted_content_type = str(constraints.get("content_type") or "").strip().lower()

        if tier <= 2 and wanted_language:
            if language and language != wanted_language:
                return False
            if not language:
                return False
        if tier == 0 and wanted_locale:
            if locale and locale != wanted_locale:
                return False
            if not locale:
                return False
        if tier <= 1 and wanted_content_type:
            if content_type and content_type != wanted_content_type:
                return False
            if not content_type:
                return False
        return True

    def _apply_constraint_tiers(
        self,
        temporal_candidates: List[str],
        constraints: Dict[str, Optional[str]],
        min_required: int,
    ) -> Tuple[List[str], int]:
        for tier in range(0, 4):
            filtered = [
                row_id
                for row_id in temporal_candidates
                if self._passes_constraints(row_id=row_id, constraints=constraints, tier=tier)
            ]
            if len(filtered) >= min_required or tier == 3:
                return filtered, tier
        return temporal_candidates, 3

    def retrieve(
        self,
        query_row: Dict[str, Any],
        candidate_rows: Optional[Sequence[Dict[str, Any]]] = None,
        top_k: int = 200,
        index_cutoff_time: Optional[Any] = None,
        objective: Optional[str] = None,
        candidate_ids: Optional[Sequence[str]] = None,
        retrieval_constraints: Optional[Dict[str, Any]] = None,
        weight_override: Optional[Dict[str, float]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        return_metadata: bool = False,
    ) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        query_text = row_text(query_row)
        query_lexical_text = row_lexical_text(query_row)
        if not query_text.strip():
            query_text = str(query_row.get("topic_key") or "general")
        if not query_lexical_text.strip():
            query_lexical_text = query_text.lower()

        query_as_of = row_as_of_time(query_row)
        index_cutoff = (
            row_as_of_time({"as_of_time": index_cutoff_time})
            if index_cutoff_time
            else query_as_of
        )
        if index_cutoff is None:
            index_cutoff = query_as_of

        lexical_scores = _normalize_scores(self._sparse_scores(query_lexical_text))
        dense_scores = _normalize_scores(self._dense_scores(query_text))
        multimodal_scores = _normalize_scores(self._multimodal_scores(query_row))
        graph_scores_raw, graph_trace = self._graph_scores(query_row)
        graph_scores = _normalize_scores(graph_scores_raw)
        trajectory_scores_raw, trajectory_trace = self._trajectory_scores(query_row)
        trajectory_scores = _normalize_scores(trajectory_scores_raw)
        weights = self._weights_for_objective(objective=objective, override=weight_override)
        fused = (
            (weights["lexical"] * lexical_scores)
            + (weights["dense_text"] * dense_scores)
            + (weights["multimodal"] * multimodal_scores)
            + (weights["graph_dense"] * graph_scores)
            + (weights["trajectory_dense"] * trajectory_scores)
        )
        base_fused = np.asarray(fused, dtype=np.float32)
        creator_retrieval = self._creator_retrieval_personalization(
            user_context=user_context,
            objective=objective,
            query_as_of=query_as_of,
            lexical_scores=lexical_scores,
            dense_scores=dense_scores,
            branch_weights=weights,
        )
        if bool(creator_retrieval.get("applied")):
            fused = np.clip(
                fused + np.asarray(creator_retrieval.get("score_shifts"), dtype=np.float32),
                0.0,
                1.0,
            ).astype(np.float32)

        requested_ids = {
            str(item).strip()
            for item in (candidate_ids or [])
            if isinstance(item, str) and item.strip()
        }
        if not requested_ids and candidate_rows:
            for row in candidate_rows:
                if not isinstance(row, dict):
                    continue
                for key in ("candidate_id", "video_id", "row_id"):
                    value = row.get(key)
                    if isinstance(value, str) and value.strip():
                        requested_ids.add(value.strip())

        max_age_days = int(
            _to_float((retrieval_constraints or {}).get("max_age_days"), 0.0)
        )
        max_age_seconds = max_age_days * 86400 if max_age_days > 0 else None
        cutoff_shard = _month_shard(
            index_cutoff.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            if index_cutoff
            else ""
        )

        temporal_candidates: List[str] = []
        shards_considered: set[str] = set()
        for idx, row_id in enumerate(self.row_ids):
            if idx >= len(fused):
                continue
            candidate_key = self.row_candidate_keys.get(row_id, row_id.split("::", 1)[0])
            if requested_ids and row_id not in requested_ids and candidate_key not in requested_ids:
                continue
            candidate_dt = parse_dt(self.row_times.get(row_id))
            if query_as_of and candidate_dt and candidate_dt >= query_as_of:
                continue
            if index_cutoff and candidate_dt and candidate_dt > index_cutoff:
                continue
            if (
                max_age_seconds is not None
                and query_as_of is not None
                and candidate_dt is not None
                and (query_as_of - candidate_dt).total_seconds() > max_age_seconds
            ):
                continue
            shard = self.row_shards.get(row_id, "unknown")
            if cutoff_shard != "unknown" and shard != "unknown" and shard > cutoff_shard:
                continue
            temporal_candidates.append(row_id)
            shards_considered.add(shard)

        constraints = {
            "language": str((retrieval_constraints or {}).get("language") or "").strip() or None,
            "locale": str((retrieval_constraints or {}).get("locale") or "").strip() or None,
            "content_type": str((retrieval_constraints or {}).get("content_type") or "").strip() or None,
        }
        min_required = max(1, min(top_k, max(1, len(temporal_candidates)), 20))
        constrained_ids, tier_used = self._apply_constraint_tiers(
            temporal_candidates=temporal_candidates,
            constraints=constraints,
            min_required=min_required,
        )

        idx_by_row_id = {row_id: idx for idx, row_id in enumerate(self.row_ids)}
        latest_by_candidate: Dict[str, Tuple[str, Optional[datetime], float]] = {}
        for row_id in constrained_ids:
            idx = idx_by_row_id.get(row_id)
            if idx is None:
                continue
            candidate_key = self.row_candidate_keys.get(row_id, row_id.split("::", 1)[0])
            candidate_dt = parse_dt(self.row_times.get(row_id))
            fused_score = float(fused[idx]) if idx < len(fused) else -1e9
            current = latest_by_candidate.get(candidate_key)
            if current is None:
                latest_by_candidate[candidate_key] = (row_id, candidate_dt, fused_score)
                continue
            _, current_dt, current_score = current
            should_replace = False
            if candidate_dt is not None and current_dt is None:
                should_replace = True
            elif candidate_dt is not None and current_dt is not None and candidate_dt > current_dt:
                should_replace = True
            elif candidate_dt == current_dt and fused_score > current_score:
                should_replace = True
            if should_replace:
                latest_by_candidate[candidate_key] = (row_id, candidate_dt, fused_score)

        output_candidates = sorted(
            latest_by_candidate.items(),
            key=lambda item: item[1][2],
            reverse=True,
        )[: max(1, top_k)]
        items: List[Dict[str, Any]] = []
        for candidate_key, (row_id, _, _) in output_candidates:
            idx = idx_by_row_id.get(row_id)
            if idx is None:
                continue
            lexical = float(lexical_scores[idx]) if idx < len(lexical_scores) else 0.0
            dense = float(dense_scores[idx]) if idx < len(dense_scores) else 0.0
            multimodal = (
                float(multimodal_scores[idx]) if idx < len(multimodal_scores) else 0.0
            )
            graph_dense = float(graph_scores[idx]) if idx < len(graph_scores) else 0.0
            trajectory_dense = (
                float(trajectory_scores[idx]) if idx < len(trajectory_scores) else 0.0
            )
            fused_score = float(fused[idx]) if idx < len(fused) else 0.0
            creator_scores = creator_retrieval.get("scores")
            creator_query_guards = creator_retrieval.get("query_guards")
            creator_score_shifts = creator_retrieval.get("score_shifts")
            creator_score = (
                float(creator_scores[idx])
                if isinstance(creator_scores, np.ndarray) and idx < len(creator_scores)
                else 0.0
            )
            creator_query_guard = (
                float(creator_query_guards[idx])
                if isinstance(creator_query_guards, np.ndarray) and idx < len(creator_query_guards)
                else 0.0
            )
            creator_shift = (
                float(creator_score_shifts[idx])
                if isinstance(creator_score_shifts, np.ndarray) and idx < len(creator_score_shifts)
                else 0.0
            )
            candidate_meta = self.row_metadata.get(row_id, {})
            hashtag_overlap = 0
            style_overlap = 0
            query_hashtags = set(graph_trace.get("hashtags") or [])
            query_style = set(graph_trace.get("style_tags") or [])
            candidate_hashtags = set(candidate_meta.get("hashtags") or [])
            candidate_style = set(candidate_meta.get("style_tags") or [])
            if query_hashtags and candidate_hashtags:
                hashtag_overlap = len(query_hashtags & candidate_hashtags)
            if query_style and candidate_style:
                style_overlap = len(query_style & candidate_style)
            items.append(
                {
                    "candidate_id": candidate_key,
                    "candidate_row_id": row_id,
                    "feature_row_id": row_id,
                    "sparse_score": lexical,
                    "dense_score": dense,
                    "fused_score": fused_score,
                    "lexical_score": lexical,
                    "dense_text_score": dense,
                    "multimodal_score": multimodal,
                    "graph_dense_score": graph_dense,
                    "trajectory_dense_score": trajectory_dense,
                    "retrieval_branch_scores": {
                        "lexical": lexical,
                        "dense_text": dense,
                        "multimodal": multimodal,
                        "graph_dense": graph_dense,
                        "trajectory_dense": trajectory_dense,
                        "creator_affinity": creator_score,
                        "fused_base": float(base_fused[idx]) if idx < len(base_fused) else fused_score,
                        "fused": fused_score,
                    },
                    "creator_retrieval_score": creator_score,
                    "creator_retrieval_trace": {
                        "version": CREATOR_RETRIEVAL_VERSION,
                        "applied": bool(creator_retrieval.get("applied")),
                        "creator_id": creator_retrieval.get("creator_id"),
                        "confidence": float(creator_retrieval.get("confidence") or 0.0),
                        "query_guard": creator_query_guard,
                        "score_shift": creator_shift,
                        "branch_scores": {
                            branch: float(scores[idx])
                            for branch, scores in dict(creator_retrieval.get("branch_scores") or {}).items()
                            if isinstance(scores, np.ndarray) and idx < len(scores)
                        },
                    },
                    "graph_trace": {
                        "graph_bundle_id": self.graph_bundle_id,
                        "graph_version": self.graph_version,
                        "query_sources": list(graph_trace.get("sources") or []),
                        "candidate_overlap": {
                            "hashtag_overlap": int(hashtag_overlap),
                            "style_overlap": int(style_overlap),
                            "same_audio_motif": bool(
                                graph_trace.get("audio_motif")
                                and graph_trace.get("audio_motif")
                                == candidate_meta.get("audio_motif")
                            ),
                        },
                    },
                    "trajectory_trace": {
                        "trajectory_manifest_id": self.trajectory_manifest_id,
                        "trajectory_version": self.trajectory_version,
                        "query_source": str(trajectory_trace.get("source") or "unknown"),
                    },
                }
            )

        if not return_metadata:
            return items
        branch_coverage = {
            "lexical": sum(1 for item in items if float(item.get("lexical_score") or 0.0) > 0.0),
            "dense_text": sum(1 for item in items if float(item.get("dense_text_score") or 0.0) > 0.0),
            "multimodal": sum(1 for item in items if float(item.get("multimodal_score") or 0.0) > 0.0),
            "graph_dense": sum(1 for item in items if float(item.get("graph_dense_score") or 0.0) > 0.0),
            "trajectory_dense": sum(
                1 for item in items if float(item.get("trajectory_dense_score") or 0.0) > 0.0
            ),
            "creator_affinity": sum(
                1 for item in items if float(item.get("creator_retrieval_score") or 0.0) > 0.0
            ),
        }
        metadata = {
            "retrieval_mode": "intersected" if requested_ids else "global",
            "constraint_tier_used": tier_used,
            "objective_effective": (objective or "engagement").strip().lower(),
            "weights": weights,
            "candidate_pool_total": len(self.row_ids),
            "candidate_pool_requested": len(requested_ids) if requested_ids else len(self.row_ids),
            "candidate_pool_temporal": len(temporal_candidates),
            "candidate_pool_constrained": len(constrained_ids),
            "candidate_pool_constrained_unique_candidates": len(latest_by_candidate),
            "branch_coverage": branch_coverage,
            "temporal_shards_considered": sorted(shards_considered),
            "temporal_shard_count": len(shards_considered),
            "retriever_artifact_version": self.artifact_version,
            "graph_bundle_id": self.graph_bundle_id,
            "graph_version": self.graph_version,
            "graph_query_sources": list(graph_trace.get("sources") or []),
            "trajectory_manifest_id": self.trajectory_manifest_id,
            "trajectory_version": self.trajectory_version,
            "creator_retrieval": {
                "enabled": bool(creator_retrieval.get("creator_id")),
                "applied": bool(creator_retrieval.get("applied")),
                "reason": creator_retrieval.get("reason"),
                "version": CREATOR_RETRIEVAL_VERSION,
                "creator_id": creator_retrieval.get("creator_id"),
                "confidence": float(creator_retrieval.get("confidence") or 0.0),
                "blend_weight": float(creator_retrieval.get("blend_weight") or 0.0),
                "positive_memory_count": int(creator_retrieval.get("positive_memory_count") or 0),
                "negative_memory_count": int(creator_retrieval.get("negative_memory_count") or 0),
                "branch_meta": dict(creator_retrieval.get("branch_meta") or {}),
            },
        }
        return items, metadata

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        filter_vocab = {
            "language": sorted(
                {
                    str(meta.get("language")).lower()
                    for meta in self.row_metadata.values()
                    if isinstance(meta.get("language"), str) and str(meta.get("language")).strip()
                }
            ),
            "locale": sorted(
                {
                    str(meta.get("locale")).lower()
                    for meta in self.row_metadata.values()
                    if isinstance(meta.get("locale"), str) and str(meta.get("locale")).strip()
                }
            ),
            "content_type": sorted(
                {
                    str(meta.get("content_type")).lower()
                    for meta in self.row_metadata.values()
                    if isinstance(meta.get("content_type"), str) and str(meta.get("content_type")).strip()
                }
            ),
        }
        shard_list = sorted(set(self.row_shards.values()))
        temporal_boundaries = {
            "min": shard_list[0] if shard_list else "unknown",
            "max": shard_list[-1] if shard_list else "unknown",
            "count": len(shard_list),
        }
        dense_dim = (
            _payload_embedding_dim(self.dense_payload)
            if isinstance(self.dense_payload, dict)
            else 0
        )
        multimodal_dim = (
            _payload_embedding_dim(self.multimodal_payload)
            if isinstance(self.multimodal_payload, dict)
            else 0
        )
        trajectory_dim = (
            _payload_embedding_dim(self.trajectory_payload)
            if isinstance(self.trajectory_payload, dict)
            else 0
        )
        index_hash_payload = {
            "artifact_version": self.artifact_version,
            "row_ids": self.row_ids,
            "row_times": self.row_times,
            "row_shards": self.row_shards,
            "sparse_backend": self.sparse_backend,
            "dense_backend": self.dense_backend,
            "multimodal_backend": self.multimodal_backend,
            "graph_backend": self.graph_backend,
            "trajectory_backend": self.trajectory_backend,
            "dense_dim": dense_dim,
            "multimodal_dim": multimodal_dim,
            "graph_dim": _payload_embedding_dim(self.graph_payload),
            "trajectory_dim": trajectory_dim,
        }
        index_hashes = {
            "lexical": _stable_hash(
                {
                    "backend": self.sparse_backend,
                    "row_ids": self.row_ids,
                    "row_text_hash": _stable_hash(self.row_texts),
                }
            ),
            "dense_text": _stable_hash(
                {
                    "backend": self.dense_backend,
                    "base": index_hash_payload,
                    "config": {"dense_model_name": self.config.dense_model_name},
                }
            ),
            "multimodal": _stable_hash(
                {
                    "backend": self.multimodal_backend,
                    "base": index_hash_payload,
                    "source": str(self.multimodal_payload.get("source") if isinstance(self.multimodal_payload, dict) else ""),
                }
            ),
            "graph_dense": _stable_hash(
                {
                    "backend": self.graph_backend,
                    "base": index_hash_payload,
                    "graph_bundle_id": str(
                        self.graph_payload.get("graph_bundle_id")
                        if isinstance(self.graph_payload, dict)
                        else ""
                    ),
                    "source": str(
                        self.graph_payload.get("source")
                        if isinstance(self.graph_payload, dict)
                        else ""
                    ),
                }
            ),
            "trajectory_dense": _stable_hash(
                {
                    "backend": self.trajectory_backend,
                    "base": index_hash_payload,
                    "trajectory_manifest_id": str(
                        self.trajectory_payload.get("trajectory_manifest_id")
                        if isinstance(self.trajectory_payload, dict)
                        else ""
                    ),
                    "source": str(
                        self.trajectory_payload.get("source")
                        if isinstance(self.trajectory_payload, dict)
                        else ""
                    ),
                }
            ),
        }
        payload = {
            "artifact_version": self.artifact_version,
            "row_ids": self.row_ids,
            "row_texts": self.row_texts,
            "row_times": self.row_times,
            "row_metadata": self.row_metadata,
            "row_shards": self.row_shards,
            "temporal_shards": sorted(set(self.row_shards.values())),
            "sparse_backend": self.sparse_backend,
            "dense_backend": self.dense_backend,
            "multimodal_backend": self.multimodal_backend,
            "graph_backend": self.graph_backend,
            "trajectory_backend": self.trajectory_backend,
            "objective_blend": self.objective_blend,
            "supported_filter_vocab": filter_vocab,
            "supported_filter_vocab_schema_hash": _stable_hash(filter_vocab),
            "temporal_shard_boundaries": temporal_boundaries,
            "index_hashes": index_hashes,
            "config": {
                "dense_model_name": self.config.dense_model_name,
                "sparse_weight": self.config.sparse_weight,
                "dense_weight": self.config.dense_weight,
                "multimodal_weight": self.config.multimodal_weight,
                "graph_weight": self.config.graph_weight,
                "trajectory_weight": self.config.trajectory_weight,
            },
            "graph_bundle_id": self.graph_bundle_id,
            "graph_version": self.graph_version,
            "trajectory_manifest_id": self.trajectory_manifest_id,
            "trajectory_version": self.trajectory_version,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with (output_dir / "sparse.pkl").open("wb") as fh:
            pickle.dump(self.sparse_payload, fh)
        with (output_dir / "dense.pkl").open("wb") as fh:
            pickle.dump(self.dense_payload, fh)
        with (output_dir / "multimodal.pkl").open("wb") as fh:
            pickle.dump(self.multimodal_payload, fh)
        with (output_dir / "graph.pkl").open("wb") as fh:
            pickle.dump(self.graph_payload, fh)
        with (output_dir / "trajectory.pkl").open("wb") as fh:
            pickle.dump(self.trajectory_payload, fh)

    @classmethod
    def load(cls, output_dir: Path) -> "HybridRetriever":
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        with (output_dir / "sparse.pkl").open("rb") as fh:
            sparse_payload = pickle.load(fh)
        with (output_dir / "dense.pkl").open("rb") as fh:
            dense_payload = pickle.load(fh)
        multimodal_path = output_dir / "multimodal.pkl"
        if multimodal_path.exists():
            with multimodal_path.open("rb") as fh:
                multimodal_payload = pickle.load(fh)
        else:
            embeddings = np.zeros((len(manifest.get("row_ids", [])), 4), dtype=np.float32)
            multimodal_payload = {"embeddings": embeddings, "dimension": 4, "source": "compat"}
        graph_path = output_dir / "graph.pkl"
        if graph_path.exists():
            with graph_path.open("rb") as fh:
                graph_payload = pickle.load(fh)
        else:
            embeddings = np.zeros((len(manifest.get("row_ids", [])), 4), dtype=np.float32)
            graph_payload = {
                "embeddings": embeddings,
                "dimension": 4,
                "source": "compat",
                "video_lookup": {},
                "creator_lookup": {},
                "hashtag_lookup": {},
                "audio_lookup": {},
                "style_lookup": {},
                "graph_bundle_id": str(manifest.get("graph_bundle_id") or ""),
                "graph_version": str(manifest.get("graph_version") or ""),
            }
        trajectory_path = output_dir / "trajectory.pkl"
        if trajectory_path.exists():
            with trajectory_path.open("rb") as fh:
                trajectory_payload = pickle.load(fh)
        else:
            embeddings = np.zeros((len(manifest.get("row_ids", [])), 4), dtype=np.float32)
            trajectory_payload = {
                "embeddings": embeddings,
                "dimension": 4,
                "source": "compat",
                "video_lookup": {},
                "trajectory_manifest_id": str(manifest.get("trajectory_manifest_id") or ""),
                "trajectory_version": str(manifest.get("trajectory_version") or ""),
            }

        cfg = HybridRetrieverTrainerConfig(
            dense_model_name=manifest.get("config", {}).get(
                "dense_model_name", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            sparse_weight=float(manifest.get("config", {}).get("sparse_weight", 0.5)),
            dense_weight=float(manifest.get("config", {}).get("dense_weight", 0.3)),
            multimodal_weight=float(
                manifest.get("config", {}).get("multimodal_weight", 0.2)
            ),
            graph_weight=float(
                manifest.get("config", {}).get("graph_weight", 0.1)
            ),
            trajectory_weight=float(
                manifest.get("config", {}).get("trajectory_weight", 0.0)
            ),
        )
        objective_blend = manifest.get("objective_blend", {})
        if not isinstance(objective_blend, dict) or not objective_blend:
            objective_blend = {
                objective: _normalize_weight_map(
                    {
                        "lexical": cfg.sparse_weight,
                        "dense_text": cfg.dense_weight,
                        "multimodal": cfg.multimodal_weight,
                        "graph_dense": cfg.graph_weight,
                        "trajectory_dense": cfg.trajectory_weight,
                    }
                )
                for objective in RUNTIME_OBJECTIVES
            }
        row_ids = [str(item) for item in manifest.get("row_ids", [])]
        row_times = {
            str(key): str(value)
            for key, value in dict(manifest.get("row_times", {})).items()
        }
        row_metadata = manifest.get("row_metadata", {})
        if not isinstance(row_metadata, dict):
            row_metadata = {}
        row_shards = manifest.get("row_shards", {})
        if not isinstance(row_shards, dict):
            row_shards = {}
        for row_id in row_ids:
            row_shards.setdefault(row_id, _month_shard(row_times.get(row_id, "")))
            row_metadata.setdefault(
                row_id,
                {
                    "row_id": row_id,
                    "video_id": row_id.split("::", 1)[0],
                    "candidate_id": row_id.split("::", 1)[0],
                    "text": "",
                    "topic_key": "general",
                    "author_id": "unknown",
                    "as_of_time": row_times.get(row_id, ""),
                    "language": None,
                    "locale": None,
                    "content_type": "general",
                },
            )

        return cls(
            row_ids=row_ids,
            row_texts=[str(item) for item in manifest.get("row_texts", [])],
            row_times=row_times,
            row_metadata=row_metadata,
            row_shards=row_shards,
            sparse_backend=str(manifest.get("sparse_backend", "tfidf")),
            sparse_payload=sparse_payload,
            dense_backend=str(manifest.get("dense_backend", "tfidf-char")),
            dense_payload=dense_payload,
            multimodal_backend=str(manifest.get("multimodal_backend", "fabric-precomputed")),
            multimodal_payload=multimodal_payload,
            graph_backend=str(manifest.get("graph_backend", "graph-dna")),
            graph_payload=graph_payload,
            trajectory_backend=str(
                manifest.get("trajectory_backend", "trajectory-precomputed")
            ),
            trajectory_payload=trajectory_payload,
            objective_blend={
                str(key): _normalize_weight_map(value if isinstance(value, dict) else {})
                for key, value in objective_blend.items()
            },
            config=cfg,
            artifact_version=str(manifest.get("artifact_version", "retriever.v1.compat")),
        )
