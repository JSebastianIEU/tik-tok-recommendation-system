from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    lgb = None


SEGMENT_GLOBAL = "global"
SEGMENT_CREATOR_COLD_START = "creator_cold_start"
SEGMENT_CREATOR_MATURE = "creator_mature"
SEGMENT_FORMAT_TUTORIAL = "format_tutorial"
SEGMENT_FORMAT_ENTERTAINMENT = "format_entertainment"
SEGMENT_IDS = (
    SEGMENT_GLOBAL,
    SEGMENT_CREATOR_COLD_START,
    SEGMENT_CREATOR_MATURE,
    SEGMENT_FORMAT_TUTORIAL,
    SEGMENT_FORMAT_ENTERTAINMENT,
)


FEATURE_NAMES = [
    "similarity",
    "query_caption_word_count",
    "query_hashtag_count",
    "query_keyword_count",
    "candidate_caption_word_count",
    "candidate_hashtag_count",
    "candidate_keyword_count",
    "delta_caption_word_count",
    "delta_hashtag_count",
    "delta_keyword_count",
    "same_author",
    "same_topic",
    "query_comment_confusion_index",
    "query_comment_help_seeking_index",
    "query_comment_sentiment_volatility",
    "query_comment_confidence",
    "query_comment_missingness_count",
    "candidate_comment_confusion_index",
    "candidate_comment_help_seeking_index",
    "candidate_comment_sentiment_volatility",
    "candidate_comment_confidence",
    "candidate_comment_missingness_count",
    "delta_comment_confusion_index",
    "delta_comment_help_seeking_index",
    "delta_comment_sentiment_volatility",
    "same_dominant_comment_intent",
    "ratio_caption_word_count",
    "ratio_hashtag_count",
    "ratio_keyword_count",
    "ratio_comment_confusion_index",
    "ratio_comment_help_seeking_index",
    "ratio_comment_sentiment_volatility",
    "query_alignment_score",
    "query_value_prop_coverage",
    "query_on_topic_ratio",
    "query_artifact_drift_ratio",
    "query_alignment_shift_early_late",
    "query_alignment_confidence",
    "candidate_alignment_score",
    "candidate_value_prop_coverage",
    "candidate_on_topic_ratio",
    "candidate_artifact_drift_ratio",
    "candidate_alignment_shift_early_late",
    "candidate_alignment_confidence",
    "delta_alignment_score",
    "delta_value_prop_coverage",
    "delta_on_topic_ratio",
    "delta_artifact_drift_ratio",
    "delta_alignment_shift_early_late",
    "delta_alignment_confidence",
    "ratio_alignment_score",
    "ratio_value_prop_coverage",
    "ratio_on_topic_ratio",
    "ratio_artifact_drift_ratio",
    "ratio_alignment_shift_early_late",
    "cross_similarity_delta_alignment_score",
    "cross_similarity_delta_alignment_shift",
    "cross_similarity_delta_caption",
    "cross_similarity_delta_hashtag",
    "cross_similarity_delta_keyword",
    "cross_similarity_delta_comment_confusion",
    "cross_similarity_delta_comment_help",
    "cross_similarity_delta_comment_volatility",
    "cross_similarity_same_topic",
    "cross_similarity_same_author",
    "candidate_is_tutorial",
    "candidate_is_entertainment",
    "cross_tutorial_similarity",
    "cross_entertainment_similarity",
    "cross_tutorial_delta_keyword",
    "cross_entertainment_delta_comment_volatility",
    "query_creator_cold_start",
    "query_creator_mature",
    "graph_similarity_hint",
    "graph_shared_hashtag_jaccard",
    "graph_shared_style_jaccard",
    "graph_same_audio_motif",
    "graph_creator_strength_delta",
    "graph_cross_similarity_hint",
    "query_traj_early_velocity",
    "query_traj_core_velocity",
    "query_traj_late_lift",
    "query_traj_stability",
    "query_traj_durability_ratio",
    "query_traj_acceleration",
    "query_traj_curvature",
    "query_traj_peak_lag_hours",
    "query_traj_regime_spike",
    "query_traj_regime_balanced",
    "query_traj_regime_durable",
    "query_traj_regime_confidence",
    "query_traj_available_ratio",
    "query_traj_missing_component_count",
    "candidate_traj_early_velocity",
    "candidate_traj_core_velocity",
    "candidate_traj_late_lift",
    "candidate_traj_stability",
    "candidate_traj_durability_ratio",
    "candidate_traj_acceleration",
    "candidate_traj_curvature",
    "candidate_traj_peak_lag_hours",
    "candidate_traj_regime_spike",
    "candidate_traj_regime_balanced",
    "candidate_traj_regime_durable",
    "candidate_traj_regime_confidence",
    "candidate_traj_available_ratio",
    "candidate_traj_missing_component_count",
    "delta_traj_early_velocity",
    "delta_traj_core_velocity",
    "delta_traj_late_lift",
    "delta_traj_stability",
    "delta_traj_durability_ratio",
    "delta_traj_acceleration",
    "delta_traj_curvature",
    "delta_traj_peak_lag_hours",
    "delta_traj_regime_confidence",
    "same_trajectory_regime",
    "traj_similarity_hint",
    "cross_similarity_delta_traj_early",
    "cross_similarity_delta_traj_late",
    "cross_similarity_delta_traj_stability",
]


def _as_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    if np.isfinite(out):
        return out
    return fallback


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(1.0, denominator)


def _row_feature(row: Dict[str, Any], key: str) -> float:
    features = row.get("features", {})
    if not isinstance(features, dict):
        return 0.0
    return _as_float(features.get(key), 0.0)


def _content_type_bucket(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw == "tutorial":
        return "tutorial"
    if raw in {"entertainment", "reaction", "showcase"}:
        return "entertainment"
    return "other"


def _trajectory_feature_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    features = row.get("features", {})
    if not isinstance(features, dict):
        return {}
    payload = features.get("trajectory_features")
    if isinstance(payload, dict):
        return payload
    return {}


def creator_stage_from_count(prior_video_count: Optional[int], cold_threshold: int = 10) -> Optional[str]:
    if prior_video_count is None:
        return None
    if prior_video_count < cold_threshold:
        return SEGMENT_CREATOR_COLD_START
    return SEGMENT_CREATOR_MATURE


def query_creator_prior_count(row: Dict[str, Any]) -> Optional[int]:
    candidates = [
        row.get("_creator_prior_video_count"),
        row.get("creator_prior_video_count"),
    ]
    features = row.get("features", {})
    if isinstance(features, dict):
        candidates.extend(
            [
                features.get("creator_prior_video_count"),
                features.get("author_prior_video_count"),
                features.get("author_video_count"),
            ]
        )
    for value in candidates:
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            return parsed
    return None


def query_creator_stage(row: Dict[str, Any], cold_threshold: int = 10) -> Optional[str]:
    return creator_stage_from_count(
        query_creator_prior_count(row),
        cold_threshold=cold_threshold,
    )


def format_segment_for_candidate(row: Dict[str, Any]) -> Optional[str]:
    bucket = _content_type_bucket(row.get("content_type"))
    if bucket == "tutorial":
        return SEGMENT_FORMAT_TUTORIAL
    if bucket == "entertainment":
        return SEGMENT_FORMAT_ENTERTAINMENT
    return None


def segment_candidates_for_pair(
    *,
    query_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
    creator_cold_threshold: int = 10,
) -> List[str]:
    out: List[str] = []
    creator_stage = query_creator_stage(query_row, cold_threshold=creator_cold_threshold)
    if creator_stage in {SEGMENT_CREATOR_COLD_START, SEGMENT_CREATOR_MATURE}:
        out.append(creator_stage)
    format_segment = format_segment_for_candidate(candidate_row)
    if format_segment in {SEGMENT_FORMAT_TUTORIAL, SEGMENT_FORMAT_ENTERTAINMENT}:
        out.append(format_segment)
    return out


def _pair_feature_vector(
    query_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
    similarity: float,
) -> List[float]:
    q_caption = _row_feature(query_row, "caption_word_count")
    q_hashtag = _row_feature(query_row, "hashtag_count")
    q_keyword = _row_feature(query_row, "keyword_count")

    c_caption = _row_feature(candidate_row, "caption_word_count")
    c_hashtag = _row_feature(candidate_row, "hashtag_count")
    c_keyword = _row_feature(candidate_row, "keyword_count")
    q_comment = (
        query_row.get("features", {}).get("comment_intelligence", {})
        if isinstance(query_row.get("features"), dict)
        else {}
    )
    c_comment = (
        candidate_row.get("features", {}).get("comment_intelligence", {})
        if isinstance(candidate_row.get("features"), dict)
        else {}
    )
    q_comment_z = q_comment.get("z_features") if isinstance(q_comment.get("z_features"), dict) else {}
    c_comment_z = c_comment.get("z_features") if isinstance(c_comment.get("z_features"), dict) else {}

    def _comment_value(
        comment_payload: Dict[str, Any],
        z_payload: Dict[str, Any],
        key: str,
        *,
        use_z: bool = False,
        fallback: float = 0.0,
    ) -> float:
        if use_z:
            return _as_float(z_payload.get(key), _as_float(comment_payload.get(key), fallback))
        return _as_float(comment_payload.get(key), fallback)

    q_traj = _trajectory_feature_payload(query_row)
    c_traj = _trajectory_feature_payload(candidate_row)
    q_confusion = _as_float(q_comment.get("confusion_index"), 0.0)
    q_help = _as_float(q_comment.get("help_seeking_index"), 0.0)
    q_vol = _as_float(q_comment.get("sentiment_volatility"), 0.0)
    q_confidence = _as_float(q_comment.get("confidence"), 0.0)
    q_missing_flags = q_comment.get("missingness_flags")
    q_missing = float(len(q_missing_flags)) if isinstance(q_missing_flags, list) else 0.0

    c_confusion = _as_float(c_comment.get("confusion_index"), 0.0)
    c_help = _as_float(c_comment.get("help_seeking_index"), 0.0)
    c_vol = _as_float(c_comment.get("sentiment_volatility"), 0.0)
    c_confidence = _as_float(c_comment.get("confidence"), 0.0)
    c_missing_flags = c_comment.get("missingness_flags")
    c_missing = float(len(c_missing_flags)) if isinstance(c_missing_flags, list) else 0.0
    q_dom = str((q_comment.get("dominant_intents") or [""])[0] or "")
    c_dom = str((c_comment.get("dominant_intents") or [""])[0] or "")
    q_alignment_score = _comment_value(q_comment, q_comment_z, "alignment_score", use_z=True)
    q_alignment_cov = _comment_value(q_comment, q_comment_z, "value_prop_coverage", use_z=True)
    q_alignment_topic = _comment_value(q_comment, q_comment_z, "on_topic_ratio", use_z=True)
    q_alignment_drift = _comment_value(q_comment, q_comment_z, "artifact_drift_ratio", use_z=True)
    q_alignment_shift = _comment_value(
        q_comment,
        q_comment_z,
        "alignment_shift_early_late",
        use_z=True,
    )
    q_alignment_conf = _comment_value(q_comment, q_comment_z, "alignment_confidence")

    c_alignment_score = _comment_value(c_comment, c_comment_z, "alignment_score", use_z=True)
    c_alignment_cov = _comment_value(c_comment, c_comment_z, "value_prop_coverage", use_z=True)
    c_alignment_topic = _comment_value(c_comment, c_comment_z, "on_topic_ratio", use_z=True)
    c_alignment_drift = _comment_value(c_comment, c_comment_z, "artifact_drift_ratio", use_z=True)
    c_alignment_shift = _comment_value(
        c_comment,
        c_comment_z,
        "alignment_shift_early_late",
        use_z=True,
    )
    c_alignment_conf = _comment_value(c_comment, c_comment_z, "alignment_confidence")

    q_regime_probs = q_traj.get("regime_probabilities", {})
    if not isinstance(q_regime_probs, dict):
        q_regime_probs = {}
    c_regime_probs = c_traj.get("regime_probabilities", {})
    if not isinstance(c_regime_probs, dict):
        c_regime_probs = {}
    q_regime = str(q_traj.get("regime_pred") or "")
    c_regime = str(c_traj.get("regime_pred") or "")
    q_traj_early = _as_float(q_traj.get("early_velocity"), 0.0)
    q_traj_core = _as_float(q_traj.get("core_velocity"), 0.0)
    q_traj_late = _as_float(q_traj.get("late_lift"), 0.0)
    q_traj_stability = _as_float(q_traj.get("stability"), 0.0)
    q_traj_durability = _as_float(q_traj.get("durability_ratio"), 0.0)
    q_traj_acceleration = _as_float(q_traj.get("acceleration_proxy"), 0.0)
    q_traj_curvature = _as_float(q_traj.get("curvature_proxy"), 0.0)
    q_traj_peak_lag = _as_float(q_traj.get("peak_lag_hours"), 0.0)
    q_traj_confidence = _as_float(q_traj.get("regime_confidence"), 0.0)
    q_traj_available_ratio = _as_float(q_traj.get("available_ratio"), 0.0)
    q_traj_missing_count = _as_float(q_traj.get("missing_component_count"), 0.0)

    c_traj_early = _as_float(c_traj.get("early_velocity"), 0.0)
    c_traj_core = _as_float(c_traj.get("core_velocity"), 0.0)
    c_traj_late = _as_float(c_traj.get("late_lift"), 0.0)
    c_traj_stability = _as_float(c_traj.get("stability"), 0.0)
    c_traj_durability = _as_float(c_traj.get("durability_ratio"), 0.0)
    c_traj_acceleration = _as_float(c_traj.get("acceleration_proxy"), 0.0)
    c_traj_curvature = _as_float(c_traj.get("curvature_proxy"), 0.0)
    c_traj_peak_lag = _as_float(c_traj.get("peak_lag_hours"), 0.0)
    c_traj_confidence = _as_float(c_traj.get("regime_confidence"), 0.0)
    c_traj_available_ratio = _as_float(c_traj.get("available_ratio"), 0.0)
    c_traj_missing_count = _as_float(c_traj.get("missing_component_count"), 0.0)

    delta_caption = abs(q_caption - c_caption)
    delta_hashtag = abs(q_hashtag - c_hashtag)
    delta_keyword = abs(q_keyword - c_keyword)
    delta_confusion = abs(q_confusion - c_confusion)
    delta_help = abs(q_help - c_help)
    delta_vol = abs(q_vol - c_vol)
    same_author = 1.0 if query_row.get("author_id") == candidate_row.get("author_id") else 0.0
    same_topic = 1.0 if query_row.get("topic_key") == candidate_row.get("topic_key") else 0.0

    ratio_caption = _safe_ratio(c_caption, q_caption)
    ratio_hashtag = _safe_ratio(c_hashtag, q_hashtag)
    ratio_keyword = _safe_ratio(c_keyword, q_keyword)
    ratio_confusion = _safe_ratio(c_confusion, q_confusion)
    ratio_help = _safe_ratio(c_help, q_help)
    ratio_vol = _safe_ratio(c_vol, q_vol)
    delta_alignment_score = abs(q_alignment_score - c_alignment_score)
    delta_alignment_cov = abs(q_alignment_cov - c_alignment_cov)
    delta_alignment_topic = abs(q_alignment_topic - c_alignment_topic)
    delta_alignment_drift = abs(q_alignment_drift - c_alignment_drift)
    delta_alignment_shift = abs(q_alignment_shift - c_alignment_shift)
    delta_alignment_conf = abs(q_alignment_conf - c_alignment_conf)
    ratio_alignment_score = _safe_ratio(c_alignment_score, q_alignment_score)
    ratio_alignment_cov = _safe_ratio(c_alignment_cov, q_alignment_cov)
    ratio_alignment_topic = _safe_ratio(c_alignment_topic, q_alignment_topic)
    ratio_alignment_drift = _safe_ratio(c_alignment_drift, q_alignment_drift)
    ratio_alignment_shift = _safe_ratio(c_alignment_shift, q_alignment_shift)

    content_bucket = _content_type_bucket(candidate_row.get("content_type"))
    is_tutorial = 1.0 if content_bucket == "tutorial" else 0.0
    is_entertainment = 1.0 if content_bucket == "entertainment" else 0.0
    creator_stage = query_creator_stage(query_row)
    creator_cold = 1.0 if creator_stage == SEGMENT_CREATOR_COLD_START else 0.0
    creator_mature = 1.0 if creator_stage == SEGMENT_CREATOR_MATURE else 0.0

    q_hashes = set(str(item).strip().lower() for item in list(query_row.get("hashtags") or []) if str(item).strip())
    c_hashes = set(str(item).strip().lower() for item in list(candidate_row.get("hashtags") or []) if str(item).strip())
    if not q_hashes:
        query_text = str(query_row.get("_runtime_text") or query_row.get("topic_key") or "")
        q_hashes = {token.strip().lower() for token in query_text.split() if token.strip().startswith("#")}
    if not c_hashes:
        candidate_text = str(candidate_row.get("_runtime_text") or candidate_row.get("topic_key") or "")
        c_hashes = {token.strip().lower() for token in candidate_text.split() if token.strip().startswith("#")}
    hash_union = len(q_hashes | c_hashes)
    shared_hashtag_jaccard = (
        float(len(q_hashes & c_hashes)) / float(hash_union) if hash_union > 0 else 0.0
    )

    q_fabric = query_row.get("_fabric_output")
    c_fabric = candidate_row.get("_fabric_output")
    q_styles = (
        set(str(item).strip().lower() for item in list(((q_fabric.get("visual") if isinstance(q_fabric, dict) else {}) or {}).get("style_tags") or []) if str(item).strip())
        if isinstance(q_fabric, dict)
        else set()
    )
    c_styles = (
        set(str(item).strip().lower() for item in list(((c_fabric.get("visual") if isinstance(c_fabric, dict) else {}) or {}).get("style_tags") or []) if str(item).strip())
        if isinstance(c_fabric, dict)
        else set()
    )
    style_union = len(q_styles | c_styles)
    shared_style_jaccard = (
        float(len(q_styles & c_styles)) / float(style_union) if style_union > 0 else 0.0
    )

    q_audio = q_fabric.get("audio") if isinstance(q_fabric, dict) else {}
    c_audio = c_fabric.get("audio") if isinstance(c_fabric, dict) else {}
    q_music = bool(q_audio.get("music_presence")) if isinstance(q_audio, dict) else False
    c_music = bool(c_audio.get("music_presence")) if isinstance(c_audio, dict) else False
    same_audio_motif = 1.0 if q_music == c_music else 0.0

    graph_similarity_hint = _as_float(
        (candidate_row.get("features", {}) if isinstance(candidate_row.get("features"), dict) else {}).get("graph_similarity_hint"),
        _as_float(candidate_row.get("_graph_similarity"), 0.0),
    )
    q_creator_strength = _as_float(
        (query_row.get("features", {}) if isinstance(query_row.get("features"), dict) else {}).get("graph_creator_neighbor_strength"),
        0.0,
    )
    c_creator_strength = _as_float(
        (candidate_row.get("features", {}) if isinstance(candidate_row.get("features"), dict) else {}).get("graph_creator_neighbor_strength"),
        0.0,
    )
    graph_creator_strength_delta = abs(q_creator_strength - c_creator_strength)

    delta_traj_early = abs(q_traj_early - c_traj_early)
    delta_traj_core = abs(q_traj_core - c_traj_core)
    delta_traj_late = abs(q_traj_late - c_traj_late)
    delta_traj_stability = abs(q_traj_stability - c_traj_stability)
    delta_traj_durability = abs(q_traj_durability - c_traj_durability)
    delta_traj_acceleration = abs(q_traj_acceleration - c_traj_acceleration)
    delta_traj_curvature = abs(q_traj_curvature - c_traj_curvature)
    delta_traj_peak_lag = abs(q_traj_peak_lag - c_traj_peak_lag)
    delta_traj_confidence = abs(q_traj_confidence - c_traj_confidence)
    trajectory_same_regime = 1.0 if q_regime and q_regime == c_regime else 0.0
    traj_similarity_hint = 1.0 / (
        1.0
        + delta_traj_early
        + delta_traj_late
        + delta_traj_stability
        + delta_traj_durability
    )

    sim = _as_float(similarity, 0.0)
    return [
        sim,
        q_caption,
        q_hashtag,
        q_keyword,
        c_caption,
        c_hashtag,
        c_keyword,
        delta_caption,
        delta_hashtag,
        delta_keyword,
        same_author,
        same_topic,
        q_confusion,
        q_help,
        q_vol,
        q_confidence,
        q_missing,
        c_confusion,
        c_help,
        c_vol,
        c_confidence,
        c_missing,
        delta_confusion,
        delta_help,
        delta_vol,
        1.0 if q_dom and q_dom == c_dom else 0.0,
        ratio_caption,
        ratio_hashtag,
        ratio_keyword,
        ratio_confusion,
        ratio_help,
        ratio_vol,
        q_alignment_score,
        q_alignment_cov,
        q_alignment_topic,
        q_alignment_drift,
        q_alignment_shift,
        q_alignment_conf,
        c_alignment_score,
        c_alignment_cov,
        c_alignment_topic,
        c_alignment_drift,
        c_alignment_shift,
        c_alignment_conf,
        delta_alignment_score,
        delta_alignment_cov,
        delta_alignment_topic,
        delta_alignment_drift,
        delta_alignment_shift,
        delta_alignment_conf,
        ratio_alignment_score,
        ratio_alignment_cov,
        ratio_alignment_topic,
        ratio_alignment_drift,
        ratio_alignment_shift,
        sim * delta_alignment_score,
        sim * delta_alignment_shift,
        sim * delta_caption,
        sim * delta_hashtag,
        sim * delta_keyword,
        sim * delta_confusion,
        sim * delta_help,
        sim * delta_vol,
        sim * same_topic,
        sim * same_author,
        is_tutorial,
        is_entertainment,
        is_tutorial * sim,
        is_entertainment * sim,
        is_tutorial * delta_keyword,
        is_entertainment * delta_vol,
        creator_cold,
        creator_mature,
        graph_similarity_hint,
        shared_hashtag_jaccard,
        shared_style_jaccard,
        same_audio_motif,
        graph_creator_strength_delta,
        sim * graph_similarity_hint,
        q_traj_early,
        q_traj_core,
        q_traj_late,
        q_traj_stability,
        q_traj_durability,
        q_traj_acceleration,
        q_traj_curvature,
        q_traj_peak_lag,
        _as_float(q_regime_probs.get("spike"), 0.0),
        _as_float(q_regime_probs.get("balanced"), 0.0),
        _as_float(q_regime_probs.get("durable"), 0.0),
        q_traj_confidence,
        q_traj_available_ratio,
        q_traj_missing_count,
        c_traj_early,
        c_traj_core,
        c_traj_late,
        c_traj_stability,
        c_traj_durability,
        c_traj_acceleration,
        c_traj_curvature,
        c_traj_peak_lag,
        _as_float(c_regime_probs.get("spike"), 0.0),
        _as_float(c_regime_probs.get("balanced"), 0.0),
        _as_float(c_regime_probs.get("durable"), 0.0),
        c_traj_confidence,
        c_traj_available_ratio,
        c_traj_missing_count,
        delta_traj_early,
        delta_traj_core,
        delta_traj_late,
        delta_traj_stability,
        delta_traj_durability,
        delta_traj_acceleration,
        delta_traj_curvature,
        delta_traj_peak_lag,
        delta_traj_confidence,
        trajectory_same_regime,
        traj_similarity_hint,
        sim * delta_traj_early,
        sim * delta_traj_late,
        sim * delta_traj_stability,
    ]


def pair_feature_vector_array(
    query_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
    similarity: float,
) -> np.ndarray:
    return np.asarray(
        [_pair_feature_vector(query_row, candidate_row, similarity)],
        dtype=np.float32,
    )


def build_ranker_training_frame(
    rows_by_id: Dict[str, Dict[str, Any]],
    pair_rows: Sequence[Dict[str, Any]],
    objective: str,
    query_split: str = "train",
    candidate_split: Optional[str] = "train",
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    grouped: Dict[str, List[Tuple[List[float], float]]] = {}
    for pair in pair_rows:
        if str(pair.get("objective")) != objective:
            continue
        query_id = str(pair.get("query_row_id"))
        candidate_id = str(pair.get("candidate_row_id"))
        query_row = rows_by_id.get(query_id)
        candidate_row = rows_by_id.get(candidate_id)
        if query_row is None or candidate_row is None:
            continue
        if str(query_row.get("split")) != query_split:
            continue
        if candidate_split and str(candidate_row.get("split")) != candidate_split:
            continue
        feature_vec = _pair_feature_vector(
            query_row=query_row,
            candidate_row=candidate_row,
            similarity=_as_float(pair.get("similarity"), 0.0),
        )
        label = _as_float(pair.get("relevance_label"), 0.0)
        grouped.setdefault(query_id, []).append((feature_vec, label))

    X: List[List[float]] = []
    y: List[float] = []
    groups: List[int] = []
    query_ids: List[str] = []
    for query_id, rows in grouped.items():
        if len(rows) < 2:
            continue
        groups.append(len(rows))
        query_ids.append(query_id)
        for feature_vec, label in rows:
            X.append(feature_vec)
            y.append(label)

    if not X:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros((0,)), [], []
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), groups, query_ids


@dataclass
class ObjectiveRankerConfig:
    objective: str
    n_estimators: int = 200
    random_state: int = 13


class ObjectiveRankerModel:
    def __init__(
        self,
        objective: str,
        backend: str,
        model: Any,
        calibrator: Optional[IsotonicRegression] = None,
    ) -> None:
        self.objective = objective
        self.backend = backend
        self.model = model
        self.calibrator = calibrator

    @classmethod
    def train(
        cls,
        config: ObjectiveRankerConfig,
        rows_by_id: Dict[str, Dict[str, Any]],
        pair_rows: Sequence[Dict[str, Any]],
    ) -> "ObjectiveRankerModel":
        X_train, y_train, groups, _ = build_ranker_training_frame(
            rows_by_id=rows_by_id,
            pair_rows=pair_rows,
            objective=config.objective,
            query_split="train",
            candidate_split="train",
        )
        if X_train.shape[0] == 0:
            raise ValueError(f"No training rows available for objective '{config.objective}'.")

        backend = "sklearn-hgbr"
        if lgb is not None:
            model = lgb.LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=config.n_estimators,
                learning_rate=0.05,
                num_leaves=31,
                random_state=config.random_state,
            )
            model.fit(X_train, y_train, group=groups)
            backend = "lightgbm-lambdarank"
        else:
            model = HistGradientBoostingRegressor(
                max_iter=config.n_estimators,
                learning_rate=0.05,
                random_state=config.random_state,
            )
            model.fit(X_train, y_train)

        calibrator = None
        X_val, y_val, _, _ = build_ranker_training_frame(
            rows_by_id=rows_by_id,
            pair_rows=pair_rows,
            objective=config.objective,
            query_split="validation",
            candidate_split="train",
        )
        if X_val.shape[0] >= 20 and len(set(y_val.tolist())) >= 2:
            val_pred = model.predict(X_val)
            calibrator = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                out_of_bounds="clip",
            )
            calibrator.fit(val_pred, np.clip(y_val / 3.0, 0.0, 1.0))

        return cls(
            objective=config.objective,
            backend=backend,
            model=model,
            calibrator=calibrator,
        )

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.model.predict(X), dtype=np.float32)
        if self.calibrator is None:
            return raw
        calibrated = np.asarray(self.calibrator.predict(raw), dtype=np.float32)
        return calibrated

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "objective": self.objective,
            "backend": self.backend,
            "feature_names": FEATURE_NAMES,
            "has_calibrator": self.calibrator is not None,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with (output_dir / "model.pkl").open("wb") as fh:
            pickle.dump(self.model, fh)
        if self.calibrator is not None:
            with (output_dir / "calibrator.pkl").open("wb") as fh:
                pickle.dump(self.calibrator, fh)

    @classmethod
    def load(cls, output_dir: Path) -> "ObjectiveRankerModel":
        meta = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        with (output_dir / "model.pkl").open("rb") as fh:
            model = pickle.load(fh)
        calibrator = None
        calibrator_path = output_dir / "calibrator.pkl"
        if calibrator_path.exists():
            with calibrator_path.open("rb") as fh:
                calibrator = pickle.load(fh)
        return cls(
            objective=str(meta["objective"]),
            backend=str(meta["backend"]),
            model=model,
            calibrator=calibrator,
        )


def _bootstrap_pair_rows(
    *,
    rows_by_id: Dict[str, Dict[str, Any]],
    pair_rows: Sequence[Dict[str, Any]],
    objective: str,
    seed: int,
) -> List[Dict[str, Any]]:
    train_pairs: List[Dict[str, Any]] = []
    holdout_pairs: List[Dict[str, Any]] = []
    for pair in pair_rows:
        if str(pair.get("objective")) != objective:
            continue
        qid = str(pair.get("query_row_id"))
        cid = str(pair.get("candidate_row_id"))
        query_row = rows_by_id.get(qid)
        candidate_row = rows_by_id.get(cid)
        if query_row is None or candidate_row is None:
            continue
        if str(query_row.get("split")) == "train" and str(candidate_row.get("split")) == "train":
            train_pairs.append(dict(pair))
        else:
            holdout_pairs.append(dict(pair))
    if not train_pairs:
        return [dict(pair) for pair in holdout_pairs]
    rng = random.Random(seed)
    sampled_train = [dict(rng.choice(train_pairs)) for _ in range(len(train_pairs))]
    return [*sampled_train, *holdout_pairs]


@dataclass
class RankerFamilyConfig:
    objective: str
    ensemble_size: int = 5
    random_seed: int = 13
    std_ref: float = 0.15
    creator_cold_threshold: int = 10


def _trajectory_head_output(
    *,
    objective: str,
    query_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
) -> Dict[str, Any]:
    q_payload = _trajectory_feature_payload(query_row)
    c_payload = _trajectory_feature_payload(candidate_row)
    q_objectives = q_payload.get("objectives", {}) if isinstance(q_payload, dict) else {}
    c_objectives = c_payload.get("objectives", {}) if isinstance(c_payload, dict) else {}
    q_objective = q_objectives.get(objective, {}) if isinstance(q_objectives, dict) else {}
    c_objective = c_objectives.get(objective, {}) if isinstance(c_objectives, dict) else {}
    q_comp = _as_float(
        q_objective.get("composite_z") if isinstance(q_objective, dict) else None,
        0.0,
    )
    c_comp = _as_float(
        c_objective.get("composite_z") if isinstance(c_objective, dict) else None,
        0.0,
    )
    q_regime_probs = q_payload.get("regime_probabilities", {}) if isinstance(q_payload, dict) else {}
    if not isinstance(q_regime_probs, dict):
        q_regime_probs = {}
    c_regime_probs = c_payload.get("regime_probabilities", {}) if isinstance(c_payload, dict) else {}
    if not isinstance(c_regime_probs, dict):
        c_regime_probs = {}
    regime_probs = {
        regime: float(
            _clip(
                (0.6 * _as_float(c_regime_probs.get(regime), 0.0))
                + (0.4 * _as_float(q_regime_probs.get(regime), 0.0)),
                0.0,
                1.0,
            )
        )
        for regime in ("spike", "balanced", "durable")
    }
    total = sum(regime_probs.values())
    if total > 0:
        regime_probs = {key: float(round(value / total, 6)) for key, value in regime_probs.items()}
    regime_pred = max(regime_probs.items(), key=lambda item: item[1])[0]
    confidence = float(round(max(regime_probs.values()) if regime_probs else 0.0, 6))
    traj_similarity = 1.0 / (1.0 + abs(c_comp - q_comp))
    available = bool(c_payload)
    return {
        "trajectory_score": float(round(c_comp, 6)),
        "trajectory_similarity": float(round(traj_similarity, 6)),
        "trajectory_regime_probabilities": regime_probs,
        "trajectory_regime_pred": regime_pred,
        "trajectory_regime_confidence": confidence,
        "trajectory_available": available,
    }


class RankerEnsembleModel:
    def __init__(
        self,
        objective: str,
        segment_id: str,
        models: Sequence[ObjectiveRankerModel],
        std_ref: float = 0.15,
    ) -> None:
        if not models:
            raise ValueError("RankerEnsembleModel requires at least one member model.")
        self.objective = objective
        self.segment_id = segment_id
        self.models = list(models)
        self.std_ref = max(1e-6, float(std_ref))

    @classmethod
    def train(
        cls,
        *,
        config: RankerFamilyConfig,
        rows_by_id: Dict[str, Dict[str, Any]],
        pair_rows: Sequence[Dict[str, Any]],
        segment_id: str,
    ) -> "RankerEnsembleModel":
        members: List[ObjectiveRankerModel] = []
        for index in range(max(1, int(config.ensemble_size))):
            bootstrap_seed = int(config.random_seed) + (index * 991)
            member_pairs = _bootstrap_pair_rows(
                rows_by_id=rows_by_id,
                pair_rows=pair_rows,
                objective=config.objective,
                seed=bootstrap_seed,
            )
            model = ObjectiveRankerModel.train(
                config=ObjectiveRankerConfig(
                    objective=config.objective,
                    random_state=int(config.random_seed) + (index * 17),
                ),
                rows_by_id=rows_by_id,
                pair_rows=member_pairs,
            )
            members.append(model)
        return cls(
            objective=config.objective,
            segment_id=segment_id,
            models=members,
            std_ref=config.std_ref,
        )

    def predict_stats(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        member_scores = [
            np.asarray(model.predict_scores(X), dtype=np.float32)
            for model in self.models
        ]
        matrix = np.vstack(member_scores)
        mean_scores = np.mean(matrix, axis=0).astype(np.float32)
        if matrix.shape[0] <= 1:
            std_scores = np.zeros_like(mean_scores, dtype=np.float32)
        else:
            std_scores = np.std(matrix, axis=0).astype(np.float32)
        return mean_scores, std_scores

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "objective": self.objective,
            "segment_id": self.segment_id,
            "ensemble_size": len(self.models),
            "std_ref": self.std_ref,
        }
        (output_dir / "ensemble_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        for idx, model in enumerate(self.models):
            model.save(output_dir / f"member_{idx:02d}")

    @classmethod
    def load(cls, output_dir: Path) -> "RankerEnsembleModel":
        manifest = json.loads((output_dir / "ensemble_manifest.json").read_text(encoding="utf-8"))
        ensemble_size = int(manifest.get("ensemble_size") or 0)
        members: List[ObjectiveRankerModel] = []
        for idx in range(ensemble_size):
            member_dir = output_dir / f"member_{idx:02d}"
            if member_dir.exists():
                members.append(ObjectiveRankerModel.load(member_dir))
        if not members:
            # backward-compatible load for single model directory
            if (output_dir / "metadata.json").exists():
                members = [ObjectiveRankerModel.load(output_dir)]
            else:
                raise ValueError(f"No ensemble members found at {output_dir}")
        return cls(
            objective=str(manifest.get("objective") or members[0].objective),
            segment_id=str(manifest.get("segment_id") or SEGMENT_GLOBAL),
            models=members,
            std_ref=float(manifest.get("std_ref") or 0.15),
        )


class RankerFamilyModel:
    def __init__(
        self,
        objective: str,
        global_ensemble: RankerEnsembleModel,
        segment_ensembles: Dict[str, RankerEnsembleModel],
        promoted_segments: Sequence[str],
        std_ref: float = 0.15,
        creator_cold_threshold: int = 10,
    ) -> None:
        self.objective = objective
        self.global_ensemble = global_ensemble
        self.segment_ensembles = dict(segment_ensembles)
        self.promoted_segments = list(promoted_segments)
        self.std_ref = max(1e-6, float(std_ref))
        self.creator_cold_threshold = int(creator_cold_threshold)

    def score_pair(
        self,
        *,
        query_row: Dict[str, Any],
        candidate_row: Dict[str, Any],
        similarity: float,
    ) -> Dict[str, Any]:
        pair_vec = pair_feature_vector_array(
            query_row=query_row,
            candidate_row=candidate_row,
            similarity=similarity,
        )
        global_mean_arr, global_std_arr = self.global_ensemble.predict_stats(pair_vec)
        global_mean = float(global_mean_arr[0])
        global_std = float(global_std_arr[0])
        selected_ranker_id = SEGMENT_GLOBAL
        segment_mean = global_mean
        segment_std = global_std
        segment_used = False

        segment_candidates = segment_candidates_for_pair(
            query_row=query_row,
            candidate_row=candidate_row,
            creator_cold_threshold=self.creator_cold_threshold,
        )
        for segment_id in segment_candidates:
            if segment_id not in self.promoted_segments:
                continue
            ensemble = self.segment_ensembles.get(segment_id)
            if ensemble is None:
                continue
            segment_mean_arr, segment_std_arr = ensemble.predict_stats(pair_vec)
            segment_mean = float(segment_mean_arr[0])
            segment_std = float(segment_std_arr[0])
            selected_ranker_id = segment_id
            segment_used = True
            break

        if segment_used:
            blend_weight = _clip(1.0 - (segment_std / self.std_ref), 0.2, 1.0)
            final_score = (blend_weight * segment_mean) + ((1.0 - blend_weight) * global_mean)
            confidence = _clip(1.0 - (segment_std / self.std_ref), 0.0, 1.0)
        else:
            blend_weight = 0.0
            final_score = global_mean
            confidence = _clip(1.0 - (global_std / self.std_ref), 0.0, 1.0)

        trajectory_head = _trajectory_head_output(
            objective=self.objective,
            query_row=query_row,
            candidate_row=candidate_row,
        )

        return {
            "final_score": float(final_score),
            "score_mean": float(segment_mean if segment_used else global_mean),
            "score_std": float(segment_std if segment_used else global_std),
            "confidence": float(confidence),
            "selected_ranker_id": selected_ranker_id,
            "global_score_mean": float(global_mean),
            "segment_blend_weight": float(blend_weight),
            "segment_candidates": segment_candidates,
            "trajectory_score": float(trajectory_head["trajectory_score"]),
            "trajectory_similarity": float(trajectory_head["trajectory_similarity"]),
            "trajectory_regime_probabilities": dict(
                trajectory_head["trajectory_regime_probabilities"]
            ),
            "trajectory_regime_pred": str(trajectory_head["trajectory_regime_pred"]),
            "trajectory_regime_confidence": float(
                trajectory_head["trajectory_regime_confidence"]
            ),
            "trajectory_available": bool(trajectory_head["trajectory_available"]),
        }

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        family_manifest = {
            "objective": self.objective,
            "format": "ranker_family.v2",
            "feature_names": FEATURE_NAMES,
            "std_ref": self.std_ref,
            "creator_cold_threshold": self.creator_cold_threshold,
            "promoted_segments": self.promoted_segments,
            "available_segments": sorted(self.segment_ensembles.keys()),
            "segment_order": [
                SEGMENT_CREATOR_COLD_START,
                SEGMENT_CREATOR_MATURE,
                SEGMENT_FORMAT_TUTORIAL,
                SEGMENT_FORMAT_ENTERTAINMENT,
            ],
        }
        (output_dir / "family_manifest.json").write_text(
            json.dumps(family_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.global_ensemble.save(output_dir / "global")
        segments_root = output_dir / "segments"
        segments_root.mkdir(parents=True, exist_ok=True)
        for segment_id, ensemble in self.segment_ensembles.items():
            ensemble.save(segments_root / segment_id)

    @classmethod
    def load(cls, output_dir: Path) -> "RankerFamilyModel":
        family_manifest_path = output_dir / "family_manifest.json"
        if not family_manifest_path.exists():
            # backward compatible with single-model objective directory
            legacy = ObjectiveRankerModel.load(output_dir)
            global_ensemble = RankerEnsembleModel(
                objective=legacy.objective,
                segment_id=SEGMENT_GLOBAL,
                models=[legacy],
                std_ref=0.15,
            )
            return cls(
                objective=legacy.objective,
                global_ensemble=global_ensemble,
                segment_ensembles={},
                promoted_segments=[],
                std_ref=0.15,
                creator_cold_threshold=10,
            )

        payload = json.loads(family_manifest_path.read_text(encoding="utf-8"))
        objective = str(payload.get("objective") or "")
        global_ensemble = RankerEnsembleModel.load(output_dir / "global")
        segments_root = output_dir / "segments"
        segment_ensembles: Dict[str, RankerEnsembleModel] = {}
        if segments_root.exists():
            for child in segments_root.iterdir():
                if child.is_dir():
                    try:
                        segment_ensembles[child.name] = RankerEnsembleModel.load(child)
                    except Exception:
                        continue
        promoted_segments = [
            str(item)
            for item in (payload.get("promoted_segments") or [])
            if isinstance(item, str)
        ]
        return cls(
            objective=objective or global_ensemble.objective,
            global_ensemble=global_ensemble,
            segment_ensembles=segment_ensembles,
            promoted_segments=promoted_segments,
            std_ref=float(payload.get("std_ref") or 0.15),
            creator_cold_threshold=int(payload.get("creator_cold_threshold") or 10),
        )
