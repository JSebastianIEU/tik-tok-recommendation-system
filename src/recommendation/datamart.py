from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from .contracts import (
    CONTRACT_VERSION,
    CanonicalComment,
    CanonicalDatasetBundle,
    CanonicalVideo,
    CanonicalVideoSnapshot,
    validate_as_of_time_policy,
    validate_contract_bundle,
    validate_feature_access_policy,
    validate_raw_dataset_jsonl_against_contract,
)


DATAMART_VERSION = "datamart.v1"
VALID_TRACKS = {"pre_publication", "post_publication"}
VALID_PAIR_OBJECTIVES = {"reach", "engagement", "conversion"}


@dataclass
class BuildTrainingDataMartConfig:
    track: str = "pre_publication"
    min_history_hours: int = 24
    label_window_hours: int = 72
    min_author_rows_for_baseline: int = 2
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    include_pair_rows: bool = True
    pair_objective: str = "engagement"
    pair_candidates_per_query: int = 8

    def __post_init__(self) -> None:
        if self.track not in VALID_TRACKS:
            raise ValueError(
                f"track must be one of {sorted(VALID_TRACKS)}; got '{self.track}'"
            )
        if self.pair_objective not in VALID_PAIR_OBJECTIVES:
            raise ValueError(
                f"pair_objective must be one of {sorted(VALID_PAIR_OBJECTIVES)}; got '{self.pair_objective}'"
            )
        if self.min_history_hours < 1:
            raise ValueError("min_history_hours must be >= 1.")
        if self.label_window_hours < 1:
            raise ValueError("label_window_hours must be >= 1.")
        if self.min_author_rows_for_baseline < 1:
            raise ValueError("min_author_rows_for_baseline must be >= 1.")
        if self.pair_candidates_per_query < 1:
            raise ValueError("pair_candidates_per_query must be >= 1.")
        if self.train_ratio <= 0 or self.train_ratio >= 1:
            raise ValueError("train_ratio must be between 0 and 1.")
        if self.validation_ratio < 0 or self.validation_ratio >= 1:
            raise ValueError("validation_ratio must be between 0 and 1.")
        if self.train_ratio + self.validation_ratio >= 1:
            raise ValueError("train_ratio + validation_ratio must be < 1.")


class PreMetrics(BaseModel):
    views: Optional[int] = Field(default=None, ge=0)
    likes: Optional[int] = Field(default=None, ge=0)
    comments_count: Optional[int] = Field(default=None, ge=0)
    shares: Optional[int] = Field(default=None, ge=0)
    has_snapshot: bool


class CommentFeatures(BaseModel):
    available: bool
    count_pre: Optional[int] = Field(default=None, ge=0)
    avg_length_pre: Optional[float] = Field(default=None, ge=0.0)
    question_ratio_pre: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class TrainingFeatures(BaseModel):
    caption_word_count: int = Field(ge=0)
    hashtag_count: int = Field(ge=0)
    keyword_count: int = Field(ge=0)
    duration_seconds: Optional[int] = Field(default=None, ge=0)
    language: Optional[str] = None
    pre_metrics: PreMetrics
    comment_features: CommentFeatures
    missingness_flags: List[str] = Field(default_factory=list)


class TrainingLabels(BaseModel):
    future_views: int = Field(ge=0)
    future_engagement_rate: float = Field(ge=0.0)
    future_shares_per_1k_views: float = Field(ge=0.0)
    future_reach_log_delta: float = Field(ge=0.0)
    author_expected_log_views: float
    residual_log_views: float
    window_hours_observed: float = Field(ge=0.0)


class TrainingTargetsZ(BaseModel):
    reach: float
    engagement: float
    conversion: float


class TrainingRow(BaseModel):
    row_id: str
    video_id: str
    author_id: str
    topic_key: str
    posted_at: datetime
    as_of_time: datetime
    split: Literal["train", "validation", "test"]
    track: Literal["pre_publication", "post_publication"]
    features: TrainingFeatures
    labels: TrainingLabels
    targets_z: TrainingTargetsZ


class PairTrainingRow(BaseModel):
    pair_id: str
    query_row_id: str
    query_video_id: str
    candidate_row_id: str
    candidate_video_id: str
    query_as_of_time: datetime
    candidate_as_of_time: datetime
    similarity: float = Field(ge=0.0, le=1.0)
    objective: Literal["reach", "engagement", "conversion"]
    objective_score: float
    relevance_label: int = Field(ge=0, le=3)


class ExcludedVideoRecord(BaseModel):
    video_id: str
    reason: str
    detail: str


class TrainingDataMartCutoffs(BaseModel):
    train_end_as_of: Optional[datetime] = None
    validation_end_as_of: Optional[datetime] = None


class TrainingDataMartStats(BaseModel):
    rows_total: int = Field(ge=0)
    rows_censored: int = Field(ge=0)
    train_count: int = Field(ge=0)
    validation_count: int = Field(ge=0)
    test_count: int = Field(ge=0)
    pair_rows_total: int = Field(ge=0)
    excluded_by_reason: Dict[str, int] = Field(default_factory=dict)
    cutoffs: TrainingDataMartCutoffs


class TrainingDataMartConfigPayload(BaseModel):
    track: Literal["pre_publication", "post_publication"]
    min_history_hours: int = Field(ge=1)
    label_window_hours: int = Field(ge=1)
    min_author_rows_for_baseline: int = Field(ge=1)
    train_ratio: float
    validation_ratio: float
    include_pair_rows: bool
    pair_objective: Literal["reach", "engagement", "conversion"]
    pair_candidates_per_query: int = Field(ge=1)


class TrainingDataMart(BaseModel):
    version: Literal["datamart.v1"]
    generated_at: datetime
    source_contract_version: Literal["contract.v1"]
    config: TrainingDataMartConfigPayload
    stats: TrainingDataMartStats
    rows: List[TrainingRow]
    pair_rows: List[PairTrainingRow]
    excluded_video_records: List[ExcludedVideoRecord]
    excluded_video_ids: List[str]
    warnings: List[str] = Field(default_factory=list)


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    variance = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


def _fit_stats(values: Sequence[float]) -> Tuple[float, float]:
    return _mean(values), _std(values)


def _zscore_from_stats(value: float, mean: float, std: float) -> float:
    if std == 0:
        return 0.0
    return (value - mean) / std


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _tokenize(value: str) -> List[str]:
    normalized = (
        value.lower()
        .replace("#", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace("/", " ")
        .replace("-", " ")
        .replace("?", " ")
        .replace("!", " ")
    )
    return [token for token in normalized.split() if len(token) >= 2]


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    inter = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return 0.0 if union == 0 else inter / union


def _infer_topic(video: CanonicalVideo) -> str:
    if video.search_query and video.search_query.strip():
        return video.search_query.strip().lower()
    if video.hashtags:
        return video.hashtags[0].replace("#", "").strip().lower() or "general"
    tokens = _tokenize(video.caption)
    return tokens[0] if tokens else "general"


def _latest_snapshot_on_or_before(
    snapshots: Sequence[CanonicalVideoSnapshot],
    as_of: datetime,
) -> Optional[CanonicalVideoSnapshot]:
    out: Optional[CanonicalVideoSnapshot] = None
    for snapshot in snapshots:
        ts = _to_utc(snapshot.scraped_at)
        if ts <= as_of and (out is None or ts > _to_utc(out.scraped_at)):
            out = snapshot
    return out


def _earliest_snapshot_on_or_after(
    snapshots: Sequence[CanonicalVideoSnapshot],
    target: datetime,
) -> Optional[CanonicalVideoSnapshot]:
    out: Optional[CanonicalVideoSnapshot] = None
    for snapshot in snapshots:
        ts = _to_utc(snapshot.scraped_at)
        if ts >= target and (out is None or ts < _to_utc(out.scraped_at)):
            out = snapshot
    return out


def _comments_on_or_before(
    comments: Sequence[CanonicalComment],
    as_of: datetime,
) -> List[CanonicalComment]:
    return [comment for comment in comments if _to_utc(comment.created_at) <= as_of]


def _split_rows_by_time(
    rows: List[Dict[str, Any]],
    train_ratio: float,
    validation_ratio: float,
) -> None:
    ordered = sorted(rows, key=lambda row: row["as_of_time"])
    if not ordered:
        return

    n = len(ordered)
    train_count = int(math.floor(n * train_ratio))
    validation_count = int(math.floor(n * validation_ratio))

    if n >= 3:
        train_count = min(max(train_count, 1), n - 2)
        validation_count = min(max(validation_count, 1), n - train_count - 1)
    else:
        train_count = max(1, n - 1)
        validation_count = 0

    for idx, row in enumerate(ordered):
        if idx < train_count:
            row["split"] = "train"
        elif idx < train_count + validation_count:
            row["split"] = "validation"
        else:
            row["split"] = "test"

    split_by_id = {row["row_id"]: row["split"] for row in ordered}
    for row in rows:
        row["split"] = split_by_id.get(row["row_id"], "test")


def _objective_score(row: Dict[str, Any], objective: str) -> float:
    return row["targets_z"][objective]


def _relevance_label(score: float) -> int:
    if score >= 1.0:
        return 3
    if score >= 0.3:
        return 2
    if score >= -0.3:
        return 1
    return 0


def _build_pair_rows(
    rows: Sequence[Dict[str, Any]],
    objective: str,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    pair_rows: List[Dict[str, Any]] = []
    row_tokens: Dict[str, List[str]] = {}
    for row in rows:
        row_tokens[row["row_id"]] = _dedupe(
            [
                *_tokenize(row["topic_key"]),
                *_tokenize(row["video_id"]),
                *row["features"]["missingness_flags"],
            ]
        )

    for query in rows:
        query_as_of = query["as_of_time"]
        query_tokens = row_tokens.get(query["row_id"], [])
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        for candidate in rows:
            if candidate["row_id"] == query["row_id"]:
                continue
            if candidate["as_of_time"] >= query_as_of:
                continue
            similarity = _jaccard(query_tokens, row_tokens.get(candidate["row_id"], []))
            candidates.append((similarity, candidate))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for similarity, candidate in candidates[:max_candidates]:
            score = _objective_score(candidate, objective)
            pair_rows.append(
                {
                    "pair_id": f"{query['row_id']}::{candidate['row_id']}",
                    "query_row_id": query["row_id"],
                    "query_video_id": query["video_id"],
                    "candidate_row_id": candidate["row_id"],
                    "candidate_video_id": candidate["video_id"],
                    "query_as_of_time": query["as_of_time"],
                    "candidate_as_of_time": candidate["as_of_time"],
                    "similarity": round(similarity, 6),
                    "objective": objective,
                    "objective_score": round(score, 6),
                    "relevance_label": _relevance_label(score),
                }
            )
    return pair_rows


def _append_exclusion(
    excluded_video_records: List[Dict[str, str]],
    video_id: str,
    reason: str,
    detail: str,
) -> None:
    excluded_video_records.append(
        {
            "video_id": video_id,
            "reason": reason,
            "detail": detail,
        }
    )


def build_training_data_mart(
    bundle: CanonicalDatasetBundle,
    config: Optional[BuildTrainingDataMartConfig] = None,
) -> Dict[str, Any]:
    cfg = config or BuildTrainingDataMartConfig()

    contract_errors = validate_contract_bundle(bundle)
    if contract_errors:
        raise ValueError(
            f"Bundle failed contract validation: {' | '.join(contract_errors[:8])}"
        )
    temporal_errors = validate_as_of_time_policy(bundle, as_of_time=_to_utc(bundle.generated_at))
    if temporal_errors:
        raise ValueError(
            f"Bundle failed point-in-time policy: {' | '.join(temporal_errors[:8])}"
        )
    required_features = (
        ["video_metadata", "video_snapshots", "author_profile", "comments", "comment_snapshots"]
        if cfg.track == "post_publication"
        else ["video_metadata", "video_snapshots", "author_profile"]
    )
    feature_errors = validate_feature_access_policy(cfg.track, required_features)
    if feature_errors:
        raise ValueError(f"Feature access policy violation: {' | '.join(feature_errors)}")

    snapshots_by_video: Dict[str, List[CanonicalVideoSnapshot]] = {}
    for snapshot in bundle.video_snapshots:
        snapshots_by_video.setdefault(snapshot.video_id, []).append(snapshot)
    for items in snapshots_by_video.values():
        items.sort(key=lambda row: _to_utc(row.scraped_at))

    comments_by_video: Dict[str, List[CanonicalComment]] = {}
    for comment in bundle.comments:
        comments_by_video.setdefault(comment.video_id, []).append(comment)
    for items in comments_by_video.values():
        items.sort(key=lambda row: _to_utc(row.created_at))

    generated_at = _to_utc(bundle.generated_at)
    excluded_video_records: List[Dict[str, str]] = []
    temp_rows: List[Dict[str, Any]] = []

    for video in bundle.videos:
        posted_at = _to_utc(video.posted_at)
        as_of = posted_at + timedelta(hours=cfg.min_history_hours)
        label_target = as_of + timedelta(hours=cfg.label_window_hours)

        if as_of >= generated_at:
            _append_exclusion(
                excluded_video_records,
                video_id=video.video_id,
                reason="insufficient_history_window",
                detail=f"as_of_time={as_of.isoformat()} >= generated_at={generated_at.isoformat()}",
            )
            continue
        if label_target > generated_at:
            _append_exclusion(
                excluded_video_records,
                video_id=video.video_id,
                reason="label_horizon_not_available",
                detail=f"label_target={label_target.isoformat()} > generated_at={generated_at.isoformat()}",
            )
            continue

        snapshots = snapshots_by_video.get(video.video_id, [])
        pre_snapshot = _latest_snapshot_on_or_before(snapshots, as_of)
        future_snapshot = _earliest_snapshot_on_or_after(snapshots, label_target)
        if future_snapshot is None:
            max_snapshot_time = (
                _to_utc(snapshots[-1].scraped_at).isoformat() if snapshots else "none"
            )
            _append_exclusion(
                excluded_video_records,
                video_id=video.video_id,
                reason="missing_future_snapshot",
                detail=f"label_target={label_target.isoformat()}, latest_snapshot={max_snapshot_time}",
            )
            continue

        pre_views = pre_snapshot.views if pre_snapshot else 0
        future_views = future_snapshot.views
        future_likes = future_snapshot.likes
        future_comments = future_snapshot.comments_count
        future_shares = future_snapshot.shares

        reach_delta = max(0, future_views - pre_views)
        engagement_rate = _safe_div(
            future_likes + future_comments + future_shares,
            max(1, future_views),
        )
        shares_per_1k = _safe_div(future_shares * 1000, max(1, future_views))

        missingness_flags: List[str] = []
        if video.duration_seconds is None:
            missingness_flags.append("missing_duration_seconds")
        if not video.language:
            missingness_flags.append("missing_language")
        if pre_snapshot is None:
            missingness_flags.append("missing_pre_snapshot")

        comment_features = {
            "available": False,
            "count_pre": None,
            "avg_length_pre": None,
            "question_ratio_pre": None,
        }
        if cfg.track == "post_publication":
            eligible_comments = _comments_on_or_before(
                comments_by_video.get(video.video_id, []),
                as_of=as_of,
            )
            count = len(eligible_comments)
            avg_length = (
                0.0
                if count == 0
                else sum(len(_tokenize(comment.text)) for comment in eligible_comments) / count
            )
            question_ratio = (
                0.0
                if count == 0
                else sum(1 for comment in eligible_comments if "?" in comment.text) / count
            )
            comment_features = {
                "available": True,
                "count_pre": count,
                "avg_length_pre": round(avg_length, 6),
                "question_ratio_pre": round(question_ratio, 6),
            }

        temp_rows.append(
            {
                "row_id": f"{video.video_id}::{as_of.isoformat()}",
                "video_id": video.video_id,
                "author_id": video.author_id,
                "topic_key": _infer_topic(video),
                "posted_at": posted_at,
                "as_of_time": as_of,
                "split": "test",
                "raw_targets": {
                    "reach": math.log1p(reach_delta),
                    "engagement": engagement_rate,
                    "conversion": shares_per_1k,
                    "log_views": math.log1p(future_views),
                },
                "labels_base": {
                    "future_views": future_views,
                    "future_engagement_rate": round(engagement_rate, 6),
                    "future_shares_per_1k_views": round(shares_per_1k, 6),
                    "future_reach_log_delta": round(math.log1p(reach_delta), 6),
                    "window_hours_observed": round(
                        (_to_utc(future_snapshot.scraped_at) - as_of).total_seconds() / 3600.0,
                        4,
                    ),
                },
                "features": {
                    "caption_word_count": len(_tokenize(video.caption)),
                    "hashtag_count": len(video.hashtags),
                    "keyword_count": len(video.keywords),
                    "duration_seconds": video.duration_seconds,
                    "language": video.language,
                    "pre_metrics": {
                        "views": pre_snapshot.views if pre_snapshot else None,
                        "likes": pre_snapshot.likes if pre_snapshot else None,
                        "comments_count": pre_snapshot.comments_count if pre_snapshot else None,
                        "shares": pre_snapshot.shares if pre_snapshot else None,
                        "has_snapshot": pre_snapshot is not None,
                    },
                    "comment_features": comment_features,
                    "missingness_flags": missingness_flags,
                },
            }
        )

    _split_rows_by_time(
        temp_rows, train_ratio=cfg.train_ratio, validation_ratio=cfg.validation_ratio
    )
    train_rows_raw = [row for row in temp_rows if row["split"] == "train"]

    train_reach_values = [row["raw_targets"]["reach"] for row in train_rows_raw]
    train_engagement_values = [row["raw_targets"]["engagement"] for row in train_rows_raw]
    train_conversion_values = [row["raw_targets"]["conversion"] for row in train_rows_raw]
    train_log_views_values = [row["raw_targets"]["log_views"] for row in train_rows_raw]

    reach_mean, reach_std = _fit_stats(train_reach_values)
    engagement_mean, engagement_std = _fit_stats(train_engagement_values)
    conversion_mean, conversion_std = _fit_stats(train_conversion_values)
    global_train_log_views_mean = _mean(train_log_views_values)

    author_train_log_views: Dict[str, List[float]] = {}
    for row in train_rows_raw:
        author_train_log_views.setdefault(row["author_id"], []).append(
            row["raw_targets"]["log_views"]
        )

    rows: List[Dict[str, Any]] = []
    for row in temp_rows:
        split = row["split"]
        author_id = row["author_id"]
        raw_log_views = row["raw_targets"]["log_views"]
        train_author_samples = author_train_log_views.get(author_id, [])

        if split == "train":
            if (
                len(train_author_samples) >= cfg.min_author_rows_for_baseline
                and len(train_author_samples) > 1
            ):
                author_expected = (sum(train_author_samples) - raw_log_views) / (
                    len(train_author_samples) - 1
                )
            elif len(train_author_samples) >= cfg.min_author_rows_for_baseline:
                author_expected = global_train_log_views_mean
            else:
                author_expected = global_train_log_views_mean
        else:
            if len(train_author_samples) >= cfg.min_author_rows_for_baseline:
                author_expected = _mean(train_author_samples)
            else:
                author_expected = global_train_log_views_mean

        residual = raw_log_views - author_expected
        rows.append(
            {
                "row_id": row["row_id"],
                "video_id": row["video_id"],
                "author_id": author_id,
                "topic_key": row["topic_key"],
                "posted_at": row["posted_at"],
                "as_of_time": row["as_of_time"],
                "split": split,
                "track": cfg.track,
                "features": row["features"],
                "labels": {
                    **row["labels_base"],
                    "author_expected_log_views": round(author_expected, 6),
                    "residual_log_views": round(residual, 6),
                },
                "targets_z": {
                    "reach": round(
                        _zscore_from_stats(row["raw_targets"]["reach"], reach_mean, reach_std),
                        6,
                    ),
                    "engagement": round(
                        _zscore_from_stats(
                            row["raw_targets"]["engagement"],
                            engagement_mean,
                            engagement_std,
                        ),
                        6,
                    ),
                    "conversion": round(
                        _zscore_from_stats(
                            row["raw_targets"]["conversion"],
                            conversion_mean,
                            conversion_std,
                        ),
                        6,
                    ),
                },
            }
        )

    pair_rows = (
        _build_pair_rows(
            rows,
            objective=cfg.pair_objective,
            max_candidates=cfg.pair_candidates_per_query,
        )
        if cfg.include_pair_rows
        else []
    )

    sorted_rows = sorted(rows, key=lambda row: row["as_of_time"])
    train_rows = [row for row in sorted_rows if row["split"] == "train"]
    validation_rows = [row for row in sorted_rows if row["split"] == "validation"]
    test_rows = [row for row in sorted_rows if row["split"] == "test"]

    excluded_by_reason: Dict[str, int] = {}
    for item in excluded_video_records:
        excluded_by_reason[item["reason"]] = excluded_by_reason.get(item["reason"], 0) + 1

    seen_video_ids: set[str] = set()
    excluded_video_ids: List[str] = []
    for item in excluded_video_records:
        video_id = item["video_id"]
        if video_id in seen_video_ids:
            continue
        seen_video_ids.add(video_id)
        excluded_video_ids.append(video_id)

    payload = {
        "version": DATAMART_VERSION,
        "generated_at": datetime.now(timezone.utc),
        "source_contract_version": CONTRACT_VERSION,
        "config": {
            "track": cfg.track,
            "min_history_hours": cfg.min_history_hours,
            "label_window_hours": cfg.label_window_hours,
            "min_author_rows_for_baseline": cfg.min_author_rows_for_baseline,
            "train_ratio": cfg.train_ratio,
            "validation_ratio": cfg.validation_ratio,
            "include_pair_rows": cfg.include_pair_rows,
            "pair_objective": cfg.pair_objective,
            "pair_candidates_per_query": cfg.pair_candidates_per_query,
        },
        "stats": {
            "rows_total": len(rows),
            "rows_censored": len(excluded_video_records),
            "train_count": len(train_rows),
            "validation_count": len(validation_rows),
            "test_count": len(test_rows),
            "pair_rows_total": len(pair_rows),
            "excluded_by_reason": excluded_by_reason,
            "cutoffs": {
                "train_end_as_of": train_rows[-1]["as_of_time"] if train_rows else None,
                "validation_end_as_of": validation_rows[-1]["as_of_time"]
                if validation_rows
                else None,
            },
        },
        "rows": rows,
        "pair_rows": pair_rows,
        "excluded_video_records": excluded_video_records,
        "excluded_video_ids": excluded_video_ids,
        "warnings": [],
    }
    validated = TrainingDataMart.model_validate(payload)
    return validated.model_dump(mode="python")


def build_training_data_mart_from_jsonl(
    raw_jsonl: str,
    as_of_time: datetime,
    source: str = "training_datamart_jsonl",
    config: Optional[BuildTrainingDataMartConfig] = None,
    strict_timestamps: bool = False,
    fail_on_warnings: bool = False,
) -> Dict[str, Any]:
    validated = validate_raw_dataset_jsonl_against_contract(
        raw_jsonl=raw_jsonl,
        as_of_time=as_of_time,
        source=source,
        strict_timestamps=strict_timestamps,
    )
    if not validated.ok or validated.bundle is None:
        raise ValueError(
            f"Cannot build training data mart: dataset failed contract validation ({len(validated.errors)} errors)"
        )
    if fail_on_warnings and validated.warnings:
        raise ValueError(
            f"Cannot build training data mart: dataset has {len(validated.warnings)} warning(s) and fail_on_warnings=True"
        )

    mart = build_training_data_mart(validated.bundle, config=config)
    if validated.warnings:
        mart["warnings"].extend(validated.warnings)
    validated_mart = TrainingDataMart.model_validate(mart)
    return validated_mart.model_dump(mode="python")


__all__ = [
    "DATAMART_VERSION",
    "BuildTrainingDataMartConfig",
    "TrainingRow",
    "PairTrainingRow",
    "TrainingDataMart",
    "build_training_data_mart",
    "build_training_data_mart_from_jsonl",
]
