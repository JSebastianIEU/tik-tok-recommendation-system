from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from .contracts import (
    CONTRACT_VERSION,
    LEGACY_CONTRACT_VERSION,
    CanonicalComment,
    CanonicalDatasetBundle,
    CanonicalVideo,
    CanonicalVideoSnapshot,
    build_contract_manifest,
    load_bundle_from_manifest,
    validate_as_of_time_policy,
    validate_contract_bundle,
    validate_feature_access_policy,
    validate_raw_dataset_jsonl_against_contract,
)
from .comment_intelligence import (
    COMMENT_INTELLIGENCE_VERSION,
    load_comment_intelligence_snapshot_manifest,
    load_comment_transfer_priors_manifest,
)


DATAMART_VERSION = "datamart.v1"
VALID_TRACKS = {"pre_publication", "post_publication"}
VALID_PAIR_OBJECTIVES = {"reach", "engagement", "conversion"}
VALID_PAIR_TARGET_SOURCES = {"scalar_v1", "trajectory_v2_composite"}
DEFAULT_TRAJECTORY_WINDOWS_HOURS = (6, 24, 96)
TRAJECTORY_COMPONENT_KEYS = (
    "early_velocity",
    "core_velocity",
    "late_lift",
    "stability",
)
DEFAULT_TRAJECTORY_WEIGHTS = {
    objective: {"early": 0.45, "stability": 0.20, "late": 0.35}
    for objective in VALID_PAIR_OBJECTIVES
}


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
    pair_target_source: str = "scalar_v1"
    pair_candidates_per_query: int = 8
    as_of_run_time: Optional[datetime] = None
    enable_trajectory_labels: bool = True
    trajectory_windows_hours: Tuple[int, int, int] = DEFAULT_TRAJECTORY_WINDOWS_HOURS
    trajectory_objective_weights: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            objective: dict(weights)
            for objective, weights in DEFAULT_TRAJECTORY_WEIGHTS.items()
        }
    )
    source_manifest_id: Optional[str] = None
    source_manifest_path: Optional[str] = None
    comment_feature_manifest_id: Optional[str] = None
    comment_feature_manifest_path: Optional[str] = None
    comment_priors_manifest_id: Optional[str] = None
    comment_priors_manifest_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.track not in VALID_TRACKS:
            raise ValueError(
                f"track must be one of {sorted(VALID_TRACKS)}; got '{self.track}'"
            )
        if self.pair_objective not in VALID_PAIR_OBJECTIVES:
            raise ValueError(
                f"pair_objective must be one of {sorted(VALID_PAIR_OBJECTIVES)}; got '{self.pair_objective}'"
            )
        if self.pair_target_source not in VALID_PAIR_TARGET_SOURCES:
            raise ValueError(
                "pair_target_source must be one of "
                f"{sorted(VALID_PAIR_TARGET_SOURCES)}; got '{self.pair_target_source}'"
            )
        if self.min_history_hours < 1:
            raise ValueError("min_history_hours must be >= 1.")
        if self.label_window_hours < 1:
            raise ValueError("label_window_hours must be >= 1.")
        if self.min_author_rows_for_baseline < 1:
            raise ValueError("min_author_rows_for_baseline must be >= 1.")
        if self.pair_candidates_per_query < 1:
            raise ValueError("pair_candidates_per_query must be >= 1.")
        if len(self.trajectory_windows_hours) != 3:
            raise ValueError("trajectory_windows_hours must contain exactly 3 values.")
        trajectory_hours = tuple(int(value) for value in self.trajectory_windows_hours)
        if any(value < 1 for value in trajectory_hours):
            raise ValueError("trajectory_windows_hours values must all be >= 1.")
        if tuple(sorted(trajectory_hours)) != trajectory_hours:
            raise ValueError("trajectory_windows_hours must be strictly increasing.")
        self.trajectory_windows_hours = trajectory_hours
        if self.pair_target_source == "trajectory_v2_composite" and not self.enable_trajectory_labels:
            raise ValueError(
                "pair_target_source='trajectory_v2_composite' requires enable_trajectory_labels=True."
            )
        if self.as_of_run_time is not None:
            self.as_of_run_time = _to_utc(self.as_of_run_time)
        normalized_weights: Dict[str, Dict[str, float]] = {}
        for objective in VALID_PAIR_OBJECTIVES:
            raw = dict(self.trajectory_objective_weights.get(objective) or {})
            early = float(raw.get("early", DEFAULT_TRAJECTORY_WEIGHTS[objective]["early"]))
            stability = float(
                raw.get("stability", DEFAULT_TRAJECTORY_WEIGHTS[objective]["stability"])
            )
            late = float(raw.get("late", DEFAULT_TRAJECTORY_WEIGHTS[objective]["late"]))
            if early < 0 or stability < 0 or late < 0:
                raise ValueError(
                    f"trajectory_objective_weights[{objective}] values must be >= 0."
                )
            if (early + stability + late) == 0:
                raise ValueError(
                    f"trajectory_objective_weights[{objective}] must not sum to zero."
                )
            normalized_weights[objective] = {
                "early": early,
                "stability": stability,
                "late": late,
            }
        self.trajectory_objective_weights = normalized_weights
        if self.train_ratio <= 0 or self.train_ratio >= 1:
            raise ValueError("train_ratio must be between 0 and 1.")
        if self.validation_ratio < 0 or self.validation_ratio >= 1:
            raise ValueError("validation_ratio must be between 0 and 1.")
        if self.train_ratio + self.validation_ratio >= 1:
            raise ValueError("train_ratio + validation_ratio must be < 1.")


class ObjectiveTrajectoryLabels(BaseModel):
    series: Dict[str, Optional[float]]
    components: Dict[str, Optional[float]]
    observed_hours: Dict[str, Optional[float]]


class ObjectiveTrajectoryTargetsZ(BaseModel):
    components_z: Dict[str, Optional[float]]
    composite_z: Optional[float] = None


class ObjectiveTargetAvailability(BaseModel):
    objective_available: bool
    components: Dict[str, bool]


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
    comment_intelligence: Dict[str, Any] = Field(default_factory=dict)
    trajectory_features: Dict[str, Any] = Field(default_factory=dict)
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
    caption: str = ""
    hashtags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    search_query: Optional[str] = None
    topic_key: str
    language: Optional[str] = None
    locale: Optional[str] = None
    content_type: str = "other"
    posted_at: datetime
    as_of_time: datetime
    split: Literal["train", "validation", "test"]
    track: Literal["pre_publication", "post_publication"]
    features: TrainingFeatures
    labels: TrainingLabels
    targets_z: TrainingTargetsZ
    labels_trajectory: Dict[str, ObjectiveTrajectoryLabels] = Field(default_factory=dict)
    targets_trajectory_z: Dict[str, ObjectiveTrajectoryTargetsZ] = Field(
        default_factory=dict
    )
    target_availability: Dict[str, ObjectiveTargetAvailability] = Field(
        default_factory=dict
    )
    target_unavailable_reasons: Dict[str, List[str]] = Field(default_factory=dict)
    target_unavailable_component_reasons: Dict[str, Dict[str, List[str]]] = Field(
        default_factory=dict
    )


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
    target_source: Literal["scalar_v1", "trajectory_v2_composite"] = "scalar_v1"
    objective_score: float
    objective_score_components: Optional[Dict[str, float]] = None
    availability_mask: Optional[Dict[str, bool]] = None
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
    pair_rows_dropped_unavailable_target: int = Field(default=0, ge=0)
    pair_rows_dropped_by_reason: Dict[str, int] = Field(default_factory=dict)
    excluded_by_reason: Dict[str, int] = Field(default_factory=dict)
    availability_by_objective: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    trajectory_unavailable_by_reason: Dict[str, int] = Field(default_factory=dict)
    trajectory_unavailable_by_component: Dict[str, Dict[str, int]] = Field(
        default_factory=dict
    )
    censorship_rates_by_split: Dict[str, Dict[str, float]] = Field(default_factory=dict)
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
    pair_target_source: Literal["scalar_v1", "trajectory_v2_composite"]
    pair_candidates_per_query: int = Field(ge=1)
    as_of_run_time: Optional[datetime] = None
    enable_trajectory_labels: bool = True
    trajectory_windows_hours: Tuple[int, int, int]
    trajectory_objective_weights: Dict[str, Dict[str, float]]
    comment_feature_manifest_id: Optional[str] = None
    comment_feature_manifest_path: Optional[str] = None
    comment_priors_manifest_id: Optional[str] = None
    comment_priors_manifest_path: Optional[str] = None


class TrainingDataMart(BaseModel):
    version: Literal["datamart.v1"]
    generated_at: datetime
    source_contract_version: Literal["contract.v1", "contract.v2"]
    source_manifest_id: Optional[str] = None
    source_manifest_path: Optional[str] = None
    comment_feature_manifest_id: Optional[str] = None
    comment_feature_manifest_path: Optional[str] = None
    comment_priors_manifest_id: Optional[str] = None
    comment_priors_manifest_path: Optional[str] = None
    comment_intelligence_version: str = COMMENT_INTELLIGENCE_VERSION
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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _infer_content_type_bucket(video: CanonicalVideo) -> str:
    text = f"{video.caption} {' '.join(video.hashtags)}".lower()
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


def _resolve_manifest_path(manifest_ref: str) -> Path:
    path = Path(manifest_ref)
    if path.is_dir():
        path = path / "manifest.json"
    return path


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_comment_intelligence_lookup(
    manifest_path: Optional[str],
) -> tuple[Optional[str], Dict[str, List[Dict[str, Any]]]]:
    if not manifest_path:
        return None, {}
    payload, rows = load_comment_intelligence_snapshot_manifest(_resolve_manifest_path(manifest_path))
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        video_id = str(row.get("video_id") or "")
        if not video_id:
            continue
        out.setdefault(video_id, []).append(row)
    for items in out.values():
        items.sort(key=lambda item: str(item.get("as_of_time") or ""))
    return str(payload.get("comment_feature_manifest_id") or ""), out


def _load_comment_priors_lookup(
    manifest_path: Optional[str],
) -> tuple[Optional[str], Dict[tuple[str, str, str], Dict[str, Any]], Optional[str]]:
    if not manifest_path:
        return None, {}, None
    payload, priors = load_comment_transfer_priors_manifest(_resolve_manifest_path(manifest_path))
    out: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for entry in priors.entries:
        key = (
            str(entry.topic_key or "general"),
            str(entry.content_type_bucket or "general"),
            str(entry.author_size_bucket or "micro"),
        )
        out[key] = entry.model_dump(mode="python")
    return (
        str(payload.get("comment_priors_manifest_id") or ""),
        out,
        str(payload.get("taxonomy_version") or ""),
    )


def _lookup_comment_intelligence_snapshot(
    rows: Sequence[Dict[str, Any]],
    as_of: datetime,
) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    out: Optional[Dict[str, Any]] = None
    out_time: Optional[datetime] = None
    for row in rows:
        raw = row.get("as_of_time")
        if not isinstance(raw, str):
            continue
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
        if parsed <= as_of and (out is None or out_time is None or parsed > out_time):
            out = row
            out_time = parsed
    return out


def _missing_comment_intelligence(reason: str, detail: str = "") -> Dict[str, Any]:
    return {
        "source": "missing",
        "available": False,
        "taxonomy_version": "",
        "dominant_intents": [],
        "confusion_index": 0.0,
        "help_seeking_index": 0.0,
        "sentiment_volatility": 0.0,
        "sentiment_shift_early_late": 0.0,
        "reply_depth_max": 0.0,
        "reply_branch_factor": 0.0,
        "reply_ratio": 0.0,
        "root_thread_concentration": 0.0,
        "alignment_score": 0.0,
        "value_prop_coverage": 0.0,
        "on_topic_ratio": 0.0,
        "artifact_drift_ratio": 0.0,
        "alignment_shift_early_late": 0.0,
        "alignment_confidence": 0.0,
        "alignment_method_version": "",
        "confidence": 0.0,
        "missingness_flags": [reason] + ([detail] if detail else []),
    }


def _latest_snapshot_on_or_before(
    snapshots: Sequence[CanonicalVideoSnapshot],
    as_of: datetime,
) -> Optional[CanonicalVideoSnapshot]:
    out: Optional[CanonicalVideoSnapshot] = None
    for snapshot in snapshots:
        if getattr(snapshot, "lateness_class", "on_time") == "late_out_of_watermark":
            continue
        event_time = _to_utc(snapshot.event_time or snapshot.scraped_at)
        ingest_time = _to_utc(snapshot.ingested_at or snapshot.scraped_at)
        if event_time <= as_of and ingest_time <= as_of and (
            out is None
            or event_time > _to_utc(out.event_time or out.scraped_at)
        ):
            out = snapshot
    return out


def _earliest_snapshot_on_or_after(
    snapshots: Sequence[CanonicalVideoSnapshot],
    target: datetime,
    ingest_cutoff: datetime,
) -> Optional[CanonicalVideoSnapshot]:
    out: Optional[CanonicalVideoSnapshot] = None
    for snapshot in snapshots:
        if getattr(snapshot, "lateness_class", "on_time") == "late_out_of_watermark":
            continue
        event_time = _to_utc(snapshot.event_time or snapshot.scraped_at)
        ingest_time = _to_utc(snapshot.ingested_at or snapshot.scraped_at)
        if (
            event_time >= target
            and ingest_time <= ingest_cutoff
            and (out is None or event_time < _to_utc(out.event_time or out.scraped_at))
        ):
            out = snapshot
    return out


def _snapshot_event_time(snapshot: CanonicalVideoSnapshot) -> datetime:
    return _to_utc(snapshot.event_time or snapshot.scraped_at)


def _snapshot_ingest_time(snapshot: CanonicalVideoSnapshot) -> datetime:
    return _to_utc(snapshot.ingested_at or snapshot.scraped_at)


def _resolve_boundary_snapshot(
    snapshots: Sequence[CanonicalVideoSnapshot],
    boundary_time: datetime,
    ingest_cutoff: datetime,
    missing_reason: str,
) -> Tuple[Optional[CanonicalVideoSnapshot], Optional[str]]:
    by_event = [
        snapshot
        for snapshot in snapshots
        if _snapshot_event_time(snapshot) <= boundary_time
    ]
    if not by_event:
        return None, missing_reason
    not_late = [
        snapshot
        for snapshot in by_event
        if getattr(snapshot, "lateness_class", "on_time") != "late_out_of_watermark"
    ]
    if not not_late:
        return None, "late_out_of_watermark"
    visible = [
        snapshot for snapshot in not_late if _snapshot_ingest_time(snapshot) <= ingest_cutoff
    ]
    if not visible:
        return None, "ingested_after_cutoff"
    visible.sort(key=lambda item: (_snapshot_event_time(item), _snapshot_ingest_time(item)))
    return visible[-1], None


def _trajectory_series_value(
    objective: str,
    snapshot: CanonicalVideoSnapshot,
) -> float:
    views = max(0, int(snapshot.views))
    likes = max(0, int(snapshot.likes))
    comments = max(0, int(snapshot.comments_count))
    shares = max(0, int(snapshot.shares))
    if objective == "reach":
        return math.log1p(views)
    if objective == "engagement":
        return _safe_div(likes + comments + shares, max(views, 1))
    if objective == "conversion":
        return _safe_div(shares * 1000, max(views, 1))
    raise ValueError(f"Unsupported objective '{objective}' for trajectory series.")


def _build_trajectory_labels(
    snapshots: Sequence[CanonicalVideoSnapshot],
    as_of: datetime,
    run_cutoff: datetime,
    windows_hours: Tuple[int, int, int],
) -> Dict[str, Any]:
    h6, h24, h96 = windows_hours
    boundaries = {
        "t0": as_of,
        "t6": as_of + timedelta(hours=h6),
        "t24": as_of + timedelta(hours=h24),
        "t96": as_of + timedelta(hours=h96),
    }
    missing_reasons = {
        "t0": "missing_anchor_snapshot",
        "t6": "missing_t6_snapshot",
        "t24": "missing_t24_snapshot",
        "t96": "missing_t96_snapshot",
    }

    boundary_snapshots: Dict[str, Optional[CanonicalVideoSnapshot]] = {}
    boundary_reasons: Dict[str, Optional[str]] = {}
    observed_hours: Dict[str, Optional[float]] = {}
    for key, boundary in boundaries.items():
        snapshot, reason = _resolve_boundary_snapshot(
            snapshots=snapshots,
            boundary_time=boundary,
            ingest_cutoff=run_cutoff,
            missing_reason=missing_reasons[key],
        )
        boundary_snapshots[key] = snapshot
        boundary_reasons[key] = reason
        observed_hours[key] = (
            round((_snapshot_event_time(snapshot) - as_of).total_seconds() / 3600.0, 4)
            if snapshot is not None
            else None
        )

    objective_labels: Dict[str, Dict[str, Any]] = {}
    objective_availability: Dict[str, Dict[str, Any]] = {}
    objective_reasons: Dict[str, List[str]] = {}
    objective_component_reasons: Dict[str, Dict[str, List[str]]] = {}
    component_dependencies = {
        "early_velocity": ("t0", "t6"),
        "core_velocity": ("t6", "t24"),
        "late_lift": ("t24", "t96"),
        "stability": ("t0", "t6", "t24", "t96"),
    }

    for objective in sorted(VALID_PAIR_OBJECTIVES):
        series: Dict[str, Optional[float]] = {}
        for key in ("t0", "t6", "t24", "t96"):
            snapshot = boundary_snapshots[key]
            series[key] = (
                round(_trajectory_series_value(objective, snapshot), 6)
                if snapshot is not None
                else None
            )

        early_velocity = (
            round((series["t6"] - series["t0"]) / float(h6), 6)
            if series["t6"] is not None and series["t0"] is not None
            else None
        )
        core_velocity = (
            round((series["t24"] - series["t6"]) / float(h24 - h6), 6)
            if series["t24"] is not None and series["t6"] is not None
            else None
        )
        late_lift = (
            round(series["t96"] - series["t24"], 6)
            if series["t96"] is not None and series["t24"] is not None
            else None
        )
        late_velocity = (
            round((series["t96"] - series["t24"]) / float(h96 - h24), 6)
            if series["t96"] is not None and series["t24"] is not None
            else None
        )
        stability = (
            round(-_std([early_velocity, core_velocity, late_velocity]), 6)
            if early_velocity is not None
            and core_velocity is not None
            and late_velocity is not None
            else None
        )

        components = {
            "early_velocity": early_velocity,
            "core_velocity": core_velocity,
            "late_lift": late_lift,
            "stability": stability,
        }
        component_available = {
            "early_velocity": early_velocity is not None,
            "core_velocity": core_velocity is not None,
            "late_lift": late_lift is not None,
            "stability": stability is not None,
        }
        component_reasons: Dict[str, List[str]] = {}
        for component_key, boundary_keys in component_dependencies.items():
            reasons_for_component = sorted(
                {
                    str(boundary_reasons[boundary_key])
                    for boundary_key in boundary_keys
                    if boundary_reasons.get(boundary_key) is not None
                }
            )
            component_reasons[component_key] = (
                reasons_for_component if not component_available[component_key] else []
            )
        objective_available = (
            component_available["early_velocity"]
            and component_available["late_lift"]
            and component_available["stability"]
        )
        reasons = sorted(
            {
                reason
                for component_key in ("early_velocity", "late_lift", "stability")
                for reason in component_reasons.get(component_key, [])
            }
        )

        objective_labels[objective] = {
            "series": series,
            "components": components,
            "observed_hours": observed_hours,
        }
        objective_availability[objective] = {
            "objective_available": objective_available,
            "components": component_available,
        }
        objective_reasons[objective] = reasons
        objective_component_reasons[objective] = component_reasons

    return {
        "labels_trajectory": objective_labels,
        "target_availability": objective_availability,
        "target_unavailable_reasons": objective_reasons,
        "target_unavailable_component_reasons": objective_component_reasons,
    }


def _derive_trajectory_features_from_labels(
    *,
    labels_trajectory: Dict[str, Dict[str, Any]],
    target_availability: Dict[str, Dict[str, Any]],
    windows_hours: Tuple[int, int, int],
) -> Dict[str, Any]:
    h6, h24, h96 = windows_hours

    def _component(objective: str, key: str) -> float:
        payload = labels_trajectory.get(objective, {})
        components = payload.get("components", {}) if isinstance(payload, dict) else {}
        value = components.get(key) if isinstance(components, dict) else None
        return float(value) if value is not None else 0.0

    def _available(objective: str, key: Optional[str] = None) -> bool:
        payload = target_availability.get(objective, {})
        if not isinstance(payload, dict):
            return False
        if key is None:
            return bool(payload.get("objective_available", False))
        components = payload.get("components", {})
        if not isinstance(components, dict):
            return False
        return bool(components.get(key, False))

    reach_series = (
        labels_trajectory.get("reach", {}).get("series", {})
        if isinstance(labels_trajectory.get("reach"), dict)
        else {}
    )
    t0 = float(reach_series.get("t0") or 0.0) if isinstance(reach_series, dict) else 0.0
    t6 = float(reach_series.get("t6") or 0.0) if isinstance(reach_series, dict) else 0.0
    t24 = float(reach_series.get("t24") or 0.0) if isinstance(reach_series, dict) else 0.0
    t96 = float(reach_series.get("t96") or 0.0) if isinstance(reach_series, dict) else 0.0
    series_points = {0.0: t0, float(h6): t6, float(h24): t24, float(h96): t96}
    peak_lag_hours = max(series_points.items(), key=lambda item: item[1])[0]

    early = (
        _component("reach", "early_velocity")
        + _component("engagement", "early_velocity")
        + _component("conversion", "early_velocity")
    ) / 3.0
    core = (
        _component("reach", "core_velocity")
        + _component("engagement", "core_velocity")
        + _component("conversion", "core_velocity")
    ) / 3.0
    late = (
        _component("reach", "late_lift")
        + _component("engagement", "late_lift")
        + _component("conversion", "late_lift")
    ) / 3.0
    stability = (
        _component("reach", "stability")
        + _component("engagement", "stability")
        + _component("conversion", "stability")
    ) / 3.0
    late_velocity = late / float(max(1, h96 - h24))
    acceleration = core - early
    curvature = late_velocity - core
    durability_ratio = _safe_div(max(0.0, late), max(1e-6, abs(t24 - t0)))

    availability_flags = []
    for objective in sorted(VALID_PAIR_OBJECTIVES):
        for component in TRAJECTORY_COMPONENT_KEYS:
            availability_flags.append(1.0 if _available(objective, component) else 0.0)
    available_ratio = sum(availability_flags) / max(1.0, float(len(availability_flags)))
    missing_count = int(len(availability_flags) - int(sum(availability_flags)))

    spike_score = (1.4 * max(early, 0.0)) + (0.8 * max(-late, 0.0)) + (
        0.7 * max(0.0, 0.35 - durability_ratio)
    )
    durable_score = (1.3 * max(late, 0.0)) + (0.9 * max(durability_ratio, 0.0)) + (
        0.5 * max(stability, 0.0)
    )
    balanced_score = max(
        0.05,
        1.0 - abs(early - late) - abs(durability_ratio - 0.5) + (0.2 * max(stability, 0.0)),
    )
    logits = [spike_score, balanced_score, durable_score]
    max_logit = max(logits)
    exp = [math.exp(value - max_logit) for value in logits]
    denom = sum(exp) if sum(exp) > 0 else 1.0
    regime_probs = {
        "spike": round(exp[0] / denom, 6),
        "balanced": round(exp[1] / denom, 6),
        "durable": round(exp[2] / denom, 6),
    }
    regime_pred = max(regime_probs.items(), key=lambda item: item[1])[0]
    sorted_probs = sorted(regime_probs.values(), reverse=True)
    regime_confidence = (
        sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else sorted_probs[0]
    )

    return {
        "early_velocity": round(early, 6),
        "core_velocity": round(core, 6),
        "late_lift": round(late, 6),
        "stability": round(stability, 6),
        "late_velocity": round(late_velocity, 6),
        "acceleration_proxy": round(acceleration, 6),
        "curvature_proxy": round(curvature, 6),
        "durability_ratio": round(durability_ratio, 6),
        "peak_lag_hours": round(float(peak_lag_hours), 6),
        "available_ratio": round(float(available_ratio), 6),
        "missing_component_count": missing_count,
        "regime_pred": regime_pred,
        "regime_probabilities": regime_probs,
        "regime_confidence": round(float(regime_confidence), 6),
        "objectives": {
            objective: {
                "objective_available": _available(objective),
                "components": {
                    "early_velocity": round(_component(objective, "early_velocity"), 6),
                    "core_velocity": round(_component(objective, "core_velocity"), 6),
                    "late_lift": round(_component(objective, "late_lift"), 6),
                    "stability": round(_component(objective, "stability"), 6),
                },
            }
            for objective in sorted(VALID_PAIR_OBJECTIVES)
        },
    }


def _comments_on_or_before(
    comments: Sequence[CanonicalComment],
    as_of: datetime,
) -> List[CanonicalComment]:
    out: List[CanonicalComment] = []
    for comment in comments:
        created = _to_utc(comment.created_at)
        ingested = _to_utc(comment.ingested_at or comment.created_at)
        if created <= as_of and ingested <= as_of:
            out.append(comment)
    return out


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


def _objective_score(
    row: Dict[str, Any],
    objective: str,
    target_source: str,
) -> Optional[float]:
    if target_source == "scalar_v1":
        return float(row["targets_z"][objective])
    if target_source == "trajectory_v2_composite":
        trajectory_targets = row.get("targets_trajectory_z")
        if not isinstance(trajectory_targets, dict):
            return None
        objective_payload = trajectory_targets.get(objective)
        if not isinstance(objective_payload, dict):
            return None
        composite = objective_payload.get("composite_z")
        if composite is None:
            return None
        return float(composite)
    return None


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
    target_source: str,
    max_candidates: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    pair_rows: List[Dict[str, Any]] = []
    dropped_by_reason: Dict[str, int] = {}
    row_tokens: Dict[str, List[str]] = {}
    for row in rows:
        row_tokens[row["row_id"]] = _dedupe(
            [
                *_tokenize(str(row.get("caption") or "")),
                *[
                    tag.replace("#", "").strip().lower()
                    for tag in list(row.get("hashtags") or [])
                    if str(tag).strip()
                ],
                *_tokenize(" ".join(str(item) for item in list(row.get("keywords") or []))),
                *_tokenize(str(row.get("search_query") or "")),
                *_tokenize(row["topic_key"]),
            ]
        )

    for query in rows:
        query_as_of = query["as_of_time"]
        if target_source == "trajectory_v2_composite":
            query_available = (
                query.get("target_availability", {})
                if isinstance(query.get("target_availability"), dict)
                else {}
            ).get(objective, {})
            if not bool(query_available.get("objective_available", False)):
                dropped_by_reason["query_target_unavailable"] = (
                    dropped_by_reason.get("query_target_unavailable", 0) + 1
                )
                continue
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
            score = _objective_score(candidate, objective, target_source=target_source)
            if score is None:
                dropped_by_reason["candidate_target_unavailable"] = (
                    dropped_by_reason.get("candidate_target_unavailable", 0) + 1
                )
                continue
            objective_components: Optional[Dict[str, float]] = None
            availability_mask: Optional[Dict[str, bool]] = None
            if target_source == "trajectory_v2_composite":
                objective_payload = (
                    candidate.get("targets_trajectory_z", {})
                    if isinstance(candidate.get("targets_trajectory_z"), dict)
                    else {}
                ).get(objective, {})
                components_z = (
                    objective_payload.get("components_z", {})
                    if isinstance(objective_payload, dict)
                    else {}
                )
                objective_components = {
                    key: float(value)
                    for key, value in components_z.items()
                    if value is not None and key in {"early_velocity", "stability", "late_lift"}
                }
                availability_payload = (
                    candidate.get("target_availability", {})
                    if isinstance(candidate.get("target_availability"), dict)
                    else {}
                ).get(objective, {})
                component_mask = (
                    availability_payload.get("components", {})
                    if isinstance(availability_payload, dict)
                    else {}
                )
                availability_mask = {
                    "query_objective_available": True,
                    "candidate_objective_available": bool(
                        availability_payload.get("objective_available", False)
                    ),
                    "candidate_early_available": bool(
                        component_mask.get("early_velocity", False)
                    ),
                    "candidate_stability_available": bool(
                        component_mask.get("stability", False)
                    ),
                    "candidate_late_available": bool(
                        component_mask.get("late_lift", False)
                    ),
                }
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
                    "target_source": target_source,
                    "objective_score": round(score, 6),
                    "objective_score_components": objective_components,
                    "availability_mask": availability_mask,
                    "relevance_label": _relevance_label(score),
                }
            )
    return pair_rows, dropped_by_reason


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
    source_manifest_id: Optional[str] = None,
    source_manifest_path: Optional[str] = None,
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

    loaded_comment_feature_manifest_id, comment_rows_by_video = _load_comment_intelligence_lookup(
        cfg.comment_feature_manifest_path
    )
    loaded_comment_priors_manifest_id, comment_priors_lookup, priors_taxonomy_version = (
        _load_comment_priors_lookup(cfg.comment_priors_manifest_path)
    )

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
    authors_by_id = {item.author_id: item for item in bundle.authors}

    generated_at = _to_utc(bundle.generated_at)
    run_cutoff = _to_utc(cfg.as_of_run_time or bundle.generated_at)
    excluded_video_records: List[Dict[str, str]] = []
    temp_rows: List[Dict[str, Any]] = []

    for video in bundle.videos:
        posted_at = _to_utc(video.posted_at)
        as_of = posted_at + timedelta(hours=cfg.min_history_hours)
        label_target = as_of + timedelta(hours=cfg.label_window_hours)

        if as_of >= run_cutoff:
            _append_exclusion(
                excluded_video_records,
                video_id=video.video_id,
                reason="insufficient_history_window",
                detail=f"as_of_time={as_of.isoformat()} >= run_cutoff={run_cutoff.isoformat()}",
            )
            continue
        if label_target > run_cutoff:
            _append_exclusion(
                excluded_video_records,
                video_id=video.video_id,
                reason="label_horizon_not_available",
                detail=f"label_target={label_target.isoformat()} > run_cutoff={run_cutoff.isoformat()}",
            )
            continue

        snapshots = snapshots_by_video.get(video.video_id, [])
        pre_snapshot = _latest_snapshot_on_or_before(snapshots, as_of)
        future_snapshot = _earliest_snapshot_on_or_after(
            snapshots,
            label_target,
            ingest_cutoff=run_cutoff,
        )
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
        topic_key = _infer_topic(video)
        content_type_bucket = _infer_content_type_bucket(video)
        author_obj = authors_by_id.get(video.author_id)
        author_size_bucket = _author_size_bucket(
            int(author_obj.followers_count if author_obj else 0)
        )

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

        comment_intelligence = _missing_comment_intelligence(
            "comment_intelligence_unavailable"
        )
        if cfg.track == "post_publication":
            snapshot_row = _lookup_comment_intelligence_snapshot(
                comment_rows_by_video.get(video.video_id, []),
                as_of=as_of,
            )
            if snapshot_row is not None and isinstance(snapshot_row.get("features"), dict):
                snapshot_features = snapshot_row.get("features", {})
                comment_intelligence = {
                    "source": "video_snapshot",
                    "available": True,
                    "taxonomy_version": str(snapshot_row.get("taxonomy_version") or ""),
                    "dominant_intents": list(snapshot_features.get("dominant_intents") or []),
                    "confusion_index": float(snapshot_features.get("confusion_index") or 0.0),
                    "help_seeking_index": float(snapshot_features.get("help_seeking_index") or 0.0),
                    "sentiment_volatility": float(
                        snapshot_features.get("sentiment_volatility") or 0.0
                    ),
                    "sentiment_shift_early_late": float(
                        snapshot_features.get("sentiment_shift_early_late") or 0.0
                    ),
                    "reply_depth_max": float(snapshot_features.get("reply_depth_max") or 0.0),
                    "reply_branch_factor": float(
                        snapshot_features.get("reply_branch_factor") or 0.0
                    ),
                    "reply_ratio": float(snapshot_features.get("reply_ratio") or 0.0),
                    "root_thread_concentration": float(
                        snapshot_features.get("root_thread_concentration") or 0.0
                    ),
                    "alignment_score": float(snapshot_features.get("alignment_score") or 0.0),
                    "value_prop_coverage": float(
                        snapshot_features.get("value_prop_coverage") or 0.0
                    ),
                    "on_topic_ratio": float(snapshot_features.get("on_topic_ratio") or 0.0),
                    "artifact_drift_ratio": float(
                        snapshot_features.get("artifact_drift_ratio") or 0.0
                    ),
                    "alignment_shift_early_late": float(
                        snapshot_features.get("alignment_shift_early_late") or 0.0
                    ),
                    "alignment_confidence": float(
                        snapshot_features.get("alignment_confidence") or 0.0
                    ),
                    "alignment_method_version": str(
                        snapshot_features.get("alignment_method_version") or ""
                    ),
                    "confidence": float(snapshot_features.get("confidence") or 0.0),
                    "missingness_flags": sorted(
                        str(key)
                        for key in (snapshot_row.get("missingness") or {}).keys()
                    ),
                }
            else:
                comment_intelligence = _missing_comment_intelligence(
                    "missing_comment_snapshot_manifest_row"
                )
        else:
            prior_row = comment_priors_lookup.get(
                (topic_key, content_type_bucket, author_size_bucket)
            )
            if prior_row is not None:
                comment_intelligence = {
                    "source": "transfer_prior",
                    "available": True,
                    "taxonomy_version": str(priors_taxonomy_version or ""),
                    "dominant_intents": list(prior_row.get("dominant_intents") or []),
                    "confusion_index": float(prior_row.get("confusion_index") or 0.0),
                    "help_seeking_index": float(prior_row.get("help_seeking_index") or 0.0),
                    "sentiment_volatility": float(prior_row.get("sentiment_volatility") or 0.0),
                    "sentiment_shift_early_late": float(
                        prior_row.get("sentiment_shift_early_late") or 0.0
                    ),
                    "reply_depth_max": float(prior_row.get("reply_depth_max") or 0.0),
                    "reply_branch_factor": float(prior_row.get("reply_branch_factor") or 0.0),
                    "reply_ratio": float(prior_row.get("reply_ratio") or 0.0),
                    "root_thread_concentration": float(
                        prior_row.get("root_thread_concentration") or 0.0
                    ),
                    "alignment_score": float(prior_row.get("prior_alignment_score") or 0.0),
                    "value_prop_coverage": float(
                        prior_row.get("prior_value_prop_coverage") or 0.0
                    ),
                    "on_topic_ratio": 0.0,
                    "artifact_drift_ratio": float(
                        prior_row.get("prior_artifact_drift_ratio") or 0.0
                    ),
                    "alignment_shift_early_late": 0.0,
                    "alignment_confidence": float(
                        prior_row.get("prior_alignment_confidence") or 0.0
                    ),
                    "alignment_method_version": "prior_transfer",
                    "confidence": float(prior_row.get("confidence") or 0.0),
                    "missingness_flags": [],
                }
            else:
                comment_intelligence = _missing_comment_intelligence(
                    "missing_transfer_prior"
                )

        if not bool(comment_intelligence.get("available")):
            missingness_flags.extend(
                [
                    f"comment_intelligence:{flag}"
                    for flag in list(comment_intelligence.get("missingness_flags") or [])
                ]
            )

        trajectory_payload = _build_trajectory_labels(
            snapshots=snapshots,
            as_of=as_of,
            run_cutoff=run_cutoff,
            windows_hours=cfg.trajectory_windows_hours,
        )
        trajectory_features = _derive_trajectory_features_from_labels(
            labels_trajectory=trajectory_payload["labels_trajectory"],
            target_availability=trajectory_payload["target_availability"],
            windows_hours=cfg.trajectory_windows_hours,
        )

        temp_rows.append(
            {
                "row_id": f"{video.video_id}::{as_of.isoformat()}",
                "video_id": video.video_id,
                "author_id": video.author_id,
                "caption": video.caption,
                "hashtags": list(video.hashtags),
                "keywords": list(video.keywords),
                "search_query": video.search_query,
                "topic_key": topic_key,
                "language": video.language,
                "locale": None,
                "content_type": content_type_bucket,
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
                "labels_trajectory_raw": trajectory_payload["labels_trajectory"],
                "target_availability": trajectory_payload["target_availability"],
                "target_unavailable_reasons": trajectory_payload[
                    "target_unavailable_reasons"
                ],
                "target_unavailable_component_reasons": trajectory_payload[
                    "target_unavailable_component_reasons"
                ],
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
                    "comment_intelligence": comment_intelligence,
                    "trajectory_features": trajectory_features,
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
    trajectory_feature_keys = (
        "early_velocity",
        "core_velocity",
        "late_lift",
        "stability",
        "late_velocity",
        "acceleration_proxy",
        "curvature_proxy",
        "durability_ratio",
        "peak_lag_hours",
        "available_ratio",
        "missing_component_count",
        "regime_confidence",
    )
    trajectory_feature_stats: Dict[str, Tuple[float, float]] = {}
    for key in trajectory_feature_keys:
        values: List[float] = []
        for row in train_rows_raw:
            features = row.get("features", {})
            if not isinstance(features, dict):
                continue
            trajectory_features = features.get("trajectory_features", {})
            if not isinstance(trajectory_features, dict):
                continue
            if trajectory_features.get(key) is None:
                continue
            values.append(float(trajectory_features.get(key) or 0.0))
        trajectory_feature_stats[key] = _fit_stats(values)
    comment_alignment_feature_keys = (
        "alignment_score",
        "value_prop_coverage",
        "on_topic_ratio",
        "artifact_drift_ratio",
        "alignment_shift_early_late",
        "alignment_confidence",
    )
    comment_alignment_stats: Dict[str, Tuple[float, float]] = {}
    for key in comment_alignment_feature_keys:
        values: List[float] = []
        for row in train_rows_raw:
            features = row.get("features", {})
            if not isinstance(features, dict):
                continue
            comment_features = features.get("comment_intelligence", {})
            if not isinstance(comment_features, dict):
                continue
            values.append(float(comment_features.get(key) or 0.0))
        comment_alignment_stats[key] = _fit_stats(values)

    trajectory_component_stats: Dict[str, Dict[str, Tuple[float, float]]] = {
        objective: {} for objective in sorted(VALID_PAIR_OBJECTIVES)
    }
    trajectory_composite_stats: Dict[str, Tuple[float, float]] = {}

    if cfg.enable_trajectory_labels:
        for objective in sorted(VALID_PAIR_OBJECTIVES):
            for component_key in TRAJECTORY_COMPONENT_KEYS:
                train_values: List[float] = []
                for row in train_rows_raw:
                    objective_labels = (
                        row.get("labels_trajectory_raw", {})
                        if isinstance(row.get("labels_trajectory_raw"), dict)
                        else {}
                    ).get(objective, {})
                    components = (
                        objective_labels.get("components", {})
                        if isinstance(objective_labels, dict)
                        else {}
                    )
                    value = components.get(component_key) if isinstance(components, dict) else None
                    if value is not None:
                        train_values.append(float(value))
                trajectory_component_stats[objective][component_key] = _fit_stats(train_values)

            weights = cfg.trajectory_objective_weights.get(
                objective, DEFAULT_TRAJECTORY_WEIGHTS[objective]
            )
            train_composites: List[float] = []
            for row in train_rows_raw:
                objective_labels = (
                    row.get("labels_trajectory_raw", {})
                    if isinstance(row.get("labels_trajectory_raw"), dict)
                    else {}
                ).get(objective, {})
                components = (
                    objective_labels.get("components", {})
                    if isinstance(objective_labels, dict)
                    else {}
                )
                early_value = (
                    float(components.get("early_velocity"))
                    if isinstance(components, dict) and components.get("early_velocity") is not None
                    else None
                )
                stability_value = (
                    float(components.get("stability"))
                    if isinstance(components, dict) and components.get("stability") is not None
                    else None
                )
                late_value = (
                    float(components.get("late_lift"))
                    if isinstance(components, dict) and components.get("late_lift") is not None
                    else None
                )
                if (
                    early_value is None
                    or stability_value is None
                    or late_value is None
                ):
                    continue
                composite_raw = (
                    (weights["early"] * early_value)
                    + (weights["stability"] * stability_value)
                    + (weights["late"] * late_value)
                )
                train_composites.append(float(composite_raw))
            trajectory_composite_stats[objective] = _fit_stats(train_composites)

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
        labels_trajectory = (
            row.get("labels_trajectory_raw", {})
            if isinstance(row.get("labels_trajectory_raw"), dict)
            else {}
        )
        target_availability = (
            row.get("target_availability", {})
            if isinstance(row.get("target_availability"), dict)
            else {}
        )
        target_unavailable_reasons = (
            row.get("target_unavailable_reasons", {})
            if isinstance(row.get("target_unavailable_reasons"), dict)
            else {}
        )
        target_unavailable_component_reasons = (
            row.get("target_unavailable_component_reasons", {})
            if isinstance(row.get("target_unavailable_component_reasons"), dict)
            else {}
        )

        targets_trajectory_z: Dict[str, Dict[str, Any]] = {}
        if cfg.enable_trajectory_labels:
            for objective in sorted(VALID_PAIR_OBJECTIVES):
                objective_labels = labels_trajectory.get(objective, {})
                components = (
                    objective_labels.get("components", {})
                    if isinstance(objective_labels, dict)
                    else {}
                )
                objective_stats = trajectory_component_stats.get(objective, {})
                components_z: Dict[str, Optional[float]] = {}
                for component_key in TRAJECTORY_COMPONENT_KEYS:
                    value = (
                        components.get(component_key)
                        if isinstance(components, dict)
                        else None
                    )
                    if value is None:
                        components_z[component_key] = None
                        continue
                    mean, std = objective_stats.get(component_key, (0.0, 0.0))
                    components_z[component_key] = round(
                        _zscore_from_stats(float(value), mean, std),
                        6,
                    )

                weights = cfg.trajectory_objective_weights.get(
                    objective, DEFAULT_TRAJECTORY_WEIGHTS[objective]
                )
                early_value = (
                    float(components.get("early_velocity"))
                    if isinstance(components, dict) and components.get("early_velocity") is not None
                    else None
                )
                stability_value = (
                    float(components.get("stability"))
                    if isinstance(components, dict) and components.get("stability") is not None
                    else None
                )
                late_value = (
                    float(components.get("late_lift"))
                    if isinstance(components, dict) and components.get("late_lift") is not None
                    else None
                )
                if (
                    early_value is None
                    or stability_value is None
                    or late_value is None
                ):
                    composite_raw: Optional[float] = None
                    composite_z: Optional[float] = None
                else:
                    composite_raw = (
                        (weights["early"] * early_value)
                        + (weights["stability"] * stability_value)
                        + (weights["late"] * late_value)
                    )
                    comp_mean, comp_std = trajectory_composite_stats.get(objective, (0.0, 0.0))
                    composite_z = round(
                        _zscore_from_stats(float(composite_raw), comp_mean, comp_std),
                        6,
                    )
                targets_trajectory_z[objective] = {
                    "components_z": components_z,
                    "composite_z": composite_z,
                }
        features_payload = (
            dict(row["features"]) if isinstance(row.get("features"), dict) else {}
        )
        comment_intelligence_payload = (
            features_payload.get("comment_intelligence")
            if isinstance(features_payload.get("comment_intelligence"), dict)
            else {}
        )
        if isinstance(comment_intelligence_payload, dict):
            comment_z: Dict[str, float] = {}
            for key in comment_alignment_feature_keys:
                mean_v, std_v = comment_alignment_stats.get(key, (0.0, 0.0))
                value = float(comment_intelligence_payload.get(key) or 0.0)
                comment_z[key] = round(_zscore_from_stats(value, mean_v, std_v), 6)
            comment_intelligence_payload = {
                **comment_intelligence_payload,
                "z_features": comment_z,
            }
            features_payload["comment_intelligence"] = comment_intelligence_payload
        trajectory_features_payload = (
            features_payload.get("trajectory_features")
            if isinstance(features_payload.get("trajectory_features"), dict)
            else {}
        )
        if isinstance(trajectory_features_payload, dict):
            z_features: Dict[str, float] = {}
            for key in trajectory_feature_keys:
                mean_v, std_v = trajectory_feature_stats.get(key, (0.0, 0.0))
                value = float(trajectory_features_payload.get(key) or 0.0)
                z_features[key] = round(_zscore_from_stats(value, mean_v, std_v), 6)
            trajectory_features_payload = {
                **trajectory_features_payload,
                "z_features": z_features,
            }
            features_payload["trajectory_features"] = trajectory_features_payload
        rows.append(
            {
                "row_id": row["row_id"],
                "video_id": row["video_id"],
                "author_id": author_id,
                "caption": row.get("caption") or "",
                "hashtags": list(row.get("hashtags") or []),
                "keywords": list(row.get("keywords") or []),
                "search_query": row.get("search_query"),
                "topic_key": row["topic_key"],
                "language": row.get("language"),
                "locale": row.get("locale"),
                "content_type": row.get("content_type") or "other",
                "posted_at": row["posted_at"],
                "as_of_time": row["as_of_time"],
                "split": split,
                "track": cfg.track,
                "features": features_payload,
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
                "labels_trajectory": labels_trajectory if cfg.enable_trajectory_labels else {},
                "targets_trajectory_z": (
                    targets_trajectory_z if cfg.enable_trajectory_labels else {}
                ),
                "target_availability": (
                    target_availability if cfg.enable_trajectory_labels else {}
                ),
                "target_unavailable_reasons": (
                    target_unavailable_reasons if cfg.enable_trajectory_labels else {}
                ),
                "target_unavailable_component_reasons": (
                    target_unavailable_component_reasons
                    if cfg.enable_trajectory_labels
                    else {}
                ),
            }
        )

    pair_drop_stats: Dict[str, int] = {}
    if cfg.include_pair_rows:
        pair_rows, pair_drop_stats = _build_pair_rows(
            rows,
            objective=cfg.pair_objective,
            target_source=cfg.pair_target_source,
            max_candidates=cfg.pair_candidates_per_query,
        )
    else:
        pair_rows = []

    sorted_rows = sorted(rows, key=lambda row: row["as_of_time"])
    train_rows = [row for row in sorted_rows if row["split"] == "train"]
    validation_rows = [row for row in sorted_rows if row["split"] == "validation"]
    test_rows = [row for row in sorted_rows if row["split"] == "test"]

    excluded_by_reason: Dict[str, int] = {}
    for item in excluded_video_records:
        excluded_by_reason[item["reason"]] = excluded_by_reason.get(item["reason"], 0) + 1

    availability_by_objective: Dict[str, Dict[str, int]] = {}
    trajectory_unavailable_by_reason: Dict[str, int] = {}
    trajectory_unavailable_by_component: Dict[str, Dict[str, int]] = {}
    censorship_rates_by_split: Dict[str, Dict[str, float]] = {}
    if cfg.enable_trajectory_labels:
        for objective in sorted(VALID_PAIR_OBJECTIVES):
            objective_counts: Dict[str, int] = {
                "objective_available": 0,
                "objective_unavailable": 0,
            }
            for component_key in TRAJECTORY_COMPONENT_KEYS:
                objective_counts[f"{component_key}_available"] = 0
                objective_counts[f"{component_key}_unavailable"] = 0
            for row in rows:
                objective_payload = (
                    row.get("target_availability", {})
                    if isinstance(row.get("target_availability"), dict)
                    else {}
                ).get(objective, {})
                components = (
                    objective_payload.get("components", {})
                    if isinstance(objective_payload, dict)
                    else {}
                )
                if bool(objective_payload.get("objective_available", False)):
                    objective_counts["objective_available"] += 1
                else:
                    objective_counts["objective_unavailable"] += 1
                row_obj_reasons = (
                    row.get("target_unavailable_reasons", {})
                    if isinstance(row.get("target_unavailable_reasons"), dict)
                    else {}
                ).get(objective, [])
                if isinstance(row_obj_reasons, list):
                    for reason in row_obj_reasons:
                        key = str(reason)
                        trajectory_unavailable_by_reason[key] = (
                            trajectory_unavailable_by_reason.get(key, 0) + 1
                        )
                row_component_reasons = (
                    row.get("target_unavailable_component_reasons", {})
                    if isinstance(row.get("target_unavailable_component_reasons"), dict)
                    else {}
                ).get(objective, {})
                if isinstance(row_component_reasons, dict):
                    for component_key, reasons in row_component_reasons.items():
                        comp_stats = trajectory_unavailable_by_component.setdefault(
                            f"{objective}.{component_key}",
                            {},
                        )
                        if isinstance(reasons, list):
                            for reason in reasons:
                                reason_key = str(reason)
                                comp_stats[reason_key] = comp_stats.get(reason_key, 0) + 1
                for component_key in TRAJECTORY_COMPONENT_KEYS:
                    if bool(components.get(component_key, False)):
                        objective_counts[f"{component_key}_available"] += 1
                    else:
                        objective_counts[f"{component_key}_unavailable"] += 1
            availability_by_objective[objective] = objective_counts

        rows_by_split = {
            "train": train_rows,
            "validation": validation_rows,
            "test": test_rows,
        }
        for split_name, split_rows in rows_by_split.items():
            split_metrics: Dict[str, float] = {"rows": float(len(split_rows))}
            for objective in sorted(VALID_PAIR_OBJECTIVES):
                if not split_rows:
                    split_metrics[f"{objective}_censorship_rate"] = 0.0
                    continue
                unavailable = 0
                for row in split_rows:
                    objective_payload = (
                        row.get("target_availability", {})
                        if isinstance(row.get("target_availability"), dict)
                        else {}
                    ).get(objective, {})
                    if not bool(objective_payload.get("objective_available", False)):
                        unavailable += 1
                split_metrics[f"{objective}_censorship_rate"] = round(
                    unavailable / len(split_rows),
                    6,
                )
            censorship_rates_by_split[split_name] = split_metrics

    seen_video_ids: set[str] = set()
    excluded_video_ids: List[str] = []
    for item in excluded_video_records:
        video_id = item["video_id"]
        if video_id in seen_video_ids:
            continue
        seen_video_ids.add(video_id)
        excluded_video_ids.append(video_id)

    resolved_contract_version = str(bundle.version or CONTRACT_VERSION)
    if resolved_contract_version not in {CONTRACT_VERSION, LEGACY_CONTRACT_VERSION}:
        resolved_contract_version = CONTRACT_VERSION
    resolved_manifest_id = source_manifest_id or cfg.source_manifest_id or bundle.manifest_id
    resolved_manifest_path = source_manifest_path or cfg.source_manifest_path
    resolved_comment_feature_manifest_id = (
        cfg.comment_feature_manifest_id or loaded_comment_feature_manifest_id
    )
    resolved_comment_priors_manifest_id = (
        cfg.comment_priors_manifest_id or loaded_comment_priors_manifest_id
    )
    resolved_comment_version = COMMENT_INTELLIGENCE_VERSION

    payload = {
        "version": DATAMART_VERSION,
        "generated_at": datetime.now(timezone.utc),
        "source_contract_version": resolved_contract_version,
        "source_manifest_id": resolved_manifest_id,
        "source_manifest_path": resolved_manifest_path,
        "comment_feature_manifest_id": resolved_comment_feature_manifest_id,
        "comment_feature_manifest_path": cfg.comment_feature_manifest_path,
        "comment_priors_manifest_id": resolved_comment_priors_manifest_id,
        "comment_priors_manifest_path": cfg.comment_priors_manifest_path,
        "comment_intelligence_version": resolved_comment_version,
        "config": {
            "track": cfg.track,
            "min_history_hours": cfg.min_history_hours,
            "label_window_hours": cfg.label_window_hours,
            "min_author_rows_for_baseline": cfg.min_author_rows_for_baseline,
            "train_ratio": cfg.train_ratio,
            "validation_ratio": cfg.validation_ratio,
            "include_pair_rows": cfg.include_pair_rows,
            "pair_objective": cfg.pair_objective,
            "pair_target_source": cfg.pair_target_source,
            "pair_candidates_per_query": cfg.pair_candidates_per_query,
            "as_of_run_time": run_cutoff,
            "enable_trajectory_labels": cfg.enable_trajectory_labels,
            "trajectory_windows_hours": cfg.trajectory_windows_hours,
            "trajectory_objective_weights": cfg.trajectory_objective_weights,
            "comment_feature_manifest_id": resolved_comment_feature_manifest_id,
            "comment_feature_manifest_path": cfg.comment_feature_manifest_path,
            "comment_priors_manifest_id": resolved_comment_priors_manifest_id,
            "comment_priors_manifest_path": cfg.comment_priors_manifest_path,
        },
        "stats": {
            "rows_total": len(rows),
            "rows_censored": len(excluded_video_records),
            "train_count": len(train_rows),
            "validation_count": len(validation_rows),
            "test_count": len(test_rows),
            "pair_rows_total": len(pair_rows),
            "pair_rows_dropped_unavailable_target": int(sum(pair_drop_stats.values())),
            "pair_rows_dropped_by_reason": pair_drop_stats,
            "excluded_by_reason": excluded_by_reason,
            "availability_by_objective": availability_by_objective,
            "trajectory_unavailable_by_reason": trajectory_unavailable_by_reason,
            "trajectory_unavailable_by_component": trajectory_unavailable_by_component,
            "censorship_rates_by_split": censorship_rates_by_split,
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
    manifest_root: Optional[Path | str] = None,
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

    manifest_id: Optional[str] = None
    manifest_path: Optional[str] = None
    if manifest_root is not None:
        manifest_payload = build_contract_manifest(
            bundle=validated.bundle,
            manifest_root=manifest_root,
            source_file_hashes={source: _sha256_text(raw_jsonl)},
            as_of_time=as_of_time,
        )
        manifest_id = str(manifest_payload["manifest_id"])
        manifest_path = str(manifest_payload["manifest_dir"])
        validated.bundle = validated.bundle.model_copy(update={"manifest_id": manifest_id})

    mart = build_training_data_mart(
        validated.bundle,
        config=config,
        source_manifest_id=manifest_id,
        source_manifest_path=manifest_path,
    )
    if validated.warnings:
        mart["warnings"].extend(validated.warnings)
    validated_mart = TrainingDataMart.model_validate(mart)
    return validated_mart.model_dump(mode="python")


def build_training_data_mart_from_manifest(
    manifest_ref: Path | str,
    config: Optional[BuildTrainingDataMartConfig] = None,
) -> Dict[str, Any]:
    bundle = load_bundle_from_manifest(manifest_ref)
    manifest_path = Path(manifest_ref)
    manifest_id = bundle.manifest_id
    if manifest_path.is_dir():
        manifest_path_value = str(manifest_path)
    else:
        manifest_path_value = str(manifest_path.parent)
    return build_training_data_mart(
        bundle=bundle,
        config=config,
        source_manifest_id=manifest_id,
        source_manifest_path=manifest_path_value,
    )


__all__ = [
    "DATAMART_VERSION",
    "BuildTrainingDataMartConfig",
    "TrainingRow",
    "PairTrainingRow",
    "TrainingDataMart",
    "build_training_data_mart",
    "build_training_data_mart_from_jsonl",
    "build_training_data_mart_from_manifest",
]
