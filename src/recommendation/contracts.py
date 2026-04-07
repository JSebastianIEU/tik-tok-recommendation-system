from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field, HttpUrl, ValidationError, model_validator


LEGACY_CONTRACT_VERSION = "contract.v1"
CONTRACT_VERSION = "contract.v2"
WATERMARK_GRACE_HOURS = 48

MODELING_TRACKS = ("pre_publication", "post_publication")
FEATURE_NAMESPACES = (
    "video_metadata",
    "video_snapshots",
    "author_profile",
    "comments",
    "comment_snapshots",
)
TRACK_ALLOWED_FEATURES = {
    "pre_publication": {"video_metadata", "video_snapshots", "author_profile"},
    "post_publication": set(FEATURE_NAMESPACES),
}

LatenessClass = Literal["on_time", "late_within_grace", "late_out_of_watermark"]
QualityTier = Literal["gold", "silver", "bronze", "quarantine"]


def _parse_iso(value: str) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        if value != value:  # nan
            return 0
        return max(0, int(round(value)))
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return 0
        if parsed != parsed:
            return 0
        return max(0, int(round(parsed)))
    return 0


def _normalize_string_array(value: Any) -> List[str]:
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
        return [item for item in items if item]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                normalized = item.strip()
                if normalized:
                    out.append(normalized)
        return out
    return []


def _normalize_hashtags(value: Any) -> List[str]:
    normalized = [
        f"#{item.replace('#', '').strip().lower()}"
        for item in _normalize_string_array(value)
        if item.strip("#").strip()
    ]
    seen: Set[str] = set()
    out: List[str] = []
    for item in normalized:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _safe_strip(value: Any, fallback: str = "") -> str:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else fallback
    return fallback


def _prefer_incoming(old_value: Any, new_value: Any, prefer_new: bool) -> Any:
    if not prefer_new:
        return old_value
    if new_value is None:
        return old_value
    if isinstance(new_value, str):
        return new_value if new_value.strip() else old_value
    if isinstance(new_value, list):
        return new_value if new_value else old_value
    return new_value


def _resolve_video_snapshot_time(
    record: Dict[str, Any],
    generated_at: datetime,
    strict_timestamps: bool,
    video_id: str,
) -> Tuple[Optional[datetime], List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []

    primary_raw = _safe_strip(record.get("scraped_at"))
    if primary_raw:
        parsed = _parse_iso(primary_raw)
        if parsed is not None:
            return parsed, warnings, errors
        warnings.append(f"video {video_id}: invalid scraped_at value '{primary_raw}'")

    secondary_raw = _safe_strip(record.get("metrics_scraped_at"))
    if secondary_raw:
        parsed = _parse_iso(secondary_raw)
        if parsed is not None:
            return parsed, warnings, errors
        warnings.append(f"video {video_id}: invalid metrics_scraped_at value '{secondary_raw}'")

    if strict_timestamps:
        errors.append(
            f"video {video_id}: missing valid snapshot timestamp (expected scraped_at or metrics_scraped_at)"
        )
        return None, warnings, errors

    warnings.append(
        f"video {video_id}: missing valid snapshot timestamp, fallback to as_of_time"
    )
    return generated_at, warnings, errors


def _resolve_comment_snapshot_time(
    comment_obj: Dict[str, Any],
    parent_snapshot_time: datetime,
    strict_timestamps: bool,
    video_id: str,
    comment_id: str,
) -> Tuple[Optional[datetime], List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []

    raw = _safe_strip(comment_obj.get("scraped_at"))
    if raw:
        parsed = _parse_iso(raw)
        if parsed is not None:
            return parsed, warnings, errors
        warnings.append(
            f"video {video_id}, comment {comment_id}: invalid scraped_at value '{raw}'"
        )

    if strict_timestamps:
        errors.append(
            f"video {video_id}, comment {comment_id}: missing valid scraped_at timestamp"
        )
        return None, warnings, errors

    warnings.append(
        f"video {video_id}, comment {comment_id}: missing valid scraped_at, fallback to parent video snapshot timestamp"
    )
    return parent_snapshot_time, warnings, errors


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _score_to_quality_tier(score: float) -> QualityTier:
    if score >= 0.85:
        return "gold"
    if score >= 0.65:
        return "silver"
    if score >= 0.4:
        return "bronze"
    return "quarantine"


class RecordQuality(BaseModel):
    quality_score: float = Field(ge=0.0, le=1.0)
    quality_tier: QualityTier
    quality_reasons: List[str] = Field(default_factory=list)
    required_fields_ok: bool = True


class SourceWatermark(BaseModel):
    source: str
    watermark_time: datetime
    records_seen: int = Field(ge=0)
    late_within_grace: int = Field(ge=0)
    late_out_of_watermark: int = Field(ge=0)


class QuarantineRecord(BaseModel):
    row_key: str
    source: str
    entity_type: str
    entity_id: str
    reason: str
    detail: str
    observed_at: datetime


class CanonicalAuthor(BaseModel):
    author_id: str
    username: Optional[str] = None
    display_name: Optional[str] = None
    followers_count: int = Field(default=0, ge=0)
    verified: Optional[bool] = None
    source: Optional[str] = None
    scraped_at: Optional[datetime] = None


class CanonicalVideo(BaseModel):
    video_id: str
    author_id: str
    caption: str = ""
    hashtags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    search_query: Optional[str] = None
    posted_at: datetime
    video_url: Optional[HttpUrl] = None
    audio_id: Optional[str] = None
    duration_seconds: Optional[int] = Field(default=None, ge=0)
    language: Optional[str] = None
    ingested_at: Optional[datetime] = None
    source: Optional[str] = None
    quality: Optional[RecordQuality] = None


class CanonicalVideoSnapshot(BaseModel):
    video_snapshot_id: str
    observation_id: Optional[str] = None
    video_id: str
    scraped_at: datetime
    event_time: Optional[datetime] = None
    ingested_at: Optional[datetime] = None
    source_watermark_time: Optional[datetime] = None
    lateness_class: LatenessClass = "on_time"
    supersedes_observation_id: Optional[str] = None
    superseded_by_observation_id: Optional[str] = None
    views: int = Field(ge=0)
    likes: int = Field(ge=0)
    comments_count: int = Field(ge=0)
    shares: int = Field(ge=0)
    plays: Optional[int] = Field(default=None, ge=0)
    position: Optional[int] = Field(default=None, ge=0)
    run_id: Optional[str] = None
    source: Optional[str] = None
    quality: Optional[RecordQuality] = None

    @model_validator(mode="after")
    def _apply_defaults(self) -> "CanonicalVideoSnapshot":
        if self.observation_id is None:
            self.observation_id = f"obs::video::{self.video_snapshot_id}"
        if self.event_time is None:
            self.event_time = self.scraped_at
        if self.ingested_at is None:
            self.ingested_at = self.event_time
        if self.source_watermark_time is None:
            self.source_watermark_time = self.event_time
        if self.quality is None:
            self.quality = _quality_for_video_snapshot(
                views=self.views,
                likes=self.likes,
                comments_count=self.comments_count,
                shares=self.shares,
                timestamp_fallback=False,
            )
        return self


class CanonicalComment(BaseModel):
    comment_id: str
    video_id: str
    author_id: Optional[str] = None
    text: str
    parent_comment_id: Optional[str] = None
    root_comment_id: Optional[str] = None
    comment_level: Optional[int] = Field(default=None, ge=0, le=8)
    created_at: datetime
    ingested_at: Optional[datetime] = None
    source: Optional[str] = None
    quality: Optional[RecordQuality] = None

    @model_validator(mode="after")
    def _apply_defaults(self) -> "CanonicalComment":
        if self.ingested_at is None:
            self.ingested_at = self.created_at
        if self.quality is None:
            self.quality = _quality_for_comment(
                text=self.text,
                created_at_fallback=False,
            )
        return self


class CanonicalCommentSnapshot(BaseModel):
    comment_snapshot_id: str
    observation_id: Optional[str] = None
    comment_id: str
    video_id: str
    scraped_at: datetime
    event_time: Optional[datetime] = None
    ingested_at: Optional[datetime] = None
    source_watermark_time: Optional[datetime] = None
    lateness_class: LatenessClass = "on_time"
    supersedes_observation_id: Optional[str] = None
    superseded_by_observation_id: Optional[str] = None
    likes: int = Field(ge=0)
    reply_count: int = Field(ge=0)
    source: Optional[str] = None
    quality: Optional[RecordQuality] = None

    @model_validator(mode="after")
    def _apply_defaults(self) -> "CanonicalCommentSnapshot":
        if self.observation_id is None:
            self.observation_id = f"obs::comment::{self.comment_snapshot_id}"
        if self.event_time is None:
            self.event_time = self.scraped_at
        if self.ingested_at is None:
            self.ingested_at = self.event_time
        if self.source_watermark_time is None:
            self.source_watermark_time = self.event_time
        if self.quality is None:
            self.quality = _quality_for_comment_snapshot(
                likes=self.likes,
                reply_count=self.reply_count,
                timestamp_fallback=False,
            )
        return self


class CanonicalDatasetBundle(BaseModel):
    version: str = CONTRACT_VERSION
    generated_at: datetime
    authors: List[CanonicalAuthor]
    videos: List[CanonicalVideo]
    video_snapshots: List[CanonicalVideoSnapshot]
    comments: List[CanonicalComment]
    comment_snapshots: List[CanonicalCommentSnapshot]
    source_watermarks: List[SourceWatermark] = Field(default_factory=list)
    quarantine_records: List[QuarantineRecord] = Field(default_factory=list)
    manifest_id: Optional[str] = None


class RawDatasetValidationResult(BaseModel):
    ok: bool
    errors: List[str]
    warnings: List[str]
    bundle: Optional[CanonicalDatasetBundle] = None


def _quality_for_video(
    *,
    caption: str,
    duration_seconds: Optional[int],
    language: Optional[str],
    video_url: Optional[str],
) -> RecordQuality:
    reasons: List[str] = []
    required_fields_ok = True
    if not caption.strip():
        reasons.append("missing_caption")
    if duration_seconds is None:
        reasons.append("missing_duration_seconds")
    if not language:
        reasons.append("missing_language")
    if not video_url:
        reasons.append("missing_video_url")
    if duration_seconds is not None and duration_seconds > 180:
        reasons.append("long_duration_outlier")
    if duration_seconds is not None and duration_seconds < 5:
        reasons.append("short_duration_outlier")
    if not caption.strip() and duration_seconds is None:
        required_fields_ok = False
    score = _clamp(1.0 - 0.12 * len(reasons), 0.0, 1.0)
    return RecordQuality(
        quality_score=round(score, 6),
        quality_tier=_score_to_quality_tier(score),
        quality_reasons=reasons,
        required_fields_ok=required_fields_ok,
    )


def _quality_for_video_snapshot(
    *,
    views: int,
    likes: int,
    comments_count: int,
    shares: int,
    timestamp_fallback: bool,
) -> RecordQuality:
    reasons: List[str] = []
    required_fields_ok = True
    if views == 0:
        reasons.append("zero_views")
    if likes + comments_count + shares == 0:
        reasons.append("zero_engagement")
    if timestamp_fallback:
        reasons.append("timestamp_fallback")
    if views > 0 and likes > views:
        reasons.append("likes_exceed_views")
    if views > 0 and shares > views:
        reasons.append("shares_exceed_views")
    if views > 0 and comments_count > views:
        reasons.append("comments_exceed_views")
    if timestamp_fallback:
        required_fields_ok = False
    score = _clamp(1.0 - 0.1 * len(reasons), 0.0, 1.0)
    return RecordQuality(
        quality_score=round(score, 6),
        quality_tier=_score_to_quality_tier(score),
        quality_reasons=reasons,
        required_fields_ok=required_fields_ok,
    )


def _quality_for_comment(
    *,
    text: str,
    created_at_fallback: bool,
) -> RecordQuality:
    reasons: List[str] = []
    required_fields_ok = True
    if len(text.strip()) < 2:
        reasons.append("very_short_text")
    if created_at_fallback:
        reasons.append("created_at_fallback")
        required_fields_ok = False
    score = _clamp(1.0 - 0.15 * len(reasons), 0.0, 1.0)
    return RecordQuality(
        quality_score=round(score, 6),
        quality_tier=_score_to_quality_tier(score),
        quality_reasons=reasons,
        required_fields_ok=required_fields_ok,
    )


def _quality_for_comment_snapshot(
    *,
    likes: int,
    reply_count: int,
    timestamp_fallback: bool,
) -> RecordQuality:
    reasons: List[str] = []
    required_fields_ok = True
    if likes == 0 and reply_count == 0:
        reasons.append("zero_comment_engagement")
    if timestamp_fallback:
        reasons.append("timestamp_fallback")
        required_fields_ok = False
    score = _clamp(1.0 - 0.12 * len(reasons), 0.0, 1.0)
    return RecordQuality(
        quality_score=round(score, 6),
        quality_tier=_score_to_quality_tier(score),
        quality_reasons=reasons,
        required_fields_ok=required_fields_ok,
    )


def _merge_author(
    existing: CanonicalAuthor,
    incoming: CanonicalAuthor,
    existing_seen_at: datetime,
    incoming_seen_at: datetime,
) -> CanonicalAuthor:
    prefer_new = incoming_seen_at >= existing_seen_at
    payload = existing.model_dump(mode="python")
    payload["username"] = _prefer_incoming(
        payload.get("username"), incoming.username, prefer_new
    )
    payload["display_name"] = _prefer_incoming(
        payload.get("display_name"), incoming.display_name, prefer_new
    )
    payload["verified"] = _prefer_incoming(
        payload.get("verified"), incoming.verified, prefer_new
    )
    payload["source"] = _prefer_incoming(payload.get("source"), incoming.source, prefer_new)
    payload["followers_count"] = max(existing.followers_count, incoming.followers_count)
    if existing.scraped_at and incoming.scraped_at:
        payload["scraped_at"] = max(existing.scraped_at, incoming.scraped_at)
    else:
        payload["scraped_at"] = incoming.scraped_at or existing.scraped_at
    return CanonicalAuthor.model_validate(payload)


def _merge_video(
    existing: CanonicalVideo,
    incoming: CanonicalVideo,
    existing_seen_at: datetime,
    incoming_seen_at: datetime,
) -> CanonicalVideo:
    prefer_new = incoming_seen_at >= existing_seen_at
    payload = existing.model_dump(mode="python")

    payload["posted_at"] = min(existing.posted_at, incoming.posted_at)
    payload["caption"] = _prefer_incoming(payload.get("caption"), incoming.caption, prefer_new)
    payload["hashtags"] = _prefer_incoming(
        payload.get("hashtags"), incoming.hashtags, prefer_new
    )
    payload["keywords"] = _prefer_incoming(
        payload.get("keywords"), incoming.keywords, prefer_new
    )
    payload["search_query"] = _prefer_incoming(
        payload.get("search_query"), incoming.search_query, prefer_new
    )
    payload["video_url"] = _prefer_incoming(
        payload.get("video_url"), incoming.video_url, prefer_new
    )
    payload["audio_id"] = _prefer_incoming(
        payload.get("audio_id"), incoming.audio_id, prefer_new
    )
    payload["duration_seconds"] = _prefer_incoming(
        payload.get("duration_seconds"), incoming.duration_seconds, prefer_new
    )
    payload["language"] = _prefer_incoming(
        payload.get("language"), incoming.language, prefer_new
    )
    payload["ingested_at"] = _prefer_incoming(
        payload.get("ingested_at"), incoming.ingested_at, prefer_new
    )
    payload["source"] = _prefer_incoming(payload.get("source"), incoming.source, prefer_new)
    payload["quality"] = _prefer_incoming(payload.get("quality"), incoming.quality, prefer_new)
    return CanonicalVideo.model_validate(payload)


def _merge_comment(
    existing: CanonicalComment,
    incoming: CanonicalComment,
    existing_seen_at: datetime,
    incoming_seen_at: datetime,
) -> CanonicalComment:
    prefer_new = incoming_seen_at >= existing_seen_at
    payload = existing.model_dump(mode="python")
    payload["created_at"] = min(existing.created_at, incoming.created_at)
    payload["author_id"] = _prefer_incoming(
        payload.get("author_id"), incoming.author_id, prefer_new
    )
    payload["text"] = _prefer_incoming(payload.get("text"), incoming.text, prefer_new)
    payload["parent_comment_id"] = _prefer_incoming(
        payload.get("parent_comment_id"), incoming.parent_comment_id, prefer_new
    )
    payload["root_comment_id"] = _prefer_incoming(
        payload.get("root_comment_id"), incoming.root_comment_id, prefer_new
    )
    payload["comment_level"] = _prefer_incoming(
        payload.get("comment_level"), incoming.comment_level, prefer_new
    )
    payload["ingested_at"] = _prefer_incoming(
        payload.get("ingested_at"), incoming.ingested_at, prefer_new
    )
    payload["source"] = _prefer_incoming(payload.get("source"), incoming.source, prefer_new)
    payload["quality"] = _prefer_incoming(payload.get("quality"), incoming.quality, prefer_new)
    return CanonicalComment.model_validate(payload)


def _normalize_raw_record(
    record: Dict[str, Any],
    generated_at: datetime,
    source: str,
    row_key: str,
    strict_timestamps: bool = False,
) -> Tuple[Optional[Dict[str, Any]], List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    video_id = _safe_strip(record.get("video_id"))
    if not video_id:
        return None, ["video_id is required"], warnings

    posted_at = _parse_iso(_safe_strip(record.get("posted_at")))
    if posted_at is None:
        warnings.append(f"video {video_id}: invalid posted_at, fallback to generated_at")
        posted_at = generated_at

    snapshot_scraped_at, ts_warnings, ts_errors = _resolve_video_snapshot_time(
        record=record,
        generated_at=generated_at,
        strict_timestamps=strict_timestamps,
        video_id=video_id,
    )
    warnings.extend(ts_warnings)
    errors.extend(ts_errors)
    if snapshot_scraped_at is None:
        return None, errors, warnings

    author_field = record.get("author")
    author_obj = author_field if isinstance(author_field, dict) else {}
    author_id = (
        _safe_strip(author_obj.get("author_id"))
        or _safe_strip(author_obj.get("username"))
        or _safe_strip(author_field)
        or f"unknown-author-{video_id}"
    ).lower()

    author = {
        "author_id": author_id,
        "username": _safe_strip(author_obj.get("username")) or None,
        "followers_count": _to_non_negative_int(author_obj.get("followers")),
        "verified": author_obj.get("verified")
        if isinstance(author_obj.get("verified"), bool)
        else None,
        "source": source,
        "scraped_at": generated_at,
    }

    metrics_obj = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    likes = _to_non_negative_int(metrics_obj.get("likes", record.get("likes")))
    comments_count = _to_non_negative_int(
        metrics_obj.get("comments_count", record.get("comments_count"))
    )
    shares = _to_non_negative_int(metrics_obj.get("shares", record.get("shares")))
    views = _to_non_negative_int(metrics_obj.get("views", record.get("views")))

    video_url = _safe_strip(record.get("video_url")) or None
    duration = (
        _to_non_negative_int((record.get("video_meta") or {}).get("duration_seconds"))
        if isinstance(record.get("video_meta"), dict)
        and (record.get("video_meta") or {}).get("duration_seconds") is not None
        else None
    )
    language = (
        _safe_strip((record.get("video_meta") or {}).get("language")).lower() or None
        if isinstance(record.get("video_meta"), dict)
        else None
    )

    video = {
        "video_id": video_id,
        "author_id": author_id,
        "caption": _safe_strip(record.get("caption")),
        "hashtags": _normalize_hashtags(record.get("hashtags")),
        "keywords": _normalize_string_array(record.get("keywords")),
        "search_query": _safe_strip(record.get("search_query")) or None,
        "posted_at": posted_at,
        "video_url": video_url,
        "audio_id": _safe_strip((record.get("audio") or {}).get("audio_id")) or None
        if isinstance(record.get("audio"), dict)
        else None,
        "duration_seconds": duration,
        "language": language,
        "ingested_at": generated_at,
        "source": source,
        "quality": _quality_for_video(
            caption=_safe_strip(record.get("caption")),
            duration_seconds=duration,
            language=language,
            video_url=video_url,
        ),
    }

    video_timestamp_fallback = not _safe_strip(record.get("scraped_at")) and not _safe_strip(
        record.get("metrics_scraped_at")
    )
    video_snapshot = {
        "video_snapshot_id": f"{video_id}::{_to_iso(snapshot_scraped_at)}::{row_key}",
        "observation_id": f"obs::video::{video_id}::{_to_iso(snapshot_scraped_at)}::{row_key}",
        "video_id": video_id,
        "scraped_at": snapshot_scraped_at,
        "event_time": snapshot_scraped_at,
        "ingested_at": generated_at,
        "source_watermark_time": snapshot_scraped_at,
        "lateness_class": "on_time",
        "views": views,
        "likes": likes,
        "comments_count": comments_count,
        "shares": shares,
        "run_id": _safe_strip(record.get("run_id")) or None,
        "source": source,
        "quality": _quality_for_video_snapshot(
            views=views,
            likes=likes,
            comments_count=comments_count,
            shares=shares,
            timestamp_fallback=video_timestamp_fallback,
        ),
    }

    comments: List[Dict[str, Any]] = []
    comment_snapshots: List[Dict[str, Any]] = []
    raw_comments = record.get("comments") if isinstance(record.get("comments"), list) else []
    for idx, raw_comment in enumerate(raw_comments, start=1):
        if isinstance(raw_comment, str):
            comment_obj = {"text": raw_comment}
        elif isinstance(raw_comment, dict):
            comment_obj = raw_comment
        else:
            continue

        text = _safe_strip(comment_obj.get("text"))
        if not text:
            continue
        raw_comment_id = _safe_strip(comment_obj.get("comment_id")) or f"comment-{idx}"
        comment_id = f"{video_id}::{raw_comment_id}"

        created_at_raw = _safe_strip(comment_obj.get("created_at"))
        created_at = _parse_iso(created_at_raw)
        created_at_fallback = created_at is None
        if created_at is None:
            warnings.append(
                f"video {video_id}, comment {comment_id}: invalid created_at, fallback to posted_at"
            )
            created_at = posted_at

        comment_snapshot_scraped_at, comment_ts_warnings, comment_ts_errors = (
            _resolve_comment_snapshot_time(
                comment_obj=comment_obj,
                parent_snapshot_time=snapshot_scraped_at,
                strict_timestamps=strict_timestamps,
                video_id=video_id,
                comment_id=comment_id,
            )
        )
        warnings.extend(comment_ts_warnings)
        errors.extend(comment_ts_errors)
        if comment_snapshot_scraped_at is None:
            continue
        comment_timestamp_fallback = not _safe_strip(comment_obj.get("scraped_at"))

        comments.append(
            {
                "comment_id": comment_id,
                "video_id": video_id,
                "author_id": _safe_strip(comment_obj.get("author_id")) or None,
                "text": text,
                "parent_comment_id": _safe_strip(comment_obj.get("parent_comment_id")) or None,
                "root_comment_id": _safe_strip(comment_obj.get("root_comment_id")) or None,
                "comment_level": _to_non_negative_int(comment_obj.get("comment_level"))
                if comment_obj.get("comment_level") is not None
                else None,
                "created_at": created_at,
                "ingested_at": generated_at,
                "source": source,
                "quality": _quality_for_comment(
                    text=text,
                    created_at_fallback=created_at_fallback,
                ),
            }
        )
        comment_snapshots.append(
            {
                "comment_snapshot_id": f"{comment_id}::{_to_iso(comment_snapshot_scraped_at)}::{row_key}",
                "observation_id": f"obs::comment::{comment_id}::{_to_iso(comment_snapshot_scraped_at)}::{row_key}",
                "comment_id": comment_id,
                "video_id": video_id,
                "scraped_at": comment_snapshot_scraped_at,
                "event_time": comment_snapshot_scraped_at,
                "ingested_at": generated_at,
                "source_watermark_time": comment_snapshot_scraped_at,
                "lateness_class": "on_time",
                "likes": _to_non_negative_int(comment_obj.get("likes")),
                "reply_count": _to_non_negative_int(comment_obj.get("reply_count")),
                "source": source,
                "quality": _quality_for_comment_snapshot(
                    likes=_to_non_negative_int(comment_obj.get("likes")),
                    reply_count=_to_non_negative_int(comment_obj.get("reply_count")),
                    timestamp_fallback=comment_timestamp_fallback,
                ),
            }
        )

    try:
        normalized = {
            "author": CanonicalAuthor.model_validate(author),
            "video": CanonicalVideo.model_validate(video),
            "video_snapshot": CanonicalVideoSnapshot.model_validate(video_snapshot),
            "comments": [CanonicalComment.model_validate(item) for item in comments],
            "comment_snapshots": [
                CanonicalCommentSnapshot.model_validate(item) for item in comment_snapshots
            ],
        }
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(part) for part in err.get("loc", []))
            errors.append(f"{loc}: {err.get('msg')}")
        return None, errors, warnings

    return normalized, errors, warnings


def _classify_lateness(
    event_time: datetime,
    previous_watermark: Optional[datetime],
    grace_hours: int = WATERMARK_GRACE_HOURS,
) -> Tuple[LatenessClass, datetime]:
    if previous_watermark is None or event_time >= previous_watermark:
        return "on_time", event_time if previous_watermark is None else max(event_time, previous_watermark)
    delay = previous_watermark - event_time
    if delay <= timedelta(hours=grace_hours):
        return "late_within_grace", previous_watermark
    return "late_out_of_watermark", previous_watermark


def validate_contract_bundle(bundle: CanonicalDatasetBundle) -> List[str]:
    errors: List[str] = []

    def _dupe_errors(values: Iterable[str], label: str) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for value in values:
            if value in seen:
                out.append(f"{label}: duplicate id '{value}'")
            else:
                seen.add(value)
        return out

    errors.extend(_dupe_errors((item.author_id for item in bundle.authors), "authors"))
    errors.extend(_dupe_errors((item.video_id for item in bundle.videos), "videos"))
    errors.extend(
        _dupe_errors((item.video_snapshot_id for item in bundle.video_snapshots), "video_snapshots")
    )
    errors.extend(
        _dupe_errors((item.observation_id for item in bundle.video_snapshots), "video_snapshots")
    )
    errors.extend(_dupe_errors((item.comment_id for item in bundle.comments), "comments"))
    errors.extend(
        _dupe_errors(
            (item.comment_snapshot_id for item in bundle.comment_snapshots),
            "comment_snapshots",
        )
    )
    errors.extend(
        _dupe_errors(
            (item.observation_id for item in bundle.comment_snapshots),
            "comment_snapshots",
        )
    )
    errors.extend(_dupe_errors((item.source for item in bundle.source_watermarks), "source_watermarks"))

    author_ids = {item.author_id for item in bundle.authors}
    video_ids = {item.video_id for item in bundle.videos}
    comments_by_id = {item.comment_id: item for item in bundle.comments}
    video_observations = {item.observation_id: item for item in bundle.video_snapshots}
    comment_observations = {item.observation_id: item for item in bundle.comment_snapshots}
    observation_ids = set(video_observations.keys()) | set(comment_observations.keys())

    for video in bundle.videos:
        if video.author_id not in author_ids:
            errors.append(
                f"videos: video '{video.video_id}' references missing author '{video.author_id}'"
            )
    for snapshot in bundle.video_snapshots:
        if snapshot.video_id not in video_ids:
            errors.append(
                f"video_snapshots: snapshot '{snapshot.video_snapshot_id}' references missing video '{snapshot.video_id}'"
            )
        if snapshot.supersedes_observation_id and snapshot.supersedes_observation_id not in observation_ids:
            errors.append(
                f"video_snapshots: observation '{snapshot.observation_id}' supersedes missing observation '{snapshot.supersedes_observation_id}'"
            )
        if snapshot.superseded_by_observation_id and snapshot.superseded_by_observation_id not in observation_ids:
            errors.append(
                f"video_snapshots: observation '{snapshot.observation_id}' superseded_by missing observation '{snapshot.superseded_by_observation_id}'"
            )

    for comment in bundle.comments:
        if comment.video_id not in video_ids:
            errors.append(
                f"comments: comment '{comment.comment_id}' references missing video '{comment.video_id}'"
            )

    for snapshot in bundle.comment_snapshots:
        if snapshot.video_id not in video_ids:
            errors.append(
                f"comment_snapshots: snapshot '{snapshot.comment_snapshot_id}' references missing video '{snapshot.video_id}'"
            )
            continue
        linked = comments_by_id.get(snapshot.comment_id)
        if linked is None:
            errors.append(
                f"comment_snapshots: snapshot '{snapshot.comment_snapshot_id}' references missing comment '{snapshot.comment_id}'"
            )
            continue
        if linked.video_id != snapshot.video_id:
            errors.append(
                f"comment_snapshots: comment '{linked.comment_id}' belongs to video '{linked.video_id}' but snapshot points to '{snapshot.video_id}'"
            )
        if snapshot.supersedes_observation_id and snapshot.supersedes_observation_id not in observation_ids:
            errors.append(
                f"comment_snapshots: observation '{snapshot.observation_id}' supersedes missing observation '{snapshot.supersedes_observation_id}'"
            )
        if snapshot.superseded_by_observation_id and snapshot.superseded_by_observation_id not in observation_ids:
            errors.append(
                f"comment_snapshots: observation '{snapshot.observation_id}' superseded_by missing observation '{snapshot.superseded_by_observation_id}'"
            )

    return errors


def validate_as_of_time_policy(bundle: CanonicalDatasetBundle, as_of_time: datetime) -> List[str]:
    errors: List[str] = []
    as_of = as_of_time.astimezone(timezone.utc)

    def _check(label: str, ts: datetime) -> None:
        if ts.astimezone(timezone.utc) > as_of:
            errors.append(f"{label}: timestamp exceeds as_of_time")

    for video in bundle.videos:
        _check(f"videos.{video.video_id}.posted_at", video.posted_at)
        if video.ingested_at is not None:
            _check(f"videos.{video.video_id}.ingested_at", video.ingested_at)
    for snapshot in bundle.video_snapshots:
        _check(f"video_snapshots.{snapshot.video_snapshot_id}.scraped_at", snapshot.scraped_at)
        _check(f"video_snapshots.{snapshot.video_snapshot_id}.event_time", snapshot.event_time)
        _check(
            f"video_snapshots.{snapshot.video_snapshot_id}.ingested_at",
            snapshot.ingested_at,
        )
        _check(
            f"video_snapshots.{snapshot.video_snapshot_id}.source_watermark_time",
            snapshot.source_watermark_time,
        )
    for comment in bundle.comments:
        _check(f"comments.{comment.comment_id}.created_at", comment.created_at)
        _check(f"comments.{comment.comment_id}.ingested_at", comment.ingested_at)
    for snapshot in bundle.comment_snapshots:
        _check(
            f"comment_snapshots.{snapshot.comment_snapshot_id}.scraped_at",
            snapshot.scraped_at,
        )
        _check(
            f"comment_snapshots.{snapshot.comment_snapshot_id}.event_time",
            snapshot.event_time,
        )
        _check(
            f"comment_snapshots.{snapshot.comment_snapshot_id}.ingested_at",
            snapshot.ingested_at,
        )
        _check(
            f"comment_snapshots.{snapshot.comment_snapshot_id}.source_watermark_time",
            snapshot.source_watermark_time,
        )

    return errors


def validate_feature_access_policy(track: str, requested_features: Sequence[str]) -> List[str]:
    if track not in TRACK_ALLOWED_FEATURES:
        return [f"unknown track '{track}'"]
    allowed = TRACK_ALLOWED_FEATURES[track]
    return [
        f"feature '{feature}' is not allowed for track '{track}'"
        for feature in requested_features
        if feature not in allowed
    ]


def validate_raw_dataset_jsonl_against_contract(
    raw_jsonl: str,
    as_of_time: datetime,
    source: str = "dataset_jsonl",
    strict_timestamps: bool = False,
) -> RawDatasetValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    quarantine_records: List[QuarantineRecord] = []
    authors_by_id: Dict[str, CanonicalAuthor] = {}
    videos_by_id: Dict[str, CanonicalVideo] = {}
    author_last_seen_at: Dict[str, datetime] = {}
    video_last_seen_at: Dict[str, datetime] = {}
    comments_by_id: Dict[str, CanonicalComment] = {}
    comment_last_seen_at: Dict[str, datetime] = {}
    video_snapshots: List[CanonicalVideoSnapshot] = []
    comment_snapshots: List[CanonicalCommentSnapshot] = []

    latest_video_obs_by_video: Dict[str, int] = {}
    latest_comment_obs_by_comment: Dict[str, int] = {}

    source_state: Dict[str, Dict[str, Any]] = {}
    generated_at = as_of_time.astimezone(timezone.utc)

    for line_idx, raw_line in enumerate(raw_jsonl.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        row_key = f"line-{line_idx}"
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"line {line_idx}: invalid JSON")
            quarantine_records.append(
                QuarantineRecord(
                    row_key=row_key,
                    source=source,
                    entity_type="row",
                    entity_id=row_key,
                    reason="invalid_json",
                    detail="Could not parse JSON record.",
                    observed_at=generated_at,
                )
            )
            continue

        if not isinstance(parsed, dict):
            errors.append(f"line {line_idx}: record must be an object")
            quarantine_records.append(
                QuarantineRecord(
                    row_key=row_key,
                    source=source,
                    entity_type="row",
                    entity_id=row_key,
                    reason="invalid_record_type",
                    detail="JSON line must decode into an object.",
                    observed_at=generated_at,
                )
            )
            continue

        normalized, row_errors, row_warnings = _normalize_raw_record(
            parsed,
            generated_at=generated_at,
            source=source,
            row_key=row_key,
            strict_timestamps=strict_timestamps,
        )
        warnings.extend([f"line {line_idx}: {warn}" for warn in row_warnings])
        errors.extend([f"line {line_idx}: {err}" for err in row_errors])
        if normalized is None:
            quarantine_records.append(
                QuarantineRecord(
                    row_key=row_key,
                    source=source,
                    entity_type="row",
                    entity_id=str(parsed.get("video_id") or row_key),
                    reason="normalization_failed",
                    detail=" | ".join(row_errors[:3]) if row_errors else "Unknown normalization error.",
                    observed_at=generated_at,
                )
            )
            continue

        author: CanonicalAuthor = normalized["author"]
        video: CanonicalVideo = normalized["video"]
        video_snapshot: CanonicalVideoSnapshot = normalized["video_snapshot"]
        incoming_seen_at = video_snapshot.scraped_at.astimezone(timezone.utc)

        state = source_state.setdefault(
            source,
            {
                "watermark": None,
                "records_seen": 0,
                "late_within_grace": 0,
                "late_out_of_watermark": 0,
            },
        )

        existing_author = authors_by_id.get(author.author_id)
        if existing_author is None:
            authors_by_id[author.author_id] = author
            author_last_seen_at[author.author_id] = incoming_seen_at
        else:
            authors_by_id[author.author_id] = _merge_author(
                existing=existing_author,
                incoming=author,
                existing_seen_at=author_last_seen_at[author.author_id],
                incoming_seen_at=incoming_seen_at,
            )
            author_last_seen_at[author.author_id] = max(
                author_last_seen_at[author.author_id], incoming_seen_at
            )

        existing_video = videos_by_id.get(video.video_id)
        if existing_video is None:
            videos_by_id[video.video_id] = video
            video_last_seen_at[video.video_id] = incoming_seen_at
        else:
            if existing_video.author_id != video.author_id:
                errors.append(
                    f"line {line_idx}: conflicting author_id for video '{video.video_id}' (existing='{existing_video.author_id}', incoming='{video.author_id}')"
                )
                quarantine_records.append(
                    QuarantineRecord(
                        row_key=row_key,
                        source=source,
                        entity_type="video",
                        entity_id=video.video_id,
                        reason="conflicting_author_id",
                        detail=f"existing={existing_video.author_id}, incoming={video.author_id}",
                        observed_at=generated_at,
                    )
                )
                continue
            if existing_video.posted_at != video.posted_at:
                warnings.append(
                    f"line {line_idx}: conflicting posted_at for video '{video.video_id}', canonical value keeps earliest timestamp"
                )
            videos_by_id[video.video_id] = _merge_video(
                existing=existing_video,
                incoming=video,
                existing_seen_at=video_last_seen_at[video.video_id],
                incoming_seen_at=incoming_seen_at,
            )
            video_last_seen_at[video.video_id] = max(
                video_last_seen_at[video.video_id], incoming_seen_at
            )

        video_lateness, watermark_after = _classify_lateness(
            event_time=video_snapshot.event_time.astimezone(timezone.utc),
            previous_watermark=state["watermark"],
        )
        watermark_before = state["watermark"] or watermark_after
        video_snapshot = video_snapshot.model_copy(
            update={
                "source_watermark_time": watermark_before,
                "lateness_class": video_lateness,
            }
        )
        state["records_seen"] += 1
        if video_lateness == "late_within_grace":
            state["late_within_grace"] += 1
        if video_lateness == "late_out_of_watermark":
            state["late_out_of_watermark"] += 1
            quarantine_records.append(
                QuarantineRecord(
                    row_key=row_key,
                    source=source,
                    entity_type="video_snapshot",
                    entity_id=video_snapshot.video_snapshot_id,
                    reason="late_out_of_watermark",
                    detail=(
                        f"event_time={_to_iso(video_snapshot.event_time.astimezone(timezone.utc))}, "
                        f"watermark={_to_iso(watermark_before.astimezone(timezone.utc))}"
                    ),
                    observed_at=generated_at,
                )
            )
            state["watermark"] = watermark_after
            continue
        state["watermark"] = watermark_after

        previous_video_idx = latest_video_obs_by_video.get(video_snapshot.video_id)
        if previous_video_idx is not None:
            previous = video_snapshots[previous_video_idx]
            video_snapshot = video_snapshot.model_copy(
                update={"supersedes_observation_id": previous.observation_id}
            )
            video_snapshots[previous_video_idx] = previous.model_copy(
                update={"superseded_by_observation_id": video_snapshot.observation_id}
            )
        latest_video_obs_by_video[video_snapshot.video_id] = len(video_snapshots)
        video_snapshots.append(video_snapshot)

        comment_snapshots_by_id = {
            item.comment_id: item.scraped_at.astimezone(timezone.utc)
            for item in normalized["comment_snapshots"]
        }
        for comment in normalized["comments"]:
            existing_comment = comments_by_id.get(comment.comment_id)
            comment_seen_at = comment_snapshots_by_id.get(comment.comment_id, incoming_seen_at)
            if existing_comment is None:
                comments_by_id[comment.comment_id] = comment
                comment_last_seen_at[comment.comment_id] = comment_seen_at
                continue
            if existing_comment.video_id != comment.video_id:
                errors.append(
                    f"line {line_idx}: conflicting video_id for comment '{comment.comment_id}'"
                )
                quarantine_records.append(
                    QuarantineRecord(
                        row_key=row_key,
                        source=source,
                        entity_type="comment",
                        entity_id=comment.comment_id,
                        reason="conflicting_video_id",
                        detail=f"existing={existing_comment.video_id}, incoming={comment.video_id}",
                        observed_at=generated_at,
                    )
                )
                continue
            comments_by_id[comment.comment_id] = _merge_comment(
                existing=existing_comment,
                incoming=comment,
                existing_seen_at=comment_last_seen_at[comment.comment_id],
                incoming_seen_at=comment_seen_at,
            )
            comment_last_seen_at[comment.comment_id] = max(
                comment_last_seen_at[comment.comment_id], comment_seen_at
            )

        for snapshot in normalized["comment_snapshots"]:
            comment_lateness, comment_watermark_after = _classify_lateness(
                event_time=snapshot.event_time.astimezone(timezone.utc),
                previous_watermark=state["watermark"],
            )
            comment_watermark_before = state["watermark"] or comment_watermark_after
            snapshot = snapshot.model_copy(
                update={
                    "source_watermark_time": comment_watermark_before,
                    "lateness_class": comment_lateness,
                }
            )
            state["records_seen"] += 1
            if comment_lateness == "late_within_grace":
                state["late_within_grace"] += 1
            if comment_lateness == "late_out_of_watermark":
                state["late_out_of_watermark"] += 1
                quarantine_records.append(
                    QuarantineRecord(
                        row_key=row_key,
                        source=source,
                        entity_type="comment_snapshot",
                        entity_id=snapshot.comment_snapshot_id,
                        reason="late_out_of_watermark",
                        detail=(
                            f"event_time={_to_iso(snapshot.event_time.astimezone(timezone.utc))}, "
                            f"watermark={_to_iso(comment_watermark_before.astimezone(timezone.utc))}"
                        ),
                        observed_at=generated_at,
                    )
                )
                state["watermark"] = comment_watermark_after
                continue
            state["watermark"] = comment_watermark_after

            previous_comment_idx = latest_comment_obs_by_comment.get(snapshot.comment_id)
            if previous_comment_idx is not None:
                prev_comment_obs = comment_snapshots[previous_comment_idx]
                snapshot = snapshot.model_copy(
                    update={"supersedes_observation_id": prev_comment_obs.observation_id}
                )
                comment_snapshots[previous_comment_idx] = prev_comment_obs.model_copy(
                    update={"superseded_by_observation_id": snapshot.observation_id}
                )
            latest_comment_obs_by_comment[snapshot.comment_id] = len(comment_snapshots)
            comment_snapshots.append(snapshot)

    source_watermarks: List[SourceWatermark] = []
    for source_name, state in sorted(source_state.items(), key=lambda item: item[0]):
        if state["watermark"] is None:
            continue
        source_watermarks.append(
            SourceWatermark(
                source=source_name,
                watermark_time=state["watermark"],
                records_seen=int(state["records_seen"]),
                late_within_grace=int(state["late_within_grace"]),
                late_out_of_watermark=int(state["late_out_of_watermark"]),
            )
        )

    bundle: Optional[CanonicalDatasetBundle] = None
    if not errors:
        try:
            bundle = CanonicalDatasetBundle.model_validate(
                {
                    "version": CONTRACT_VERSION,
                    "generated_at": generated_at,
                    "authors": list(authors_by_id.values()),
                    "videos": list(videos_by_id.values()),
                    "video_snapshots": video_snapshots,
                    "comments": list(comments_by_id.values()),
                    "comment_snapshots": comment_snapshots,
                    "source_watermarks": source_watermarks,
                    "quarantine_records": quarantine_records,
                }
            )
        except ValidationError as exc:
            for err in exc.errors():
                loc = ".".join(str(part) for part in err.get("loc", []))
                errors.append(f"bundle.{loc}: {err.get('msg')}")

    if bundle is not None and not errors:
        errors.extend(validate_contract_bundle(bundle))
        errors.extend(validate_as_of_time_policy(bundle, as_of_time=as_of_time))

    return RawDatasetValidationResult(
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        bundle=bundle,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return _to_iso(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_json_default)


def _schema_hash(model: type[BaseModel]) -> str:
    return _sha256_text(_canonical_json(model.model_json_schema()))


def _rows_to_dict_rows(rows: Sequence[BaseModel]) -> List[Dict[str, Any]]:
    return [item.model_dump(mode="json") for item in rows]


def _write_partitioned_entity(
    entity_name: str,
    rows: Sequence[Dict[str, Any]],
    root: Path,
) -> Dict[str, Any]:
    entity_dir = root / entity_name
    entity_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        empty_file = entity_dir / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")
        return {"format": "jsonl", "path": str(empty_file), "rows": 0}

    try:
        import pandas as pd  # type: ignore

        frame = pd.DataFrame(rows)
        if "source" in frame.columns:
            output_dir = entity_dir / "partitioned"
            output_dir.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(output_dir, index=False, partition_cols=["source"])
            return {"format": "parquet", "path": str(output_dir), "rows": len(rows)}
        output_file = entity_dir / "data.parquet"
        frame.to_parquet(output_file, index=False)
        return {"format": "parquet", "path": str(output_file), "rows": len(rows)}
    except Exception:
        output_file = entity_dir / "data.jsonl"
        with output_file.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=_json_default))
                handle.write("\n")
        return {"format": "jsonl", "path": str(output_file), "rows": len(rows)}


def _entity_payloads(bundle: CanonicalDatasetBundle) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "authors": _rows_to_dict_rows(bundle.authors),
        "videos": _rows_to_dict_rows(bundle.videos),
        "video_snapshots": _rows_to_dict_rows(bundle.video_snapshots),
        "comments": _rows_to_dict_rows(bundle.comments),
        "comment_snapshots": _rows_to_dict_rows(bundle.comment_snapshots),
        "source_watermarks": _rows_to_dict_rows(bundle.source_watermarks),
        "quarantine_records": _rows_to_dict_rows(bundle.quarantine_records),
    }


def _schema_hashes() -> Dict[str, str]:
    return {
        "CanonicalAuthor": _schema_hash(CanonicalAuthor),
        "CanonicalVideo": _schema_hash(CanonicalVideo),
        "CanonicalVideoSnapshot": _schema_hash(CanonicalVideoSnapshot),
        "CanonicalComment": _schema_hash(CanonicalComment),
        "CanonicalCommentSnapshot": _schema_hash(CanonicalCommentSnapshot),
        "SourceWatermark": _schema_hash(SourceWatermark),
        "QuarantineRecord": _schema_hash(QuarantineRecord),
        "CanonicalDatasetBundle": _schema_hash(CanonicalDatasetBundle),
    }


def build_contract_manifest(
    bundle: CanonicalDatasetBundle,
    manifest_root: Path | str,
    source_file_hashes: Optional[Dict[str, str]] = None,
    as_of_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    root = Path(manifest_root)
    root.mkdir(parents=True, exist_ok=True)

    entities_payload = _entity_payloads(bundle)
    entity_hashes = {
        key: _sha256_text(_canonical_json(value))
        for key, value in entities_payload.items()
    }
    schema_hashes = _schema_hashes()
    deterministic_seed = {
        "contract_version": CONTRACT_VERSION,
        "as_of_time": _to_iso((as_of_time or bundle.generated_at).astimezone(timezone.utc)),
        "bundle_generated_at": _to_iso(bundle.generated_at.astimezone(timezone.utc)),
        "entity_hashes": entity_hashes,
        "schema_hashes": schema_hashes,
        "source_file_hashes": source_file_hashes or {},
        "watermark_snapshot": _rows_to_dict_rows(bundle.source_watermarks),
        "quarantine_count": len(bundle.quarantine_records),
    }
    manifest_id = _sha256_text(_canonical_json(deterministic_seed))
    manifest_dir = root / manifest_id
    if manifest_dir.exists():
        manifest_path = manifest_dir / "manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        shutil.rmtree(manifest_dir)

    manifest_dir.mkdir(parents=True, exist_ok=False)
    exports_root = manifest_dir / "exports"
    exports_root.mkdir(parents=True, exist_ok=True)

    entity_files: Dict[str, Dict[str, Any]] = {}
    for entity_name, rows in entities_payload.items():
        entity_files[entity_name] = _write_partitioned_entity(entity_name, rows, exports_root)

    bundle_with_manifest = bundle.model_copy(update={"manifest_id": manifest_id})
    bundle_json = bundle_with_manifest.model_dump(mode="json")
    (manifest_dir / "bundle.json").write_text(
        json.dumps(bundle_json, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )

    payload = {
        **deterministic_seed,
        "generated_at": _to_iso(datetime.now(timezone.utc)),
        "manifest_id": manifest_id,
        "manifest_dir": str(manifest_dir),
        "entity_files": entity_files,
        "bundle_file": str(manifest_dir / "bundle.json"),
    }
    (manifest_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return payload


def load_bundle_from_manifest(manifest_ref: Path | str) -> CanonicalDatasetBundle:
    path = Path(manifest_ref)
    manifest_path = path
    if path.is_dir():
        manifest_path = path / "manifest.json"
    if manifest_path.name != "manifest.json":
        if path.exists() and path.is_file():
            manifest_path = path
        else:
            manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle_file = Path(str(manifest.get("bundle_file") or ""))
    if not bundle_file.is_absolute():
        if not bundle_file.exists():
            bundle_file = manifest_path.parent / bundle_file
    if not bundle_file.exists():
        bundle_file = manifest_path.parent / "bundle.json"
    payload = json.loads(bundle_file.read_text(encoding="utf-8"))
    bundle = CanonicalDatasetBundle.model_validate(payload)
    expected_manifest_id = str(manifest.get("manifest_id") or "")
    if expected_manifest_id and bundle.manifest_id != expected_manifest_id:
        bundle = bundle.model_copy(update={"manifest_id": expected_manifest_id})
    entity_hashes = manifest.get("entity_hashes") or {}
    schema_hashes = manifest.get("schema_hashes") or {}
    expected_seed = {
        "contract_version": manifest.get("contract_version"),
        "as_of_time": manifest.get("as_of_time"),
        "bundle_generated_at": manifest.get("bundle_generated_at"),
        "entity_hashes": entity_hashes,
        "schema_hashes": schema_hashes,
        "source_file_hashes": manifest.get("source_file_hashes") or {},
        "watermark_snapshot": manifest.get("watermark_snapshot") or [],
        "quarantine_count": int(manifest.get("quarantine_count") or 0),
    }
    recomputed_id = _sha256_text(_canonical_json(expected_seed))
    if expected_manifest_id and recomputed_id != expected_manifest_id:
        raise ValueError(
            f"Manifest id mismatch. expected='{expected_manifest_id}', recomputed='{recomputed_id}'"
        )
    recomputed_entities = {
        key: _sha256_text(_canonical_json(value))
        for key, value in _entity_payloads(bundle).items()
    }
    for key, expected_hash in entity_hashes.items():
        actual_hash = recomputed_entities.get(key)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Manifest entity hash mismatch for '{key}'. expected='{expected_hash}', actual='{actual_hash}'"
            )
    current_schema_hashes = _schema_hashes()
    for key, expected_hash in schema_hashes.items():
        actual_hash = current_schema_hashes.get(key)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Manifest schema hash mismatch for '{key}'. expected='{expected_hash}', actual='{actual_hash}'"
            )
    return bundle


__all__ = [
    "LEGACY_CONTRACT_VERSION",
    "CONTRACT_VERSION",
    "WATERMARK_GRACE_HOURS",
    "FEATURE_NAMESPACES",
    "MODELING_TRACKS",
    "SourceWatermark",
    "QuarantineRecord",
    "RecordQuality",
    "CanonicalAuthor",
    "CanonicalVideo",
    "CanonicalVideoSnapshot",
    "CanonicalComment",
    "CanonicalCommentSnapshot",
    "CanonicalDatasetBundle",
    "RawDatasetValidationResult",
    "build_contract_manifest",
    "load_bundle_from_manifest",
    "validate_contract_bundle",
    "validate_as_of_time_policy",
    "validate_feature_access_policy",
    "validate_raw_dataset_jsonl_against_contract",
]
