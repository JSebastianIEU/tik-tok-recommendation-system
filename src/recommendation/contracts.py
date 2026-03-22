from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field, HttpUrl, ValidationError


CONTRACT_VERSION = "contract.v1"
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
    # stable dedupe
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
    return CanonicalComment.model_validate(payload)


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


class CanonicalVideoSnapshot(BaseModel):
    video_snapshot_id: str
    video_id: str
    scraped_at: datetime
    views: int = Field(ge=0)
    likes: int = Field(ge=0)
    comments_count: int = Field(ge=0)
    shares: int = Field(ge=0)
    plays: Optional[int] = Field(default=None, ge=0)
    position: Optional[int] = Field(default=None, ge=0)
    run_id: Optional[str] = None
    source: Optional[str] = None


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


class CanonicalCommentSnapshot(BaseModel):
    comment_snapshot_id: str
    comment_id: str
    video_id: str
    scraped_at: datetime
    likes: int = Field(ge=0)
    reply_count: int = Field(ge=0)
    source: Optional[str] = None


class CanonicalDatasetBundle(BaseModel):
    version: str = CONTRACT_VERSION
    generated_at: datetime
    authors: List[CanonicalAuthor]
    videos: List[CanonicalVideo]
    video_snapshots: List[CanonicalVideoSnapshot]
    comments: List[CanonicalComment]
    comment_snapshots: List[CanonicalCommentSnapshot]


class RawDatasetValidationResult(BaseModel):
    ok: bool
    errors: List[str]
    warnings: List[str]
    bundle: Optional[CanonicalDatasetBundle] = None


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
        "verified": author_obj.get("verified") if isinstance(author_obj.get("verified"), bool) else None,
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

    video = {
        "video_id": video_id,
        "author_id": author_id,
        "caption": _safe_strip(record.get("caption")),
        "hashtags": _normalize_hashtags(record.get("hashtags")),
        "keywords": _normalize_string_array(record.get("keywords")),
        "search_query": _safe_strip(record.get("search_query")) or None,
        "posted_at": posted_at,
        "video_url": _safe_strip(record.get("video_url")) or None,
        "audio_id": _safe_strip((record.get("audio") or {}).get("audio_id")) or None
        if isinstance(record.get("audio"), dict)
        else None,
        "duration_seconds": _to_non_negative_int((record.get("video_meta") or {}).get("duration_seconds"))
        if isinstance(record.get("video_meta"), dict) and (record.get("video_meta") or {}).get("duration_seconds") is not None
        else None,
        "language": _safe_strip((record.get("video_meta") or {}).get("language")).lower() or None
        if isinstance(record.get("video_meta"), dict)
        else None,
        "ingested_at": generated_at,
        "source": source,
    }

    video_snapshot = {
        "video_snapshot_id": f"{video_id}::{_to_iso(snapshot_scraped_at)}::{row_key}",
        "video_id": video_id,
        "scraped_at": snapshot_scraped_at,
        "views": views,
        "likes": likes,
        "comments_count": comments_count,
        "shares": shares,
        "run_id": _safe_strip(record.get("run_id")) or None,
        "source": source,
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

        created_at = _parse_iso(_safe_strip(comment_obj.get("created_at")))
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
            }
        )
        comment_snapshots.append(
            {
                "comment_snapshot_id": f"{comment_id}::{_to_iso(comment_snapshot_scraped_at)}::{row_key}",
                "comment_id": comment_id,
                "video_id": video_id,
                "scraped_at": comment_snapshot_scraped_at,
                "likes": _to_non_negative_int(comment_obj.get("likes")),
                "reply_count": _to_non_negative_int(comment_obj.get("reply_count")),
                "source": source,
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
    errors.extend(_dupe_errors((item.comment_id for item in bundle.comments), "comments"))
    errors.extend(
        _dupe_errors(
            (item.comment_snapshot_id for item in bundle.comment_snapshots),
            "comment_snapshots",
        )
    )

    author_ids = {item.author_id for item in bundle.authors}
    video_ids = {item.video_id for item in bundle.videos}
    comments_by_id = {item.comment_id: item for item in bundle.comments}

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

    return errors


def validate_as_of_time_policy(bundle: CanonicalDatasetBundle, as_of_time: datetime) -> List[str]:
    errors: List[str] = []
    as_of = as_of_time.astimezone(timezone.utc)

    def _check(label: str, ts: datetime) -> None:
        if ts.astimezone(timezone.utc) > as_of:
            errors.append(f"{label}: timestamp exceeds as_of_time")

    for video in bundle.videos:
        _check(f"videos.{video.video_id}.posted_at", video.posted_at)
    for snapshot in bundle.video_snapshots:
        _check(f"video_snapshots.{snapshot.video_snapshot_id}.scraped_at", snapshot.scraped_at)
    for comment in bundle.comments:
        _check(f"comments.{comment.comment_id}.created_at", comment.created_at)
    for snapshot in bundle.comment_snapshots:
        _check(
            f"comment_snapshots.{snapshot.comment_snapshot_id}.scraped_at",
            snapshot.scraped_at,
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
    authors_by_id: Dict[str, CanonicalAuthor] = {}
    videos_by_id: Dict[str, CanonicalVideo] = {}
    author_last_seen_at: Dict[str, datetime] = {}
    video_last_seen_at: Dict[str, datetime] = {}
    comments_by_id: Dict[str, CanonicalComment] = {}
    comment_last_seen_at: Dict[str, datetime] = {}
    video_snapshots: List[CanonicalVideoSnapshot] = []
    comment_snapshots: List[CanonicalCommentSnapshot] = []

    for line_idx, raw_line in enumerate(raw_jsonl.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"line {line_idx}: invalid JSON")
            continue

        if not isinstance(parsed, dict):
            errors.append(f"line {line_idx}: record must be an object")
            continue

        normalized, row_errors, row_warnings = _normalize_raw_record(
            parsed,
            generated_at=as_of_time.astimezone(timezone.utc),
            source=source,
            row_key=f"line-{line_idx}",
            strict_timestamps=strict_timestamps,
        )
        warnings.extend([f"line {line_idx}: {warn}" for warn in row_warnings])
        errors.extend([f"line {line_idx}: {err}" for err in row_errors])
        if normalized is None:
            continue

        author: CanonicalAuthor = normalized["author"]
        video: CanonicalVideo = normalized["video"]
        video_snapshot: CanonicalVideoSnapshot = normalized["video_snapshot"]
        incoming_seen_at = video_snapshot.scraped_at.astimezone(timezone.utc)

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

        comment_snapshots.extend(normalized["comment_snapshots"])

    bundle: Optional[CanonicalDatasetBundle] = None
    if not errors:
        try:
            bundle = CanonicalDatasetBundle.model_validate(
                {
                    "version": CONTRACT_VERSION,
                    "generated_at": as_of_time.astimezone(timezone.utc),
                    "authors": list(authors_by_id.values()),
                    "videos": list(videos_by_id.values()),
                    "video_snapshots": video_snapshots,
                    "comments": list(comments_by_id.values()),
                    "comment_snapshots": comment_snapshots,
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


__all__ = [
    "CONTRACT_VERSION",
    "FEATURE_NAMESPACES",
    "MODELING_TRACKS",
    "CanonicalAuthor",
    "CanonicalVideo",
    "CanonicalVideoSnapshot",
    "CanonicalComment",
    "CanonicalCommentSnapshot",
    "CanonicalDatasetBundle",
    "RawDatasetValidationResult",
    "validate_contract_bundle",
    "validate_as_of_time_policy",
    "validate_feature_access_policy",
    "validate_raw_dataset_jsonl_against_contract",
]
