from __future__ import annotations

from datetime import datetime
from typing import Any


def _parse_iso(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).isoformat()
    except ValueError:
        return text


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    return text or None


def _dedupe_latest(rows: list[dict[str, Any]], *, key: str, ts_field: str) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        k = row.get(key)
        if k is None:
            continue
        sk = str(k)
        current = by_key.get(sk)
        if current is None:
            by_key[sk] = row
            continue
        prev_ts = str(current.get(ts_field) or "")
        new_ts = str(row.get(ts_field) or "")
        if new_ts >= prev_ts:
            by_key[sk] = row
    return list(by_key.values())


def _silver_videos(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = _dedupe_latest(rows, key="video_id", ts_field="created_at")
    out: list[dict[str, Any]] = []
    for row in deduped:
        likes = _to_int(row.get("likes"))
        comments_count = _to_int(row.get("comments_count"))
        shares = _to_int(row.get("shares"))
        plays = _to_int(row.get("plays"))
        engagement_rate = None
        if plays and plays > 0:
            engagement_rate = ((likes or 0) + (comments_count or 0) + (shares or 0)) / plays

        tags = row.get("hashtags")
        hashtags = None
        if isinstance(tags, list):
            hashtags = sorted({str(t).strip() for t in tags if str(t).strip()})

        out.append(
            {
                **row,
                "video_id": str(row.get("video_id") or ""),
                "created_at": _parse_iso(row.get("created_at")),
                "snapshot_scraped_at": _parse_iso(row.get("snapshot_scraped_at")),
                "duration_sec": _to_int(row.get("duration_sec")),
                "likes": likes,
                "comments_count": comments_count,
                "shares": shares,
                "plays": plays,
                "hashtags": hashtags,
                "engagement_rate": engagement_rate,
            }
        )
    out.sort(key=lambda r: (str(r.get("created_at") or ""), str(r.get("video_id") or "")), reverse=True)
    return out


def _normalize_comment_obj(comment: dict[str, Any]) -> dict[str, Any]:
    text = _normalize_text(comment.get("text"))
    parent = comment.get("parent_comment_id")
    return {
        **comment,
        "comment_id": str(comment.get("comment_id") or ""),
        "scraped_at": _parse_iso(comment.get("scraped_at")),
        "likes": _to_int(comment.get("likes")),
        "reply_count": _to_int(comment.get("reply_count")),
        "text": text,
        "text_length": len(text or ""),
        "is_reply": bool(parent),
    }


def _silver_comments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = _dedupe_latest(rows, key="comment_id", ts_field="row_created_at")
    out: list[dict[str, Any]] = []
    for row in deduped:
        text = _normalize_text(row.get("text"))
        parent = row.get("parent_comment_id")
        out.append(
            {
                **row,
                "comment_id": str(row.get("comment_id") or ""),
                "video_id": str(row.get("video_id") or ""),
                "row_created_at": _parse_iso(row.get("row_created_at")),
                "video_created_at": _parse_iso(row.get("video_created_at")),
                "comment_scraped_at": _parse_iso(row.get("comment_scraped_at")),
                "comment_likes": _to_int(row.get("comment_likes")),
                "reply_count": _to_int(row.get("reply_count")),
                "text": text,
                "text_length": len(text or ""),
                "is_reply": bool(parent),
            }
        )
    out.sort(
        key=lambda r: (str(r.get("row_created_at") or ""), str(r.get("comment_id") or "")),
        reverse=True,
    )
    return out


def _silver_authors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = _dedupe_latest(rows, key="author_id", ts_field="row_created_at")
    out: list[dict[str, Any]] = []
    for row in deduped:
        videos_count = _to_int(row.get("videos_count")) or 0
        comments_count = _to_int(row.get("comments_count")) or 0
        comments_per_video = None
        if videos_count > 0:
            comments_per_video = comments_count / videos_count
        out.append(
            {
                **row,
                "author_id": str(row.get("author_id") or ""),
                "row_created_at": _parse_iso(row.get("row_created_at")),
                "latest_video_created_at": _parse_iso(row.get("latest_video_created_at")),
                "videos_count": videos_count,
                "comments_count": comments_count,
                "avg_video_plays": _to_float(row.get("avg_video_plays")),
                "comments_per_video": comments_per_video,
            }
        )
    out.sort(
        key=lambda r: (str(r.get("row_created_at") or ""), str(r.get("author_id") or "")),
        reverse=True,
    )
    return out


def _silver_full(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = _dedupe_latest(rows, key="video_id", ts_field="created_at")
    out: list[dict[str, Any]] = []
    for row in deduped:
        comments = row.get("comments") if isinstance(row.get("comments"), list) else []
        normalized_comments = [_normalize_comment_obj(c) for c in comments if isinstance(c, dict)]

        likes = _to_int(row.get("likes"))
        comments_count = _to_int(row.get("comments_count"))
        shares = _to_int(row.get("shares"))
        plays = _to_int(row.get("plays"))
        engagement_rate = None
        if plays and plays > 0:
            engagement_rate = ((likes or 0) + (comments_count or 0) + (shares or 0)) / plays

        out.append(
            {
                **row,
                "video_id": str(row.get("video_id") or ""),
                "created_at": _parse_iso(row.get("created_at")),
                "snapshot_scraped_at": _parse_iso(row.get("snapshot_scraped_at")),
                "likes": likes,
                "comments_count": comments_count,
                "shares": shares,
                "plays": plays,
                "engagement_rate": engagement_rate,
                "comments": normalized_comments,
                "comment_count_extracted": len(normalized_comments),
            }
        )

    out.sort(key=lambda r: (str(r.get("created_at") or ""), str(r.get("video_id") or "")), reverse=True)
    return out


def build_silver_dataset(dataset: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if dataset == "videos":
        return _silver_videos(rows)
    if dataset == "comments":
        return _silver_comments(rows)
    if dataset == "authors":
        return _silver_authors(rows)
    if dataset == "full":
        return _silver_full(rows)
    raise ValueError(f"Unsupported dataset: {dataset}")
