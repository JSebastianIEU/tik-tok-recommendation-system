"""Local corpus loader for the Python recommender service.

Reads the same artifact-bundle JSON that the Node gateway uses and converts
it into candidate dicts compatible with ``RecommenderRuntime.recommend()``.
Results are cached in-process and invalidated on file mtime change.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.common.config import settings

MAX_COMMENTS_PER_VIDEO = 8


def _str(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [s.strip() for s in value if isinstance(s, str) and s.strip()]
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    return []


def _num(value: Any, fallback: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else fallback
    if isinstance(value, str):
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else fallback
        except ValueError:
            pass
    return fallback


def _parse_iso(value: Any) -> Optional[str]:
    text = _str(value)
    if not text:
        return None
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
        return text
    except (ValueError, TypeError):
        return None


def _parse_timestamp_ms(value: Any) -> float:
    text = _str(value)
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000.0
    except (ValueError, TypeError):
        return 0.0


class _CachedCorpus:
    __slots__ = ("records", "mtime_ns", "source_path")

    def __init__(self, records: List[Dict[str, Any]], mtime_ns: int, source_path: str) -> None:
        self.records = records
        self.mtime_ns = mtime_ns
        self.source_path = source_path


_cache: Optional[_CachedCorpus] = None


def _build_candidates_from_bundle(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    authors_by_id: Dict[str, Dict[str, Any]] = {}
    for author in (bundle.get("authors") or []):
        if not isinstance(author, dict):
            continue
        author_id = _str(author.get("author_id"))
        if author_id:
            authors_by_id[author_id] = author

    latest_snapshot: Dict[str, Dict[str, Any]] = {}
    for snap in (bundle.get("video_snapshots") or []):
        if not isinstance(snap, dict):
            continue
        vid = _str(snap.get("video_id"))
        if not vid:
            continue
        ts = max(
            _parse_timestamp_ms(snap.get("event_time")),
            _parse_timestamp_ms(snap.get("scraped_at")),
            _parse_timestamp_ms(snap.get("ingested_at")),
        )
        existing = latest_snapshot.get(vid)
        if existing is None:
            latest_snapshot[vid] = snap
        else:
            existing_ts = max(
                _parse_timestamp_ms(existing.get("event_time")),
                _parse_timestamp_ms(existing.get("scraped_at")),
                _parse_timestamp_ms(existing.get("ingested_at")),
            )
            if ts >= existing_ts:
                latest_snapshot[vid] = snap

    comments_by_video: Dict[str, List[Tuple[float, str]]] = {}
    for comment in (bundle.get("comments") or []):
        if not isinstance(comment, dict):
            continue
        vid = _str(comment.get("video_id"))
        text = _str(comment.get("text"))
        if not vid or not text:
            continue
        ts = _parse_timestamp_ms(comment.get("created_at"))
        comments_by_video.setdefault(vid, []).append((ts, text))

    for bucket in comments_by_video.values():
        bucket.sort(key=lambda item: item[0], reverse=True)

    records: List[Dict[str, Any]] = []
    for video in (bundle.get("videos") or []):
        if not isinstance(video, dict):
            continue
        video_id = _str(video.get("video_id"))
        if not video_id:
            continue

        author_id = _str(video.get("author_id"))
        author_row = authors_by_id.get(author_id) if author_id else None
        caption = _str(video.get("caption"))
        hashtags = _str_list(video.get("hashtags"))
        keywords = _str_list(video.get("keywords"))
        search_query = _str(video.get("search_query"))
        language = _str(video.get("language")).lower() or None
        posted_at = _parse_iso(video.get("posted_at"))
        duration_seconds = _num(video.get("duration_seconds"))

        comment_bucket = comments_by_video.get(video_id, [])
        comment_texts = [text for _, text in comment_bucket[:MAX_COMMENTS_PER_VIDEO]]

        text_parts = [caption] + hashtags + keywords
        text = " ".join(part for part in text_parts if part)

        records.append({
            "candidate_id": video_id,
            "video_id": video_id,
            "caption": caption,
            "text": text,
            "hashtags": hashtags,
            "keywords": keywords,
            "topic_key": search_query or None,
            "author_id": author_id or (
                _str(author_row.get("username")) if author_row else None
            ) or "unknown",
            "as_of_time": posted_at,
            "posted_at": posted_at,
            "language": language,
            "locale": None,
            "content_type": None,
            "duration_seconds": int(duration_seconds) if duration_seconds > 0 else None,
            "signal_hints": {},
            "_comments": comment_texts,
            "_search_query": search_query,
        })

    records.sort(
        key=lambda r: (
            _parse_timestamp_ms(r.get("posted_at")) or 0.0,
            0.0,
        ),
        reverse=True,
    )
    return records


def _load_corpus(bundle_path: str) -> List[Dict[str, Any]]:
    global _cache
    resolved = os.path.abspath(bundle_path)
    try:
        mtime_ns = os.stat(resolved).st_mtime_ns
    except OSError:
        return []
    if _cache is not None and _cache.source_path == resolved and _cache.mtime_ns == mtime_ns:
        return _cache.records
    try:
        with open(resolved, encoding="utf-8") as fh:
            bundle = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []
    records = _build_candidates_from_bundle(bundle)
    _cache = _CachedCorpus(records=records, mtime_ns=mtime_ns, source_path=resolved)
    return records


class CorpusScopeSpec:
    """Lightweight value object describing which subset of the corpus to return."""

    __slots__ = (
        "topic_key",
        "language",
        "locale",
        "content_type",
        "max_candidates",
        "exclude_video_ids",
    )

    def __init__(
        self,
        *,
        topic_key: Optional[str] = None,
        language: Optional[str] = None,
        locale: Optional[str] = None,
        content_type: Optional[str] = None,
        max_candidates: int = 500,
        exclude_video_ids: Optional[Sequence[str]] = None,
    ) -> None:
        self.topic_key = topic_key.strip().lower() if isinstance(topic_key, str) and topic_key.strip() else None
        self.language = language.strip().lower() if isinstance(language, str) and language.strip() else None
        self.locale = locale.strip().lower() if isinstance(locale, str) and locale.strip() else None
        self.content_type = content_type.strip().lower() if isinstance(content_type, str) and content_type.strip() else None
        self.max_candidates = max(1, min(2000, int(max_candidates)))
        self.exclude_video_ids = frozenset(
            s.strip() for s in (exclude_video_ids or []) if isinstance(s, str) and s.strip()
        )


def _topic_matches(record: Dict[str, Any], topic_key: str) -> bool:
    topic_lower = topic_key.lower()
    if (record.get("topic_key") or "").lower() == topic_lower:
        return True
    if (record.get("_search_query") or "").lower() == topic_lower:
        return True
    for tag in (record.get("hashtags") or []):
        if tag.lower().lstrip("#") == topic_lower:
            return True
    for kw in (record.get("keywords") or []):
        if kw.lower() == topic_lower:
            return True
    return False


class CorpusResolver:
    """Resolves candidates from a local artifact bundle, applying scope filters."""

    def __init__(self, bundle_path: Optional[str] = None) -> None:
        self._bundle_path = (bundle_path or "").strip() or settings.recommender_corpus_bundle_path

    @property
    def available(self) -> bool:
        return bool(self._bundle_path) and os.path.isfile(self._bundle_path)

    def resolve_candidates(self, scope: CorpusScopeSpec) -> List[Dict[str, Any]]:
        if not self._bundle_path:
            return []
        all_records = _load_corpus(self._bundle_path)
        if not all_records:
            return []

        filtered: List[Dict[str, Any]] = []
        for record in all_records:
            vid = record.get("candidate_id") or record.get("video_id") or ""
            if vid in scope.exclude_video_ids:
                continue
            if scope.language and (record.get("language") or "") and record["language"] != scope.language:
                continue
            if scope.locale and (record.get("locale") or "") and record["locale"] != scope.locale:
                continue
            if scope.content_type and (record.get("content_type") or "") and record["content_type"] != scope.content_type:
                continue
            if scope.topic_key and not _topic_matches(record, scope.topic_key):
                continue
            filtered.append(record)
            if len(filtered) >= scope.max_candidates:
                break

        return filtered
