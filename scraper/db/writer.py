from __future__ import annotations

import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

from psycopg import Connection, InterfaceError, OperationalError

from .client import connect, get_connection


@dataclass
class ScrapeContext:
    """
    Helper object describing a logical scrape run.

    This aligns with the `scrape_runs` table and lets callers tag all
    inserted snapshots with a shared run id + source label.
    """

    scrape_run_id: uuid.UUID
    source: str
    started_at: datetime


class BatchedRecordWriter:
    """Keep one DB connection open and commit every N writes."""

    def __init__(self, *, db_url: Optional[str] = None, commit_every: int = 50) -> None:
        self._db_url = db_url
        self._commit_every = max(1, int(commit_every))
        self._write_retries = max(0, int(os.getenv("SCRAPER_DB_WRITE_RETRIES", "1")))
        self._conn: Optional[Connection] = None
        self._pending = 0

    def __enter__(self) -> "BatchedRecordWriter":
        self._conn = connect(self._db_url)
        return self

    def _close_conn(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            logging.warning("db_close_failed", exc_info=True)

    def _reconnect(self) -> None:
        self._close_conn()
        self._conn = connect(self._db_url)
        self._pending = 0

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._conn is None:
            return
        try:
            if exc:
                try:
                    self._conn.rollback()
                except Exception:  # noqa: BLE001
                    logging.warning("db_rollback_failed_on_exit", exc_info=True)
            elif self._pending > 0:
                try:
                    self._conn.commit()
                except Exception:  # noqa: BLE001
                    logging.warning("db_commit_failed_on_exit", exc_info=True)
        finally:
            self._close_conn()
            self._conn = None
            self._pending = 0

    def write(
        self,
        normalized: Dict[str, Any],
        *,
        scrape_ctx: Optional[ScrapeContext] = None,
        position: Optional[int] = None,
        skip_existing: bool = False,
    ) -> bool:
        if self._conn is None:
            raise RuntimeError("BatchedRecordWriter must be used as a context manager.")
        attempts = self._write_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                wrote = write_normalized_record(
                    normalized,
                    conn=self._conn,
                    scrape_ctx=scrape_ctx,
                    position=position,
                    skip_existing=skip_existing,
                )
                if wrote:
                    self._pending += 1
                    if self._pending >= self._commit_every:
                        self._conn.commit()
                        self._pending = 0
                return wrote
            except (OperationalError, InterfaceError):
                if attempt >= attempts:
                    raise
                logging.warning(
                    "db_write_retry attempt=%s/%s due_to_connection_error",
                    attempt,
                    attempts - 1,
                )
                self._reconnect()
        # Unreachable, but keeps static analyzers happy.
        return False

    def rollback(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.rollback()
        except Exception:  # noqa: BLE001
            logging.warning("db_rollback_failed", exc_info=True)
            self._reconnect()
        self._pending = 0


def load_existing_video_ids(db_url: Optional[str] = None) -> set[str]:
    """
    Load all video_ids from the videos table.

    Kept for backwards compatibility. For large datasets, prefer DB-side checks
    via `skip_existing=True` in `write_normalized_record`.
    """
    with get_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT video_id FROM videos")
            return {row[0] for row in cur.fetchall() if row[0]}


def create_scrape_run(
    source: str,
    *,
    db_url: Optional[str] = None,
) -> ScrapeContext:
    """Insert a new scrape_runs row and return its context."""
    run_id = uuid.uuid4()
    started_at = datetime.utcnow()
    with get_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO scrape_runs (scrape_run_id, source, started_at)
                VALUES (%s, %s, %s)
                """,
                (str(run_id), source, started_at),
            )
    return ScrapeContext(scrape_run_id=run_id, source=source, started_at=started_at)


def _upsert_author(conn: Connection, author: Dict[str, Any]) -> None:
    if not author:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO authors (author_id, username, display_name, bio, avatar_url, verified)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (author_id) DO UPDATE SET
                username = EXCLUDED.username,
                display_name = EXCLUDED.display_name,
                bio = EXCLUDED.bio,
                avatar_url = EXCLUDED.avatar_url,
                verified = EXCLUDED.verified
            """,
            (
                author.get("author_id"),
                author.get("username"),
                author.get("display_name"),
                author.get("bio"),
                author.get("avatar_url"),
                author.get("verified"),
            ),
        )


def _upsert_audio(conn: Connection, audio: Dict[str, Any]) -> Optional[str]:
    if not audio:
        return None
    audio_id = audio.get("audio_id")
    if not audio_id:
        return None
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO audios (audio_id, audio_name, audio_author_name, is_original)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (audio_id) DO UPDATE SET
                audio_name = EXCLUDED.audio_name,
                audio_author_name = EXCLUDED.audio_author_name,
                is_original = EXCLUDED.is_original
            """,
            (
                audio_id,
                audio.get("audio_name"),
                audio.get("audio_author_name"),
                audio.get("is_original"),
            ),
        )
    return str(audio_id)


def _normalize_hashtag_tags(tags: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        if not raw:
            continue
        tag = raw.lstrip("#").strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def _ensure_hashtags(conn: Connection, tags: Sequence[str]) -> List[int]:
    """Ensure all hashtag tags exist and return their ids."""
    normalized_tags = _normalize_hashtag_tags(tags)
    if not normalized_tags:
        return []

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO hashtags (tag)
            SELECT tag
            FROM unnest(%s::text[]) AS t(tag)
            ON CONFLICT (tag) DO NOTHING
            """,
            (normalized_tags,),
        )
        cur.execute(
            "SELECT hashtag_id, tag FROM hashtags WHERE tag = ANY(%s::text[])",
            (normalized_tags,),
        )
        tag_to_id = {str(tag): int(hashtag_id) for hashtag_id, tag in cur.fetchall()}
    return [tag_to_id[tag] for tag in normalized_tags if tag in tag_to_id]


def _link_video_hashtags(conn: Connection, video_id: str, hashtag_ids: Sequence[int]) -> None:
    if not video_id or not hashtag_ids:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO video_hashtags (video_id, hashtag_id)
            SELECT %s, hashtag_id
            FROM unnest(%s::int[]) AS h(hashtag_id)
            ON CONFLICT (video_id, hashtag_id) DO NOTHING
            """,
            (video_id, list(hashtag_ids)),
        )


def _video_exists(conn: Connection, video_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM videos WHERE video_id = %s LIMIT 1", (video_id,))
        return cur.fetchone() is not None


def _upsert_video(
    conn: Connection,
    video: Dict[str, Any],
    author_id: str,
    audio_id: Optional[str],
) -> str:
    video_id = video.get("video_id")
    if not video_id:
        raise ValueError("video.video_id is required")

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO videos (
                video_id, author_id, url, caption, duration_sec,
                thumbnail_url, created_at, audio_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (video_id) DO UPDATE SET
                author_id = EXCLUDED.author_id,
                url = EXCLUDED.url,
                caption = EXCLUDED.caption,
                duration_sec = EXCLUDED.duration_sec,
                thumbnail_url = EXCLUDED.thumbnail_url,
                created_at = EXCLUDED.created_at,
                audio_id = EXCLUDED.audio_id
            """,
            (
                video_id,
                author_id,
                video.get("url"),
                video.get("caption"),
                video.get("duration_sec"),
                video.get("thumbnail_url"),
                video.get("created_at"),
                audio_id,
            ),
        )
    return str(video_id)


def _insert_video_snapshot(
    conn: Connection,
    video_id: str,
    metrics: Dict[str, Any],
    scraped_at: datetime,
    scrape_ctx: Optional[ScrapeContext] = None,
    position: Optional[int] = None,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO video_snapshots (
                video_id, scraped_at, likes, comments_count, shares, plays,
                scrape_run_id, position
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (video_id, scraped_at) DO UPDATE SET
                likes = EXCLUDED.likes,
                comments_count = EXCLUDED.comments_count,
                shares = EXCLUDED.shares,
                plays = EXCLUDED.plays,
                scrape_run_id = COALESCE(EXCLUDED.scrape_run_id, video_snapshots.scrape_run_id),
                position = COALESCE(EXCLUDED.position, video_snapshots.position)
            RETURNING video_snapshot_id
            """,
            (
                video_id,
                scraped_at,
                metrics.get("likes"),
                metrics.get("comments_count"),
                metrics.get("shares"),
                metrics.get("plays"),
                str(scrape_ctx.scrape_run_id) if scrape_ctx else None,
                position,
            ),
        )
        row = cur.fetchone()
    return int(row[0])


def _insert_author_metric_snapshot(
    conn: Connection,
    author_snapshot: Dict[str, Any],
    scraped_at: datetime,
) -> None:
    if not author_snapshot:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO author_metric_snapshots (
                author_id, video_id, scraped_at,
                follower_count, following_count, author_likes_count
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (author_id, video_id, scraped_at) DO UPDATE SET
                follower_count = EXCLUDED.follower_count,
                following_count = EXCLUDED.following_count,
                author_likes_count = EXCLUDED.author_likes_count
            """,
            (
                author_snapshot.get("author_id"),
                author_snapshot.get("video_id"),
                scraped_at,
                author_snapshot.get("follower_count"),
                author_snapshot.get("following_count"),
                author_snapshot.get("author_likes_count"),
            ),
        )


def _upsert_comments_and_snapshots(
    conn: Connection,
    video_id: str,
    comments: Iterable[Dict[str, Any]],
    video_snapshot_id: int,
    scraped_at: datetime,
) -> None:
    for c in comments:
        comment_id = c.get("comment_id") or c.get("id")
        if not comment_id:
            # Derive a deterministic id when source comment id is unavailable.
            text = (c.get("text") or "").strip().lower()
            username = (c.get("username") or "").strip().lower()
            parent_id = (c.get("parent_comment_id") or "").strip().lower()
            seed = f"{video_id}|{username}|{parent_id}|{text[:256]}"
            digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]
            comment_id = f"gen:{digest}"

        comment_author_id = c.get("author_id")
        comment_username = c.get("username")
        if comment_author_id:
            # Comment authors are often not the same as the video author.
            # Upsert a minimal author row first to satisfy FK constraints.
            _upsert_author(
                conn,
                {
                    "author_id": str(comment_author_id),
                    "username": comment_username,
                    "display_name": None,
                    "bio": None,
                    "avatar_url": None,
                    "verified": None,
                },
            )

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO comments (
                    comment_id, video_id, author_id, username, text, parent_comment_id
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (comment_id) DO UPDATE SET
                    video_id = EXCLUDED.video_id,
                    author_id = COALESCE(EXCLUDED.author_id, comments.author_id),
                    username = COALESCE(EXCLUDED.username, comments.username),
                    text = EXCLUDED.text,
                    parent_comment_id = COALESCE(EXCLUDED.parent_comment_id, comments.parent_comment_id)
                """,
                (
                    comment_id,
                    video_id,
                    str(comment_author_id) if comment_author_id is not None else None,
                    c.get("username"),
                    c.get("text"),
                    c.get("parent_comment_id"),
                ),
            )
            cur.execute(
                """
                INSERT INTO comment_snapshots (
                    comment_id, video_snapshot_id, scraped_at, likes, reply_count
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (comment_id, video_snapshot_id) DO UPDATE SET
                    likes = EXCLUDED.likes,
                    reply_count = EXCLUDED.reply_count
                """,
                (
                    comment_id,
                    video_snapshot_id,
                    scraped_at,
                    c.get("likes"),
                    c.get("reply_count"),
                ),
            )


def _write_with_connection(
    conn: Connection,
    normalized: Dict[str, Any],
    *,
    scrape_ctx: Optional[ScrapeContext] = None,
    position: Optional[int] = None,
    skip_existing: bool = False,
    existing_video_ids: Optional[set[str]] = None,
) -> bool:
    author = normalized.get("author") or {}
    video = normalized.get("video") or {}
    author_snapshot = normalized.get("authorMetricSnapshot") or {}
    comments = normalized.get("comments") or []

    raw_video_id = video.get("video_id")
    video_id = str(raw_video_id) if raw_video_id is not None else None
    if skip_existing and video_id:
        if existing_video_ids is not None and video_id in existing_video_ids:
            return False
        if existing_video_ids is None and _video_exists(conn, video_id):
            return False

    hashtags = video.get("hashtags") or []
    metrics = {
        "likes": video.get("likes"),
        "comments_count": video.get("comments_count"),
        "shares": video.get("shares"),
        "plays": video.get("plays"),
    }
    scraped_at_raw = video.get("scraped_at") or datetime.utcnow().isoformat()
    scraped_at = (
        datetime.fromisoformat(scraped_at_raw.replace("Z", "+00:00"))
        if isinstance(scraped_at_raw, str)
        else scraped_at_raw
    )

    _upsert_author(conn, author)
    audio_id = _upsert_audio(
        conn,
        {
            "audio_id": video.get("audio_id"),
            "audio_name": video.get("audio_name"),
            "audio_author_name": None,
            "is_original": None,
        },
    )

    author_id = author.get("author_id")
    if not author_id:
        raise ValueError("normalized.author.author_id is required")

    persisted_video_id = _upsert_video(conn, video, author_id=author_id, audio_id=audio_id)
    hashtag_ids = _ensure_hashtags(conn, hashtags)
    _link_video_hashtags(conn, persisted_video_id, hashtag_ids)

    video_snapshot_id = _insert_video_snapshot(
        conn,
        video_id=persisted_video_id,
        metrics=metrics,
        scraped_at=scraped_at,
        scrape_ctx=scrape_ctx,
        position=position,
    )
    _insert_author_metric_snapshot(conn, author_snapshot, scraped_at=scraped_at)
    _upsert_comments_and_snapshots(
        conn,
        video_id=persisted_video_id,
        comments=comments,
        video_snapshot_id=video_snapshot_id,
        scraped_at=scraped_at,
    )
    return True


def write_normalized_record(
    normalized: Dict[str, Any],
    *,
    db_url: Optional[str] = None,
    conn: Optional[Connection] = None,
    scrape_ctx: Optional[ScrapeContext] = None,
    position: Optional[int] = None,
    skip_existing: bool = False,
    existing_video_ids: Optional[set[str]] = None,
) -> bool:
    """
    Persist a single normalized record into Postgres.

    Returns True if a write happened, False if the record was skipped.
    """
    if conn is not None:
        return _write_with_connection(
            conn,
            normalized,
            scrape_ctx=scrape_ctx,
            position=position,
            skip_existing=skip_existing,
            existing_video_ids=existing_video_ids,
        )

    with get_connection(db_url) as managed_conn:
        return _write_with_connection(
            managed_conn,
            normalized,
            scrape_ctx=scrape_ctx,
            position=position,
            skip_existing=skip_existing,
            existing_video_ids=existing_video_ids,
        )


def dry_run_print_sql(normalized: Dict[str, Any]) -> None:
    """
    Helper for development: print a high-level summary of what would be written.

    This avoids requiring a live Postgres instance just to inspect the mapping.
    """
    author = normalized.get("author") or {}
    video = normalized.get("video") or {}
    comments = normalized.get("comments") or []
    print(
        f"[DRY RUN] would upsert author={author.get('author_id')}, "
        f"video={video.get('video_id')} ({video.get('url')}) "
        f"with {len(comments)} comments"
    )
