from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Sequence, Tuple
from urllib.parse import urlparse, urlunparse

import psycopg
from psycopg import Connection

from .client import get_database_url
from .schema import apply_schema


@dataclass
class MergeSummary:
    source_count: int
    target_db: str
    table_counts: Dict[str, int]
    started_at: datetime
    ended_at: datetime


def _redact_db_url(db_url: str) -> str:
    parsed = urlparse(db_url)
    if not parsed.username and not parsed.password:
        return db_url
    netloc = parsed.hostname or ""
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username:
        netloc = f"{parsed.username}:***@{netloc}"
    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))


def _iter_rows(
    conn: Connection,
    query: str,
    *,
    batch_size: int = 1000,
) -> Iterator[tuple]:
    cursor_name = f"merge_{uuid.uuid4().hex[:8]}"
    with conn.cursor(name=cursor_name) as cur:
        cur.itersize = batch_size
        cur.execute(query)
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                yield row


def _merge_authors(src: Connection, dst: Connection, batch_size: int) -> int:
    count = 0
    query = """
        SELECT author_id, username, display_name, bio, avatar_url, verified
        FROM authors
    """
    upsert = """
        INSERT INTO authors (author_id, username, display_name, bio, avatar_url, verified)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (author_id) DO UPDATE SET
            username = COALESCE(authors.username, EXCLUDED.username),
            display_name = COALESCE(authors.display_name, EXCLUDED.display_name),
            bio = COALESCE(authors.bio, EXCLUDED.bio),
            avatar_url = COALESCE(authors.avatar_url, EXCLUDED.avatar_url),
            verified = COALESCE(authors.verified, EXCLUDED.verified)
    """
    with dst.cursor() as cur:
        for row in _iter_rows(src, query, batch_size=batch_size):
            cur.execute(upsert, row)
            count += 1
    return count


def _merge_audios(src: Connection, dst: Connection, batch_size: int) -> int:
    count = 0
    query = """
        SELECT audio_id, audio_name, audio_author_name, is_original
        FROM audios
    """
    upsert = """
        INSERT INTO audios (audio_id, audio_name, audio_author_name, is_original)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (audio_id) DO UPDATE SET
            audio_name = COALESCE(audios.audio_name, EXCLUDED.audio_name),
            audio_author_name = COALESCE(audios.audio_author_name, EXCLUDED.audio_author_name),
            is_original = COALESCE(audios.is_original, EXCLUDED.is_original)
    """
    with dst.cursor() as cur:
        for row in _iter_rows(src, query, batch_size=batch_size):
            cur.execute(upsert, row)
            count += 1
    return count


def _merge_videos(src: Connection, dst: Connection, batch_size: int) -> int:
    count = 0
    query = """
        SELECT video_id, author_id, url, caption, duration_sec, thumbnail_url, created_at, audio_id
        FROM videos
    """
    upsert = """
        INSERT INTO videos (
            video_id, author_id, url, caption, duration_sec, thumbnail_url, created_at, audio_id
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (video_id) DO UPDATE SET
            author_id = COALESCE(videos.author_id, EXCLUDED.author_id),
            url = COALESCE(videos.url, EXCLUDED.url),
            caption = COALESCE(videos.caption, EXCLUDED.caption),
            duration_sec = COALESCE(videos.duration_sec, EXCLUDED.duration_sec),
            thumbnail_url = COALESCE(videos.thumbnail_url, EXCLUDED.thumbnail_url),
            created_at = COALESCE(videos.created_at, EXCLUDED.created_at),
            audio_id = COALESCE(videos.audio_id, EXCLUDED.audio_id)
    """
    with dst.cursor() as cur:
        for row in _iter_rows(src, query, batch_size=batch_size):
            cur.execute(upsert, row)
            count += 1
    return count


def _merge_hashtags(src: Connection, dst: Connection, batch_size: int) -> int:
    count = 0
    query = "SELECT tag FROM hashtags"
    insert_sql = "INSERT INTO hashtags (tag) VALUES (%s) ON CONFLICT (tag) DO NOTHING"
    with dst.cursor() as cur:
        for (tag,) in _iter_rows(src, query, batch_size=batch_size):
            cur.execute(insert_sql, (tag,))
            count += 1
    return count


def _merge_video_hashtags(src: Connection, dst: Connection, batch_size: int) -> int:
    count = 0
    query = """
        SELECT vh.video_id, h.tag
        FROM video_hashtags vh
        JOIN hashtags h ON h.hashtag_id = vh.hashtag_id
    """
    insert_sql = """
        INSERT INTO video_hashtags (video_id, hashtag_id)
        SELECT %s, hashtag_id
        FROM hashtags
        WHERE tag = %s
        ON CONFLICT (video_id, hashtag_id) DO NOTHING
    """
    with dst.cursor() as cur:
        for row in _iter_rows(src, query, batch_size=batch_size):
            cur.execute(insert_sql, row)
            count += 1
    return count


def _merge_scrape_runs(src: Connection, dst: Connection, batch_size: int) -> int:
    count = 0
    query = """
        SELECT scrape_run_id, source, started_at
        FROM scrape_runs
    """
    insert_sql = """
        INSERT INTO scrape_runs (scrape_run_id, source, started_at)
        VALUES (%s, %s, %s)
        ON CONFLICT (scrape_run_id) DO NOTHING
    """
    with dst.cursor() as cur:
        for row in _iter_rows(src, query, batch_size=batch_size):
            cur.execute(insert_sql, row)
            count += 1
    return count


def _merge_video_snapshots(
    src: Connection,
    dst: Connection,
    batch_size: int,
) -> Tuple[int, Dict[int, int]]:
    count = 0
    src_to_dst_snapshot_id: Dict[int, int] = {}
    query = """
        SELECT
            video_snapshot_id, video_id, scraped_at, likes, comments_count, shares, plays,
            scrape_run_id, position
        FROM video_snapshots
    """
    upsert = """
        INSERT INTO video_snapshots (
            video_id, scraped_at, likes, comments_count, shares, plays, scrape_run_id, position
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (video_id, scraped_at) DO UPDATE SET
            likes = COALESCE(video_snapshots.likes, EXCLUDED.likes),
            comments_count = COALESCE(video_snapshots.comments_count, EXCLUDED.comments_count),
            shares = COALESCE(video_snapshots.shares, EXCLUDED.shares),
            plays = COALESCE(video_snapshots.plays, EXCLUDED.plays),
            scrape_run_id = COALESCE(video_snapshots.scrape_run_id, EXCLUDED.scrape_run_id),
            position = COALESCE(video_snapshots.position, EXCLUDED.position)
        RETURNING video_snapshot_id
    """
    with dst.cursor() as cur:
        for row in _iter_rows(src, query, batch_size=batch_size):
            (
                src_snapshot_id,
                video_id,
                scraped_at,
                likes,
                comments_count,
                shares,
                plays,
                scrape_run_id,
                position,
            ) = row
            cur.execute(
                upsert,
                (
                    video_id,
                    scraped_at,
                    likes,
                    comments_count,
                    shares,
                    plays,
                    scrape_run_id,
                    position,
                ),
            )
            out = cur.fetchone()
            if out:
                src_to_dst_snapshot_id[int(src_snapshot_id)] = int(out[0])
            count += 1
    return count, src_to_dst_snapshot_id


def _merge_author_metric_snapshots(src: Connection, dst: Connection, batch_size: int) -> int:
    count = 0
    query = """
        SELECT author_id, video_id, scraped_at, follower_count, following_count, author_likes_count
        FROM author_metric_snapshots
    """
    upsert = """
        INSERT INTO author_metric_snapshots (
            author_id, video_id, scraped_at, follower_count, following_count, author_likes_count
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (author_id, video_id, scraped_at) DO UPDATE SET
            follower_count = COALESCE(author_metric_snapshots.follower_count, EXCLUDED.follower_count),
            following_count = COALESCE(author_metric_snapshots.following_count, EXCLUDED.following_count),
            author_likes_count = COALESCE(author_metric_snapshots.author_likes_count, EXCLUDED.author_likes_count)
    """
    with dst.cursor() as cur:
        for row in _iter_rows(src, query, batch_size=batch_size):
            cur.execute(upsert, row)
            count += 1
    return count


def _merge_comments(src: Connection, dst: Connection, batch_size: int) -> int:
    count = 0
    query = """
        SELECT comment_id, video_id, author_id, username, text, parent_comment_id
        FROM comments
    """
    upsert = """
        INSERT INTO comments (comment_id, video_id, author_id, username, text, parent_comment_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (comment_id) DO UPDATE SET
            video_id = COALESCE(comments.video_id, EXCLUDED.video_id),
            author_id = COALESCE(comments.author_id, EXCLUDED.author_id),
            username = COALESCE(comments.username, EXCLUDED.username),
            text = COALESCE(comments.text, EXCLUDED.text),
            parent_comment_id = COALESCE(comments.parent_comment_id, EXCLUDED.parent_comment_id)
    """
    with dst.cursor() as cur:
        for row in _iter_rows(src, query, batch_size=batch_size):
            cur.execute(upsert, row)
            count += 1
    return count


def _merge_comment_snapshots(
    src: Connection,
    dst: Connection,
    batch_size: int,
    src_to_dst_snapshot_id: Dict[int, int],
) -> int:
    count = 0
    query = """
        SELECT comment_id, video_snapshot_id, scraped_at, likes, reply_count
        FROM comment_snapshots
    """
    upsert = """
        INSERT INTO comment_snapshots (comment_id, video_snapshot_id, scraped_at, likes, reply_count)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (comment_id, video_snapshot_id) DO UPDATE SET
            likes = COALESCE(comment_snapshots.likes, EXCLUDED.likes),
            reply_count = COALESCE(comment_snapshots.reply_count, EXCLUDED.reply_count)
    """
    with dst.cursor() as cur:
        for comment_id, src_snapshot_id, scraped_at, likes, reply_count in _iter_rows(
            src,
            query,
            batch_size=batch_size,
        ):
            dst_snapshot_id = src_to_dst_snapshot_id.get(int(src_snapshot_id))
            if not dst_snapshot_id:
                continue
            cur.execute(
                upsert,
                (comment_id, dst_snapshot_id, scraped_at, likes, reply_count),
            )
            count += 1
    return count


def merge_databases(
    *,
    target_db_url: str,
    source_db_urls: Sequence[str],
    batch_size: int = 1000,
    init_target_schema: bool = True,
) -> MergeSummary:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not source_db_urls:
        raise ValueError("At least one source DB is required.")

    target_resolved = get_database_url(target_db_url)
    sources_resolved = [get_database_url(url) for url in source_db_urls]
    target_redacted = _redact_db_url(target_resolved)
    source_redacted = {_redact_db_url(url) for url in sources_resolved}
    if target_redacted in source_redacted:
        raise ValueError("Target DB cannot also be one of the source DBs.")

    if init_target_schema:
        apply_schema(db_url=target_resolved)

    started_at = datetime.now(timezone.utc)
    counts: Dict[str, int] = {
        "authors": 0,
        "audios": 0,
        "videos": 0,
        "hashtags": 0,
        "video_hashtags": 0,
        "scrape_runs": 0,
        "video_snapshots": 0,
        "author_metric_snapshots": 0,
        "comments": 0,
        "comment_snapshots": 0,
    }

    for source_url in sources_resolved:
        with psycopg.connect(source_url) as src_conn, psycopg.connect(target_resolved) as dst_conn:
            src_conn.autocommit = False
            dst_conn.autocommit = False

            counts["authors"] += _merge_authors(src_conn, dst_conn, batch_size)
            dst_conn.commit()

            counts["audios"] += _merge_audios(src_conn, dst_conn, batch_size)
            dst_conn.commit()

            counts["videos"] += _merge_videos(src_conn, dst_conn, batch_size)
            dst_conn.commit()

            counts["hashtags"] += _merge_hashtags(src_conn, dst_conn, batch_size)
            dst_conn.commit()

            counts["video_hashtags"] += _merge_video_hashtags(src_conn, dst_conn, batch_size)
            dst_conn.commit()

            counts["scrape_runs"] += _merge_scrape_runs(src_conn, dst_conn, batch_size)
            dst_conn.commit()

            merged_video_snapshots, src_to_dst_snapshot_id = _merge_video_snapshots(
                src_conn, dst_conn, batch_size
            )
            counts["video_snapshots"] += merged_video_snapshots
            dst_conn.commit()

            counts["author_metric_snapshots"] += _merge_author_metric_snapshots(
                src_conn, dst_conn, batch_size
            )
            dst_conn.commit()

            counts["comments"] += _merge_comments(src_conn, dst_conn, batch_size)
            dst_conn.commit()

            counts["comment_snapshots"] += _merge_comment_snapshots(
                src_conn,
                dst_conn,
                batch_size,
                src_to_dst_snapshot_id,
            )
            dst_conn.commit()

    ended_at = datetime.now(timezone.utc)
    return MergeSummary(
        source_count=len(sources_resolved),
        target_db=target_redacted,
        table_counts=counts,
        started_at=started_at,
        ended_at=ended_at,
    )
