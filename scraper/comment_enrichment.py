from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from TikTokApi import TikTokApi

from scraper.db.client import connect, get_database_url
from scraper.db.comment_lineage import ensure_comment_lineage_columns
from scraper.db.writer import write_comments_for_existing_video
from scraper.scrape_tiktok_sample import (
    _create_api_session,
    fetch_comments_for_video,
    get_ms_token,
)


@dataclass
class CommentEnrichmentSummary:
    candidates_selected: int = 0
    processed: int = 0
    videos_with_comments: int = 0
    comments_written: int = 0
    video_fetch_errors: int = 0
    reply_fetch_errors: int = 0
    persist_failed: int = 0
    exhausted: int = 0
    seeded_jobs: int = 0
    job_status_counts: dict[str, int] | None = None


def _ensure_jobs_table(db_url: str) -> None:
    sql = """
    CREATE TABLE IF NOT EXISTS comment_enrichment_jobs (
        video_id TEXT PRIMARY KEY REFERENCES videos(video_id) ON DELETE CASCADE,
        status TEXT NOT NULL DEFAULT 'pending',
        attempt_count INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        last_attempt_at TIMESTAMPTZ,
        next_retry_at TIMESTAMPTZ,
        completed_at TIMESTAMPTZ,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_comment_enrichment_jobs_status ON comment_enrichment_jobs(status);
    CREATE INDEX IF NOT EXISTS idx_comment_enrichment_jobs_next_retry ON comment_enrichment_jobs(next_retry_at);
    """
    with connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def _recover_stale_running_jobs(db_url: str, stale_minutes: int = 60) -> None:
    with connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE comment_enrichment_jobs
                SET status = 'failed',
                    last_error = COALESCE(last_error, 'recovered_stale_running_job'),
                    next_retry_at = NOW(),
                    updated_at = NOW()
                WHERE status = 'running'
                  AND last_attempt_at IS NOT NULL
                  AND last_attempt_at < NOW() - (%s * INTERVAL '1 minute')
                """,
                (max(1, int(stale_minutes)),),
            )
        conn.commit()


def _seed_jobs_from_videos(
    db_url: str,
    *,
    limit: int,
    include_existing: bool,
    max_existing_comments: int,
) -> int:
    params: list[int] = []
    if include_existing:
        where_clause = "TRUE"
    elif max_existing_comments <= 0:
        where_clause = "NOT EXISTS (SELECT 1 FROM comments c WHERE c.video_id = v.video_id)"
    else:
        where_clause = "(SELECT COUNT(*) FROM comments c WHERE c.video_id = v.video_id) <= %s"
        params.append(max_existing_comments)
    sql = f"""
        INSERT INTO comment_enrichment_jobs (video_id, status, updated_at)
        SELECT v.video_id, 'pending', NOW()
        FROM videos v
        LEFT JOIN comment_enrichment_jobs j ON j.video_id = v.video_id
        WHERE j.video_id IS NULL
          AND {where_clause}
        ORDER BY v.created_at DESC NULLS LAST
        LIMIT %s
        ON CONFLICT (video_id) DO NOTHING
    """
    params.append(max(1, int(limit)))

    with connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            inserted = cur.rowcount or 0
        conn.commit()
    return inserted


def _get_job_status_counts(conn) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT status, COUNT(*)::INT AS cnt
            FROM comment_enrichment_jobs
            GROUP BY status
            """
        )
        rows = cur.fetchall()
    return {str(status): int(cnt) for status, cnt in rows}


def _select_candidates_from_jobs(
    conn,
    *,
    limit: int,
    max_attempts_per_video: int,
) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT video_id
            FROM comment_enrichment_jobs
            WHERE status IN ('pending', 'failed')
              AND attempt_count < %s
              AND (next_retry_at IS NULL OR next_retry_at <= NOW())
            ORDER BY COALESCE(last_attempt_at, TIMESTAMPTZ 'epoch') ASC
            LIMIT %s
            """,
            (max(1, int(max_attempts_per_video)), max(1, int(limit))),
        )
        return [str(row[0]) for row in cur.fetchall() if row and row[0]]


def _mark_running(conn, video_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE comment_enrichment_jobs
            SET status = 'running',
                attempt_count = attempt_count + 1,
                last_attempt_at = NOW(),
                updated_at = NOW()
            WHERE video_id = %s
            RETURNING attempt_count
            """,
            (video_id,),
        )
        row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 1


def _mark_done(conn, video_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE comment_enrichment_jobs
            SET status = 'done',
                last_error = NULL,
                next_retry_at = NULL,
                completed_at = NOW(),
                updated_at = NOW()
            WHERE video_id = %s
            """,
            (video_id,),
        )


def _mark_retry_or_exhausted(
    conn,
    *,
    video_id: str,
    attempt_count: int,
    max_attempts_per_video: int,
    last_error: str,
    retry_backoff_base_sec: float,
) -> str:
    if attempt_count >= max(1, int(max_attempts_per_video)):
        status = "exhausted"
        next_retry_sql = "NULL"
        params = (status, last_error, video_id)
        sql = f"""
            UPDATE comment_enrichment_jobs
            SET status = %s,
                last_error = %s,
                next_retry_at = {next_retry_sql},
                updated_at = NOW()
            WHERE video_id = %s
        """
    else:
        status = "failed"
        backoff = max(1.0, float(retry_backoff_base_sec)) * (2 ** max(0, attempt_count - 1))
        backoff = min(backoff, 21600.0)  # 6h cap
        params = (status, last_error, backoff, video_id)
        sql = """
            UPDATE comment_enrichment_jobs
            SET status = %s,
                last_error = %s,
                next_retry_at = NOW() + (%s * INTERVAL '1 second'),
                updated_at = NOW()
            WHERE video_id = %s
        """

    with conn.cursor() as cur:
        cur.execute(sql, params)
    return status


async def _run_enrichment(args: argparse.Namespace) -> int:
    db_url = get_database_url(args.db_url)
    ms_token = get_ms_token(args.ms_token)
    ensure_comment_lineage_columns(db_url=db_url)
    _ensure_jobs_table(db_url)
    _recover_stale_running_jobs(db_url, stale_minutes=args.stale_running_minutes)

    summary = CommentEnrichmentSummary()
    session_recoveries = 0
    max_session_recoveries = max(1, int(os.getenv("TIKTOK_MAX_SESSION_RECOVERIES", "3")))
    delay_sec = max(0.0, float(args.delay))

    async with TikTokApi() as api:
        await _create_api_session(
            api,
            ms_token=ms_token,
            max_attempts=5,
        )

        conn = connect(db_url)
        try:
            summary.seeded_jobs = _seed_jobs_from_videos(
                db_url,
                limit=max(args.limit * 3, args.limit),
                include_existing=args.include_existing,
                max_existing_comments=max(0, int(args.max_existing_comments)),
            )
            candidate_video_ids = _select_candidates_from_jobs(
                conn,
                limit=args.limit,
                max_attempts_per_video=args.max_attempts_per_video,
            )
            summary.candidates_selected = len(candidate_video_ids)
            if not candidate_video_ids:
                print("No candidate videos found for comment enrichment.")
            else:
                for idx, video_id in enumerate(candidate_video_ids, start=1):
                    summary.processed += 1
                    attempt_count = 1
                    try:
                        attempt_count = _mark_running(conn, video_id)
                        comments, video_errs, reply_errs, session_invalid = await fetch_comments_for_video(
                            api,
                            video_id,
                            args.comments,
                            replies_per_comment=args.replies,
                            min_likes_for_replies=args.min_likes_for_replies,
                        )
                        if session_invalid and session_recoveries < max_session_recoveries:
                            session_recoveries += 1
                            await _create_api_session(
                                api,
                                ms_token=ms_token,
                                max_attempts=3,
                            )
                            comments, retry_video_errs, retry_reply_errs, _ = await fetch_comments_for_video(
                                api,
                                video_id,
                                args.comments,
                                replies_per_comment=args.replies,
                                min_likes_for_replies=args.min_likes_for_replies,
                            )
                            video_errs += retry_video_errs
                            reply_errs += retry_reply_errs

                        summary.video_fetch_errors += video_errs
                        summary.reply_fetch_errors += reply_errs

                        if comments:
                            try:
                                written = write_comments_for_existing_video(
                                    video_id,
                                    comments,
                                    conn=conn,
                                )
                                if written > 0:
                                    _mark_done(conn, video_id)
                                    summary.videos_with_comments += 1
                                    summary.comments_written += written
                                else:
                                    final = _mark_retry_or_exhausted(
                                        conn,
                                        video_id=video_id,
                                        attempt_count=attempt_count,
                                        max_attempts_per_video=args.max_attempts_per_video,
                                        last_error="no_comment_rows_written",
                                        retry_backoff_base_sec=args.retry_backoff_base_sec,
                                    )
                                    if final == "exhausted":
                                        summary.exhausted += 1
                            except Exception as exc:  # noqa: BLE001
                                summary.persist_failed += 1
                                final = _mark_retry_or_exhausted(
                                    conn,
                                    video_id=video_id,
                                    attempt_count=attempt_count,
                                    max_attempts_per_video=args.max_attempts_per_video,
                                    last_error=f"persist_failed:{exc}",
                                    retry_backoff_base_sec=args.retry_backoff_base_sec,
                                )
                                if final == "exhausted":
                                    summary.exhausted += 1
                        else:
                            final = _mark_retry_or_exhausted(
                                conn,
                                video_id=video_id,
                                attempt_count=attempt_count,
                                max_attempts_per_video=args.max_attempts_per_video,
                                last_error="empty_or_blocked",
                                retry_backoff_base_sec=args.retry_backoff_base_sec,
                            )
                            if final == "exhausted":
                                summary.exhausted += 1
                    except Exception as exc:  # noqa: BLE001
                        summary.persist_failed += 1
                        try:
                            final = _mark_retry_or_exhausted(
                                conn,
                                video_id=video_id,
                                attempt_count=attempt_count,
                                max_attempts_per_video=args.max_attempts_per_video,
                                last_error=f"unexpected_error:{exc}",
                                retry_backoff_base_sec=args.retry_backoff_base_sec,
                            )
                            if final == "exhausted":
                                summary.exhausted += 1
                        except Exception:  # noqa: BLE001
                            pass
                    finally:
                        try:
                            conn.commit()
                        except Exception:  # noqa: BLE001
                            try:
                                conn.rollback()
                            except Exception:  # noqa: BLE001
                                pass
                            try:
                                conn.close()
                            except Exception:  # noqa: BLE001
                                pass
                            conn = connect(db_url)

                    if delay_sec > 0 and idx < len(candidate_video_ids):
                        await asyncio.sleep(delay_sec)
            summary.job_status_counts = _get_job_status_counts(conn)
        finally:
            conn.close()

    print(
        "Comment enrichment summary: "
        f"candidates={summary.candidates_selected} processed={summary.processed} "
        f"videos_with_comments={summary.videos_with_comments} comments_written={summary.comments_written} "
        f"video_fetch_errors={summary.video_fetch_errors} reply_fetch_errors={summary.reply_fetch_errors} "
        f"persist_failed={summary.persist_failed} exhausted={summary.exhausted} "
        f"seeded_jobs={summary.seeded_jobs} "
        f"job_status_counts={summary.job_status_counts or {}}"
    )
    if args.summary_path:
        path = Path(args.summary_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "candidates_selected": summary.candidates_selected,
            "processed": summary.processed,
            "videos_with_comments": summary.videos_with_comments,
            "comments_written": summary.comments_written,
            "video_fetch_errors": summary.video_fetch_errors,
            "reply_fetch_errors": summary.reply_fetch_errors,
            "persist_failed": summary.persist_failed,
            "exhausted": summary.exhausted,
            "seeded_jobs": summary.seeded_jobs,
            "job_status_counts": summary.job_status_counts or {},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Summary written to {path}")
    return 1 if summary.persist_failed > 0 else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enrich comments for already-scraped videos stored in Postgres/Supabase."
    )
    parser.add_argument("--db-url", default=None, help="Optional DB URL override.")
    parser.add_argument("--ms-token", dest="ms_token", help="MS_TOKEN override.")
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Number of candidate videos to process in this run (default: 300).",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Seed jobs for videos with existing comments (default: seed only missing comments).",
    )
    parser.add_argument(
        "--max-existing-comments",
        type=int,
        default=0,
        help="When include-existing is false, also seed videos with <= N existing comments (default: 0).",
    )
    parser.add_argument(
        "--comments",
        type=int,
        default=5,
        help="Max comments per video to fetch (default: 5).",
    )
    parser.add_argument(
        "--replies",
        type=int,
        default=2,
        help="Max replies per comment to fetch (default: 2).",
    )
    parser.add_argument(
        "--min-likes-for-replies",
        type=int,
        default=10,
        help="Only fetch replies for comments with at least this many likes (default: 10).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between videos (default: 0.5).",
    )
    parser.add_argument(
        "--max-attempts-per-video",
        type=int,
        default=5,
        help="Maximum enrichment attempts per video before marking exhausted (default: 5).",
    )
    parser.add_argument(
        "--retry-backoff-base-sec",
        type=float,
        default=900.0,
        help="Base backoff in seconds for failed videos (exponential, default: 900).",
    )
    parser.add_argument(
        "--stale-running-minutes",
        type=int,
        default=60,
        help="Recover jobs stuck in running state older than this many minutes (default: 60).",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional JSON path to write run summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run_enrichment(args))
