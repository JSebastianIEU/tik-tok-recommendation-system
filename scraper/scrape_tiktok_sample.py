"""
Collect sample TikTok data using the unofficial TikTokApi wrapper.

This adapts the provided reference script and optionally persists
flattened records into the Postgres 3NF schema used by the Selenium
post scraper.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict

from TikTokApi import TikTokApi

# Reduce TikTokApi noise (status 10201, etc.)
logging.getLogger("TikTokApi").setLevel(logging.WARNING)


def _is_probable_bot_block_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    markers = (
        "empty response",
        "status_code': 4",
        "status_code\": 4",
        "captcha",
        "rate limit",
        "too many requests",
    )
    return any(m in msg for m in markers)


def get_ms_token(explicit: str | None) -> str:
    token = explicit or os.environ.get("MS_TOKEN")
    if not token:
        raise SystemExit(
            "No ms_token provided. Set MS_TOKEN in your env or pass --ms-token on the command line."
        )
    return token


async def _create_api_session(
    api: TikTokApi,
    *,
    ms_token: str,
    max_attempts: int = 5,
) -> None:
    for attempt in range(1, max_attempts + 1):
        try:
            headless = os.getenv("TIKTOK_HEADLESS", "false").lower() in ("1", "true", "yes")
            browser = os.getenv("TIKTOK_BROWSER", "webkit").strip() or "webkit"
            await api.create_sessions(
                ms_tokens=[ms_token],
                num_sessions=1,
                sleep_after=5,
                headless=headless,
                browser=browser,
                timeout=60000,
            )
            return
        except Exception as e:  # noqa: BLE001
            print(f"[attempt {attempt}/{max_attempts}] Failed to create session: {e}")
            if attempt == max_attempts:
                raise


def _resolve_db_url(explicit: str | None) -> str | None:
    raw = (explicit or os.getenv("DATABASE_URL") or "").strip()
    return raw or None


def _dry_run_print(normalized: Dict[str, Any]) -> None:
    try:
        from scraper.db.writer import dry_run_print_sql
    except ModuleNotFoundError:
        author = normalized.get("author") or {}
        video = normalized.get("video") or {}
        comments = normalized.get("comments") or []
        print(
            f"[DRY RUN] author={author.get('author_id')} "
            f"video={video.get('video_id')} comments={len(comments)}"
        )
        return
    dry_run_print_sql(normalized)


async def iter_videos_trending(api: TikTokApi, count: int) -> AsyncIterator[Dict[str, Any]]:
    async for video in api.trending.videos(count=count):
        yield video.as_dict


async def iter_videos_hashtag(api: TikTokApi, name: str, count: int) -> AsyncIterator[Dict[str, Any]]:
    ht = api.hashtag(name)
    async for video in ht.videos(count=count):
        yield video.as_dict


async def iter_videos_user(api: TikTokApi, name: str, count: int) -> AsyncIterator[Dict[str, Any]]:
    user = api.user(name)
    async for video in user.videos(count=count):
        yield video.as_dict


async def iter_videos_keyword(api: TikTokApi, query: str, count: int) -> AsyncIterator[Dict[str, Any]]:
    """
    Keyword search over TikTokApi. Uses search.search_type(..., "item") when
    search.videos is not available (e.g. TikTokApi 6.x).
    """
    search = getattr(api, "search", None)
    if not search:
        raise RuntimeError("TikTokApi search API is not available in this version.")

    # TikTokApi 6.x: search has search_type(term, "item") for videos, not .videos
    search_type = getattr(search, "search_type", None)
    if callable(search_type):
        found = 0
        async for item in search_type(query, "item", count=count):
            if found >= count:
                return
            as_dict = getattr(item, "as_dict", None)
            if isinstance(as_dict, dict):
                yield as_dict
            elif isinstance(item, dict):
                yield item
            else:
                yield {}
            found += 1
        return

    # Fallback: try search.videos if it exists (older versions)
    videos_method = getattr(search, "videos", None)
    if not callable(videos_method):
        raise RuntimeError(
            "TikTokApi search.videos and search.search_type are not available. "
            "Keyword search may require a different TikTokApi version."
        )

    call_variants = [
        lambda: videos_method(query, count=count),
        lambda: videos_method(query=query, count=count),
        lambda: videos_method(keyword=query, count=count),
        lambda: videos_method(keywords=query, count=count),
    ]
    last_error: Exception | None = None
    for factory in call_variants:
        try:
            stream = factory()
            async for video in stream:
                yield video.as_dict
            return
        except TypeError as exc:
            last_error = exc
            continue
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    raise RuntimeError(f"Keyword search failed for '{query}': {last_error}")


def _comment_to_dict(c: Any, *, parent_comment_id: str | None = None) -> Dict[str, Any]:
    """Convert TikTokApi Comment to normalized dict."""
    c_dict = getattr(c, "as_dict", None) or {}
    user = c_dict.get("user") or {}
    uid = user.get("uid")
    text = getattr(c, "text", None) or c_dict.get("text", "")
    comment_id = getattr(c, "id", None) or c_dict.get("cid")
    return {
        "comment_id": str(comment_id) if comment_id else None,
        "author_id": str(uid) if uid is not None else None,
        "username": user.get("unique_id"),
        "text": text or "",
        "parent_comment_id": parent_comment_id,
        "comment_level": 1 if parent_comment_id else 0,
        "root_comment_id": parent_comment_id or (str(comment_id) if comment_id else None),
    }


async def fetch_comments_for_video(
    api: TikTokApi,
    video_id: str,
    count: int,
    replies_per_comment: int = 0,
    min_likes_for_replies: int = 0,
) -> tuple[list[Dict[str, Any]], int, int, bool]:
    """Fetch top comments (and optionally replies) for a video via TikTokApi.
    When min_likes_for_replies > 0, only fetch replies for comments with at least that many likes.
    """
    if not video_id or count <= 0:
        return [], 0, 0, False
    attempts = max(1, int(os.getenv("TIKTOK_COMMENT_FETCH_ATTEMPTS", "2")))
    base_backoff = max(0.0, float(os.getenv("TIKTOK_COMMENT_BACKOFF_SEC", "1.0")))

    for attempt in range(1, attempts + 1):
        comments: list[Dict[str, Any]] = []
        video_level_errors = 0
        reply_level_errors = 0
        session_invalid = False
        try:
            video_obj = api.video(id=str(video_id))
            async for c in video_obj.comments(count=count):
                parent_id = getattr(c, "id", None) or (getattr(c, "as_dict", None) or {}).get("cid")
                parent_id = str(parent_id) if parent_id else None
                comments.append(_comment_to_dict(c, parent_comment_id=None))

                if replies_per_comment > 0 and parent_id:
                    likes = getattr(c, "likes_count", None) or (getattr(c, "as_dict", None) or {}).get("digg_count", 0)
                    if min_likes_for_replies > 0 and (likes is None or likes < min_likes_for_replies):
                        continue
                    try:
                        async for reply in c.replies(count=replies_per_comment):
                            comments.append(_comment_to_dict(reply, parent_comment_id=parent_id))
                    except Exception:  # noqa: BLE001
                        reply_level_errors += 1
                        logging.warning("reply_fetch_failed video_id=%s parent_comment_id=%s", video_id, parent_id)
            return comments, 0, reply_level_errors, False
        except Exception as exc:  # noqa: BLE001
            video_level_errors = 1
            msg = str(exc).lower()
            session_invalid = (
                "no sessions created" in msg
                or "no valid sessions" in msg
                or ("session" in msg and "closed" in msg)
            )
            should_retry = attempt < attempts and (_is_probable_bot_block_error(exc) or session_invalid)
            if should_retry:
                await asyncio.sleep(base_backoff * attempt)
                continue
            logging.warning("comment_fetch_failed video_id=%s error=%s", video_id, exc)
            return [], video_level_errors, reply_level_errors, session_invalid

    return [], 1, 0, False


def _to_iso_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return datetime.fromtimestamp(int(stripped), tz=timezone.utc).isoformat()
        return stripped or None
    return str(value)


def flatten_video(v: Dict[str, Any]) -> Dict[str, Any]:
    author = v.get("author", {}) or {}
    stats = v.get("stats", {}) or {}
    music = v.get("music", {}) or {}
    video_id = v.get("id")

    return {
        "id": video_id,
        "desc": v.get("desc"),
        "createTime": v.get("createTime"),
        "author": {
            "id": author.get("id"),
            "uniqueId": author.get("uniqueId"),
            "nickname": author.get("nickname"),
            "verified": author.get("verified"),
        },
        "stats": {
            "playCount": stats.get("playCount"),
            "diggCount": stats.get("diggCount"),
            "commentCount": stats.get("commentCount"),
            "shareCount": stats.get("shareCount"),
        },
        "music": {
            "id": music.get("id"),
            "title": music.get("title"),
            "authorName": music.get("authorName"),
        },
        "raw": v,
    }


def _build_tiktok_page_url(raw: Dict[str, Any], video_id: Any) -> str | None:
    """Build TikTok page URL from raw video dict (required for DB videos.url)."""
    share_url = raw.get("shareUrl") or raw.get("share_url") or raw.get("webVideoUrl")
    if isinstance(share_url, str) and "tiktok.com" in share_url and "/video/" in share_url:
        return share_url.split("?")[0].split("#")[0].rstrip("/")
    author = raw.get("author") or {}
    unique_id = author.get("uniqueId") if isinstance(author, dict) else None
    if video_id and unique_id:
        return f"https://www.tiktok.com/@{unique_id}/video/{video_id}"
    return None


def to_normalized(
    flat: Dict[str, Any],
    *,
    source: str,
    position: int | None,
    comments: list[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Map the flattened TikTokApi record into the shared normalized schema
    expected by scraper.db.writer.write_normalized_record.
    """
    raw = flat.get("raw") or {}
    author = flat.get("author") or raw.get("author") or {}
    stats = flat.get("stats") or raw.get("stats") or {}
    music = flat.get("music") or raw.get("music") or {}

    author_id = author.get("id")
    video_id = flat.get("id")
    if author_id is not None:
        author_id = str(author_id)
    if video_id is not None:
        video_id = str(video_id)
    scraped_at = datetime.now(timezone.utc).isoformat()

    normalized_author = {
        "author_id": author_id,
        "username": author.get("uniqueId"),
        "display_name": author.get("nickname"),
        "bio": raw.get("author", {}).get("signature"),
        "avatar_url": None,
        "verified": bool(author.get("verified")),
    }

    page_url = _build_tiktok_page_url(raw, video_id)
    audio_name = music.get("title") or ""
    normalized_video = {
        "video_id": video_id,
        "author_id": author_id,
        "scraped_at": scraped_at,
        "url": page_url,
        "caption": flat.get("desc"),
        "hashtags": [h.get("name") for h in raw.get("challenges", []) if isinstance(h, dict) and h.get("name")],
        "audio_name": audio_name,
        "audio_id": music.get("id"),
        "duration_sec": raw.get("video", {}).get("duration"),
        "thumbnail_url": raw.get("video", {}).get("originCover"),
        "created_at": _to_iso_datetime(flat.get("createTime")),
        "likes": stats.get("diggCount"),
        "comments_count": stats.get("commentCount"),
        "shares": stats.get("shareCount"),
        "plays": stats.get("playCount"),
        "source": source,
        "position": position,
    }

    normalized_author_metrics = {
        "author_id": author_id,
        "video_id": video_id,
        "scraped_at": scraped_at,
        "follower_count": raw.get("authorStats", {}).get("followerCount"),
        "following_count": raw.get("authorStats", {}).get("followingCount"),
        "author_likes_count": raw.get("authorStats", {}).get("heartCount"),
    }

    normalized_comments = []
    if comments:
        for c in comments:
            normalized_comments.append({
                "comment_id": c.get("comment_id") or c.get("id"),
                "author_id": c.get("author_id"),
                "username": c.get("username"),
                "text": c.get("text") or "",
                "parent_comment_id": c.get("parent_comment_id") or None,
                "root_comment_id": c.get("root_comment_id") or None,
                "comment_level": c.get("comment_level"),
            })

    return {
        "author": normalized_author,
        "video": normalized_video,
        "authorMetricSnapshot": normalized_author_metrics,
        "comments": normalized_comments,
    }


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="Collect sample TikTok data using TikTokApi.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_trending = subparsers.add_parser("trending", help="Collect trending videos")
    p_trending.add_argument("--count", type=int, default=200, help="Number of videos to fetch (default: 200)")

    p_hashtag = subparsers.add_parser("hashtag", help="Collect videos for a hashtag")
    p_hashtag.add_argument("--name", required=True, help="Hashtag name (without #)")
    p_hashtag.add_argument("--count", type=int, default=200, help="Number of videos to fetch (default: 200)")

    p_user = subparsers.add_parser("user", help="Collect videos for a user")
    p_user.add_argument("--name", required=True, help="User uniqueId / username")
    p_user.add_argument("--count", type=int, default=200, help="Number of videos to fetch (default: 200)")
    p_keyword = subparsers.add_parser("keyword", help="Collect videos for a search keyword")
    p_keyword.add_argument("--name", required=True, help="Keyword phrase")
    p_keyword.add_argument("--count", type=int, default=200, help="Number of videos to fetch (default: 200)")

    parser.add_argument(
        "--ms-token",
        dest="ms_token",
        help="TikTok ms_token cookie value (or set MS_TOKEN env var).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="tiktok_sample.jsonl",
        help="Path to output JSONL file (default: tiktok_sample.jsonl)",
    )
    parser.add_argument(
        "--db-url",
        help="Optional Postgres URL (falls back to DATABASE_URL env var).",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Logical source label override for this run; defaults based on mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write to Postgres; only emit JSONL and print summaries.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip DB write for videos already in DB (avoids duplicate storage, saves writes).",
    )
    parser.add_argument(
        "--comments",
        type=int,
        default=5,
        help="Max comments per video to fetch (default 5).",
    )
    parser.add_argument(
        "--replies",
        type=int,
        default=5,
        help="Max replies per comment to fetch (default 5).",
    )
    parser.add_argument(
        "--min-likes-for-replies",
        type=int,
        default=10,
        help="Only fetch replies for comments with at least this many likes (default 10).",
    )

    args = parser.parse_args()
    ms_token = get_ms_token(args.ms_token)
    resolved_db_url = _resolve_db_url(args.db_url)
    db_enabled = bool(resolved_db_url) and not args.dry_run

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async with TikTokApi() as api:
        await _create_api_session(
            api,
            ms_token=ms_token,
            max_attempts=5,
        )

        if args.mode == "trending":
            source_label = args.source or "trending"
            source_iter = iter_videos_trending(api, args.count)
        elif args.mode == "hashtag":
            source_label = args.source or f"hashtag:{args.name}"
            source_iter = iter_videos_hashtag(api, args.name, args.count)
        elif args.mode == "user":
            source_label = args.source or f"user:{args.name}"
            source_iter = iter_videos_user(api, args.name, args.count)
        elif args.mode == "keyword":
            source_label = args.source or f"keyword:{args.name}"
            source_iter = iter_videos_keyword(api, args.name, args.count)
        else:
            raise SystemExit(f"Unknown mode: {args.mode}")

        scrape_ctx = None
        db_writer = None
        if db_enabled:
            from scraper.db.writer import BatchedRecordWriter, create_scrape_run

            scrape_ctx = create_scrape_run(source_label, db_url=resolved_db_url)
            commit_every = max(1, int(os.getenv("SCRAPER_DB_COMMIT_EVERY", "50")))
            db_writer = BatchedRecordWriter(db_url=resolved_db_url, commit_every=commit_every)
        elif not args.dry_run:
            print("DB disabled (no --db-url and no DATABASE_URL). Writing JSONL only.")

        count = 0
        persisted_ok = 0
        persisted_skipped = 0
        persist_failed = 0
        comment_fetch_errors = 0
        reply_fetch_errors = 0
        comment_fetch_attempts = 0
        comment_fetch_successes = 0
        comment_circuit_open = False
        session_recoveries = 0
        max_session_recoveries = max(1, int(os.getenv("TIKTOK_MAX_SESSION_RECOVERIES", "3")))
        delay_every_n = int(os.getenv("TIKTOK_DELAY_EVERY_N", "30"))
        delay_sec = float(os.getenv("TIKTOK_DELAY_SEC", "2"))
        comment_circuit_min_attempts = max(1, int(os.getenv("TIKTOK_COMMENT_CIRCUIT_MIN_ATTEMPTS", "40")))
        comment_circuit_fail_rate = min(1.0, max(0.0, float(os.getenv("TIKTOK_COMMENT_CIRCUIT_FAIL_RATE", "0.85"))))

        writer_ctx = db_writer if db_writer is not None else None
        manager = writer_ctx if writer_ctx is not None else nullcontext(None)
        with out_path.open("w", encoding="utf-8") as f, manager as active_writer:
            try:
                async for raw in source_iter:
                    flat = flatten_video(raw)
                    f.write(json.dumps(flat, ensure_ascii=False) + "\n")
                    count += 1

                    comments: list[Dict[str, Any]] = []
                    if args.comments > 0 and not comment_circuit_open:
                        video_id = flat.get("id")
                        if video_id:
                            comment_fetch_attempts += 1
                            comments, video_errs, reply_errs, session_invalid = await fetch_comments_for_video(
                                api,
                                str(video_id),
                                args.comments,
                                replies_per_comment=args.replies,
                                min_likes_for_replies=args.min_likes_for_replies,
                            )
                            if session_invalid and session_recoveries < max_session_recoveries:
                                session_recoveries += 1
                                logging.warning(
                                    "session_recovery_attempt=%s video_id=%s",
                                    session_recoveries,
                                    video_id,
                                )
                                await _create_api_session(
                                    api,
                                    ms_token=ms_token,
                                    max_attempts=3,
                                )
                                comments, retry_video_errs, retry_reply_errs, _ = await fetch_comments_for_video(
                                    api,
                                    str(video_id),
                                    args.comments,
                                    replies_per_comment=args.replies,
                                    min_likes_for_replies=args.min_likes_for_replies,
                                )
                                video_errs += retry_video_errs
                                reply_errs += retry_reply_errs
                            comment_fetch_errors += video_errs
                            reply_fetch_errors += reply_errs
                            if video_errs == 0:
                                comment_fetch_successes += 1
                            if delay_sec > 0 and comments:
                                await asyncio.sleep(delay_sec * 0.5)
                            if comment_fetch_attempts >= comment_circuit_min_attempts:
                                fail_rate = comment_fetch_errors / comment_fetch_attempts
                                if fail_rate >= comment_circuit_fail_rate:
                                    comment_circuit_open = True
                                    logging.warning(
                                        "comment_circuit_open source=%s attempts=%s errors=%s fail_rate=%.2f",
                                        source_label,
                                        comment_fetch_attempts,
                                        comment_fetch_errors,
                                        fail_rate,
                                    )

                    normalized = to_normalized(
                        flat,
                        source=source_label,
                        position=count,
                        comments=comments,
                    )
                    if args.dry_run:
                        _dry_run_print(normalized)
                    elif db_enabled and active_writer is not None:
                        video = normalized.get("video", {})
                        author = normalized.get("author", {})
                        if video.get("url") and author.get("author_id"):
                            try:
                                wrote = active_writer.write(
                                    normalized,
                                    scrape_ctx=scrape_ctx,
                                    position=count,
                                    skip_existing=args.skip_existing,
                                )
                                if wrote:
                                    persisted_ok += 1
                                else:
                                    persisted_skipped += 1
                            except Exception as exc:  # noqa: BLE001
                                persist_failed += 1
                                active_writer.rollback()
                                logging.warning("db_persist_failed video_id=%s error=%s", video.get("video_id"), exc)

                    if delay_every_n > 0 and count % delay_every_n == 0 and delay_sec > 0:
                        await asyncio.sleep(delay_sec)
            except Exception as e:  # noqa: BLE001
                recoverable: tuple[type, ...] = (KeyError,)
                try:
                    from TikTokApi.exceptions import EmptyResponseException, CaptchaException
                    recoverable = (EmptyResponseException, CaptchaException, KeyError)
                except ImportError:
                    pass
                if isinstance(e, recoverable):
                    print(f"Stopped early ({type(e).__name__}): {e}", file=sys.stderr)
                else:
                    raise

        print(f"Wrote {count} records to {out_path}")
        if db_enabled:
            print(
                "DB summary: "
                f"persisted_ok={persisted_ok} persisted_skipped={persisted_skipped} persist_failed={persist_failed}"
            )
        print(
            "Comment summary: "
            f"video_fetch_errors={comment_fetch_errors} "
            f"reply_fetch_errors={reply_fetch_errors} "
            f"fetch_attempts={comment_fetch_attempts} "
            f"fetch_successes={comment_fetch_successes} "
            f"circuit_open={comment_circuit_open}"
        )
        return 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
