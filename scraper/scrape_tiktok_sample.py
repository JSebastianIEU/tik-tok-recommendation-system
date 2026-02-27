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
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable
from urllib.parse import urlparse

from TikTokApi import TikTokApi

from scraper.db.writer import ScrapeContext, create_scrape_run, dry_run_print_sql, write_normalized_record


def get_ms_token(explicit: str | None) -> str:
    token = explicit or os.environ.get("MS_TOKEN")
    if not token:
        raise SystemExit(
            "No ms_token provided. Set MS_TOKEN in your env or pass --ms-token on the command line."
        )
    return token


def _parse_proxy_line(line: str) -> Dict[str, Any] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    scheme = "http"
    host_port = line
    if "://" in line:
        parsed = urlparse(line)
        if not parsed.netloc:
            return None
        scheme = parsed.scheme or "http"
        host_port = parsed.netloc

    if ":" not in host_port:
        return None
    host, port_str = host_port.split(":", 1)
    if not host or not port_str:
        return None
    try:
        int(port_str)
    except ValueError:
        return None

    server = f"{scheme}://{host}:{port_str}"
    return {"server": server}


def load_random_proxy(path: str) -> Dict[str, Any] | None:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None

    proxies: list[Dict[str, Any]] = []
    for line in text.splitlines():
        cfg = _parse_proxy_line(line)
        if cfg:
            proxies.append(cfg)

    if not proxies:
        return None
    return random.choice(proxies)


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


def to_normalized(flat: Dict[str, Any], *, source: str, position: int | None) -> Dict[str, Any]:
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
    scraped_at = datetime.now(timezone.utc).isoformat()

    normalized_author = {
        "author_id": author_id,
        "username": author.get("uniqueId"),
        "display_name": author.get("nickname"),
        "bio": raw.get("author", {}).get("signature"),
        "avatar_url": None,
        "verified": bool(author.get("verified")),
    }

    audio_name = music.get("title") or ""
    normalized_video = {
        "video_id": video_id,
        "author_id": author_id,
        "scraped_at": scraped_at,
        "url": raw.get("video", {}).get("playAddr"),
        "caption": flat.get("desc"),
        "hashtags": [h.get("name") for h in raw.get("challenges", []) if isinstance(h, dict) and h.get("name")],
        "audio_name": audio_name,
        "audio_id": music.get("id"),
        "duration_sec": raw.get("video", {}).get("duration"),
        "thumbnail_url": raw.get("video", {}).get("originCover"),
        "created_at": str(flat.get("createTime")) if flat.get("createTime") is not None else None,
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

    return {
        "author": normalized_author,
        "video": normalized_video,
        "authorMetricSnapshot": normalized_author_metrics,
        "comments": [],
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
        "--proxies-file",
        help="Optional path to a file containing proxies (one per line, e.g. http://host:port or socks4://host:port).",
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

    args = parser.parse_args()
    ms_token = get_ms_token(args.ms_token)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async with TikTokApi() as api:
        max_attempts = 5

        for attempt in range(1, max_attempts + 1):
            proxy_cfg = None
            if args.proxies_file:
                proxy_cfg = load_random_proxy(args.proxies_file)
                if proxy_cfg:
                    print(f"[attempt {attempt}/{max_attempts}] Using proxy {proxy_cfg['server']}")
                else:
                    print(
                        f"[attempt {attempt}/{max_attempts}] No valid proxies found in {args.proxies_file}, trying without proxy"
                    )

            try:
                await api.create_sessions(
                    ms_tokens=[ms_token],
                    num_sessions=1,
                    sleep_after=3,
                    browser=os.getenv("TIKTOK_BROWSER", "chromium"),
                    proxies=[proxy_cfg] if proxy_cfg else None,
                    timeout=60000,
                )
                break
            except Exception as e:  # noqa: BLE001
                print(f"[attempt {attempt}/{max_attempts}] Failed to create session: {e}")
                if attempt == max_attempts:
                    raise

        if args.mode == "trending":
            source_label = args.source or "trending"
            source_iter = iter_videos_trending(api, args.count)
        elif args.mode == "hashtag":
            source_label = args.source or f"hashtag:{args.name}"
            source_iter = iter_videos_hashtag(api, args.name, args.count)
        elif args.mode == "user":
            source_label = args.source or f"user:{args.name}"
            source_iter = iter_videos_user(api, args.name, args.count)
        else:
            raise SystemExit(f"Unknown mode: {args.mode}")

        scrape_ctx: ScrapeContext | None = None
        if not args.dry_run:
            scrape_ctx = create_scrape_run(source_label, db_url=args.db_url)

        count = 0
        with out_path.open("w", encoding="utf-8") as f:
            async for raw in source_iter:
                flat = flatten_video(raw)
                f.write(json.dumps(flat, ensure_ascii=False) + "\n")
                count += 1

                normalized = to_normalized(flat, source=source_label, position=count)
                if args.dry_run:
                    dry_run_print_sql(normalized)
                else:
                    write_normalized_record(
                        normalized,
                        db_url=args.db_url,
                        scrape_ctx=scrape_ctx,
                        position=count,
                    )

        print(f"Wrote {count} records to {out_path}")
        return 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()

