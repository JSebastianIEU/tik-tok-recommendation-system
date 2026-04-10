#!/usr/bin/env python3
"""Backfill video_hashtags bridge table by parsing hashtags from video captions.

Problem: Only ~2.9% of videos have bridge entries in video_hashtags, despite 90%+
having hashtags embedded in their captions. The scraper's writer code correctly
handles hashtag linking, but the upstream scraper wasn't consistently extracting
hashtags from captions into the normalized record's `hashtags` field.

Solution: For every video whose caption contains `#tag` patterns but has zero
rows in video_hashtags, parse the hashtags out, upsert into the hashtags table,
and create the bridge links. This is idempotent — safe to re-run.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scraper.db.client import get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backfill_video_hashtags")

# Matches #hashtag patterns (Latin, digits, underscores, plus common Unicode letters)
HASHTAG_RE = re.compile(
    r"#([A-Za-z0-9\u00C0-\u024F\u1E00-\u1EFF_]{1,100})",
    re.UNICODE,
)


def extract_hashtags_from_caption(caption: str) -> List[str]:
    """Extract unique, lowercased hashtag tags from a caption string."""
    if not caption:
        return []
    raw = HASHTAG_RE.findall(caption)
    seen = set()
    result = []
    for tag in raw:
        normalized = tag.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def fetch_orphaned_videos(conn, batch_size: int = 5000) -> List[Dict[str, Any]]:
    """Fetch videos that have # in their caption but zero video_hashtags links."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT v.video_id, v.caption
            FROM videos v
            WHERE v.caption LIKE '%%#%%'
              AND NOT EXISTS (
                  SELECT 1 FROM video_hashtags vh WHERE vh.video_id = v.video_id
              )
            ORDER BY v.video_id
            LIMIT %(limit)s
            """,
            {"limit": batch_size},
        )
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def ensure_hashtags_batch(conn, tags: List[str]) -> Dict[str, int]:
    """Upsert a batch of hashtag tags and return {tag: hashtag_id} mapping."""
    if not tags:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO hashtags (tag)
            SELECT unnest(%(tags)s::text[])
            ON CONFLICT (tag) DO NOTHING
            """,
            {"tags": tags},
        )
        cur.execute(
            """
            SELECT hashtag_id, tag FROM hashtags
            WHERE tag = ANY(%(tags)s::text[])
            """,
            {"tags": tags},
        )
        return {row[1]: row[0] for row in cur.fetchall()}


def link_video_hashtags_batch(
    conn,
    links: List[Tuple[str, int]],
) -> int:
    """Bulk insert video_hashtags links. Returns count of rows inserted."""
    if not links:
        return 0
    video_ids = [l[0] for l in links]
    hashtag_ids = [l[1] for l in links]
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO video_hashtags (video_id, hashtag_id)
            SELECT * FROM unnest(%(video_ids)s::text[], %(hashtag_ids)s::int[])
            ON CONFLICT (video_id, hashtag_id) DO NOTHING
            """,
            {"video_ids": video_ids, "hashtag_ids": hashtag_ids},
        )
        return cur.rowcount


def backfill(db_url: str, batch_size: int = 5000, dry_run: bool = False) -> Dict[str, int]:
    """Run the full backfill process in batches until no orphans remain."""
    stats = {
        "videos_processed": 0,
        "videos_with_hashtags": 0,
        "hashtags_created": 0,
        "links_created": 0,
        "batches": 0,
    }

    while True:
        with get_connection(db_url) as conn:
            orphans = fetch_orphaned_videos(conn, batch_size=batch_size)
            if not orphans:
                break

            stats["batches"] += 1
            all_tags = set()
            video_tags_map: Dict[str, List[str]] = {}

            for row in orphans:
                tags = extract_hashtags_from_caption(row["caption"])
                if tags:
                    video_tags_map[str(row["video_id"])] = tags
                    all_tags.update(tags)

            stats["videos_processed"] += len(orphans)
            stats["videos_with_hashtags"] += len(video_tags_map)

            if not video_tags_map:
                # All orphans had # in caption but no valid hashtags parsed
                # This shouldn't loop forever because we process in order
                break

            if dry_run:
                logger.info("[DRY RUN] Batch %d: %d orphans, %d with hashtags, %d unique tags",
                            stats['batches'], len(orphans), len(video_tags_map), len(all_tags))
                # In dry-run we can't continue because we didn't actually link anything,
                # so the same orphans would be fetched again
                break

            # Upsert all tags
            tag_id_map = ensure_hashtags_batch(conn, sorted(all_tags))
            new_tags = len(all_tags) - len([t for t in all_tags if t in tag_id_map])
            stats["hashtags_created"] += max(0, new_tags)

            # Build link tuples
            links = []
            for video_id, tags in video_tags_map.items():
                for tag in tags:
                    hid = tag_id_map.get(tag)
                    if hid is not None:
                        links.append((video_id, hid))

            inserted = link_video_hashtags_batch(conn, links)
            stats["links_created"] += inserted

            logger.info("Batch %d: %d orphans -> %d with hashtags -> %d links created (%d unique tags)",
                        stats['batches'], len(orphans), len(video_tags_map), inserted, len(all_tags))

    return stats


def print_bridge_stats(db_url: str) -> None:
    """Print current state of the video_hashtags bridge table."""
    with get_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM videos")
            total_videos = cur.fetchone()[0]

            cur.execute("SELECT COUNT(DISTINCT video_id) FROM video_hashtags")
            linked_videos = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM video_hashtags")
            total_links = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM hashtags")
            total_hashtags = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM videos WHERE caption LIKE '%%#%%'"
            )
            videos_with_hash = cur.fetchone()[0]

    pct = (linked_videos / total_videos * 100) if total_videos else 0
    logger.info("Video-Hashtag Bridge Stats: %d videos, %d with # in caption, "
                "%d with bridge links (%.1f%%), %d total links, %d unique hashtags",
                total_videos, videos_with_hash, linked_videos, pct, total_links, total_hashtags)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-url",
        default=os.getenv("DATABASE_URL", ""),
        help="Postgres connection string. Defaults to DATABASE_URL.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of orphaned videos to process per batch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and count but don't write to DB.",
    )
    args = parser.parse_args()

    db_url = str(args.db_url).strip()
    if not db_url:
        raise SystemExit("DATABASE_URL or --db-url is required.")

    logger.info("Before backfill:")
    print_bridge_stats(db_url)

    if args.dry_run:
        logger.info("[DRY RUN MODE - no changes will be written]")

    logger.info("Running backfill...")
    stats = backfill(db_url, batch_size=args.batch_size, dry_run=args.dry_run)

    logger.info("Backfill complete: %d batches, %s videos processed, %s with hashtags, %s links created",
                stats['batches'], f"{stats['videos_processed']:,}",
                f"{stats['videos_with_hashtags']:,}", f"{stats['links_created']:,}")

    if not args.dry_run:
        logger.info("After backfill:")
        print_bridge_stats(db_url)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
