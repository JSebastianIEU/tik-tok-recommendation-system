#!/usr/bin/env python3
"""Export Supabase data to JSONL format compatible with the training pipeline."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import psycopg
except ImportError:
    print("psycopg not installed. Run: pip install psycopg[binary]")
    sys.exit(1)


def main() -> int:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set.")
        return 1

    output_path = Path("data/real/tiktok_posts_real.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to database...")
    conn = psycopg.connect(db_url)
    cur = conn.cursor()

    # Fetch videos with their latest snapshot metrics
    print("Fetching videos with engagement metrics...")
    cur.execute("""
        SELECT
            v.video_id,
            v.url,
            v.caption,
            v.duration_sec,
            v.created_at,
            v.author_id,
            v.audio_id,
            a.username,
            a.display_name,
            a.verified,
            au.audio_name as audio_title,
            vs.likes,
            vs.comments_count,
            vs.shares,
            vs.plays,
            vs.scraped_at
        FROM videos v
        LEFT JOIN authors a ON v.author_id = a.author_id
        LEFT JOIN audios au ON v.audio_id = au.audio_id
        LEFT JOIN LATERAL (
            SELECT likes, comments_count, shares, plays, scraped_at
            FROM video_snapshots
            WHERE video_id = v.video_id
            ORDER BY scraped_at DESC
            LIMIT 1
        ) vs ON true
        WHERE v.caption IS NOT NULL
          AND v.caption != ''
          AND vs.plays IS NOT NULL
          AND vs.plays > 0
        ORDER BY v.created_at DESC
    """)

    video_rows = cur.fetchall()
    video_cols = [desc[0] for desc in cur.description]
    print(f"  Found {len(video_rows)} videos with metrics")

    # Fetch hashtags per video
    print("Fetching hashtags...")
    cur.execute("""
        SELECT vh.video_id, h.tag
        FROM video_hashtags vh
        JOIN hashtags h ON vh.hashtag_id = h.hashtag_id
    """)
    hashtag_map: dict[str, list[str]] = {}
    for vid, tag in cur.fetchall():
        hashtag_map.setdefault(vid, []).append(f"#{tag}")

    # Fetch comments per video (top 5 by length)
    print("Fetching comments...")
    cur.execute("""
        SELECT c.video_id, c.comment_id, c.text, c.username
        FROM comments c
        WHERE c.text IS NOT NULL AND c.text != ''
        ORDER BY c.video_id, length(c.text) DESC
    """)
    comment_map: dict[str, list[dict]] = {}
    for vid, cid, text, username in cur.fetchall():
        if vid not in comment_map:
            comment_map[vid] = []
        if len(comment_map[vid]) < 5:
            comment_map[vid].append({
                "comment_id": cid,
                "text": text.replace("\n", " ").replace("\r", " "),
                "likes": 0,
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            })

    # Fetch author follower counts from metric snapshots
    print("Fetching author metrics...")
    cur.execute("""
        SELECT DISTINCT ON (author_id) author_id, follower_count
        FROM author_metric_snapshots
        ORDER BY author_id, scraped_at DESC
    """)
    author_followers: dict[str, int] = {}
    for aid, follower_count in cur.fetchall():
        if follower_count is not None:
            author_followers[aid] = follower_count

    # Build JSONL records
    print("Building JSONL records...")
    records = []
    for row in video_rows:
        d = dict(zip(video_cols, row))
        video_id = d["video_id"]
        caption = (d["caption"] or "").replace("\n", " ").replace("\r", " ").replace("\u2028", " ").replace("\u2029", " ").replace("\x0b", " ").replace("\x0c", " ").replace("\x1c", " ").replace("\x1d", " ").replace("\x1e", " ").replace("\x85", " ")

        # Extract keywords from caption (simple word extraction)
        words = caption.replace("#", "").split()
        keywords = [w for w in words if len(w) > 3][:5]

        # Build search query from caption (first few meaningful words)
        search_words = [w for w in words if not w.startswith("#") and len(w) > 2][:4]
        search_query = " ".join(search_words) if search_words else caption[:50]

        posted_at = d["created_at"]
        if posted_at:
            posted_at_str = posted_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        else:
            continue

        record = {
            "video_id": video_id,
            "video_url": d["url"] or f"https://www.tiktok.com/video/{video_id}",
            "caption": caption,
            "hashtags": hashtag_map.get(video_id, []),
            "keywords": keywords,
            "search_query": search_query,
            "posted_at": posted_at_str,
            "likes": d["likes"] or 0,
            "comments_count": d["comments_count"] or 0,
            "shares": d["shares"] or 0,
            "views": d["plays"] or 0,
            "author": {
                "author_id": d["author_id"] or "unknown",
                "username": d["username"] or "unknown",
                "followers": author_followers.get(d["author_id"], 0),
            },
            "audio": {
                "audio_id": d["audio_id"] or "unknown",
                "audio_title": d["audio_title"] or "Unknown Audio",
            },
            "video_meta": {
                "duration_seconds": d["duration_sec"] or 0,
                "language": "en",
            },
            "comments": comment_map.get(video_id, []),
        }
        records.append(record)

    conn.close()

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            line = json.dumps(record, ensure_ascii=False)
            # Remove Unicode line separators that break splitlines()-based parsers
            line = line.replace("\u2028", " ").replace("\u2029", " ")
            f.write(line + "\n")

    print(f"\nExported {len(records)} records to {output_path}")
    print(f"  Videos with hashtags: {sum(1 for r in records if r['hashtags'])}")
    print(f"  Videos with comments: {sum(1 for r in records if r['comments'])}")
    print(f"  Avg views: {sum(r['views'] for r in records) / max(len(records), 1):.0f}")
    print(f"  Avg likes: {sum(r['likes'] for r in records) / max(len(records), 1):.0f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
