#!/usr/bin/env python3
"""Export a canonical contract bundle and manifest from the scraper Postgres DB."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scraper.db.client import get_connection
from src.recommendation import CanonicalDatasetBundle, build_contract_manifest


def _parse_iso_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _parse_iso_datetime(text)


def _to_non_negative_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _fetch_all(cursor, sql: str, params: Any = ()) -> List[Dict[str, Any]]:
    cursor.execute(sql, params)
    columns = [desc.name for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def export_bundle_from_db(
    *,
    db_url: str,
    as_of_time: datetime,
    limit_videos: Optional[int] = None,
    since: Optional[datetime] = None,
) -> CanonicalDatasetBundle:
    with get_connection(db_url) as conn:
        with conn.cursor() as cursor:
            video_filters: List[str] = ["v.created_at <= %(as_of_time)s"]
            params: Dict[str, Any] = {"as_of_time": as_of_time}
            if since is not None:
                video_filters.append("v.created_at >= %(since)s")
                params["since"] = since
            video_where = " AND ".join(video_filters)
            video_limit_sql = ""
            if limit_videos is not None:
                params["limit_videos"] = int(limit_videos)
                video_limit_sql = "LIMIT %(limit_videos)s"

            selected_videos = _fetch_all(
                cursor,
                f"""
WITH latest_author_metrics AS (
    SELECT DISTINCT ON (author_id)
        author_id,
        scraped_at,
        follower_count
    FROM author_metric_snapshots
    WHERE scraped_at <= %(as_of_time)s
    ORDER BY author_id, scraped_at DESC, author_metric_snapshot_id DESC
)
SELECT
    v.video_id,
    v.author_id,
    v.caption,
    v.created_at,
    v.url,
    v.audio_id,
    v.duration_sec,
    v.thumbnail_url,
    a.username,
    a.display_name,
    a.verified,
    lam.scraped_at AS author_scraped_at,
    lam.follower_count AS author_followers_count,
    COALESCE(ht.tags, '{{}}'::text[]) AS hashtags
FROM videos v
JOIN authors a ON a.author_id = v.author_id
LEFT JOIN latest_author_metrics lam ON lam.author_id = a.author_id
LEFT JOIN LATERAL (
    SELECT ARRAY_AGG(h.tag ORDER BY h.tag) AS tags
    FROM video_hashtags vh
    JOIN hashtags h ON h.hashtag_id = vh.hashtag_id
    WHERE vh.video_id = v.video_id
) ht ON TRUE
WHERE {video_where}
ORDER BY v.created_at ASC, v.video_id ASC
{video_limit_sql}
""",
                params,
            )
            if not selected_videos:
                raise RuntimeError("No videos matched the export criteria.")

            video_ids = [str(row["video_id"]) for row in selected_videos]
            author_ids = sorted({str(row["author_id"]) for row in selected_videos})

            authors = _fetch_all(
                cursor,
                """
WITH latest_author_metrics AS (
    SELECT DISTINCT ON (author_id)
        author_id,
        scraped_at,
        follower_count
    FROM author_metric_snapshots
    WHERE scraped_at <= %(as_of_time)s
    ORDER BY author_id, scraped_at DESC, author_metric_snapshot_id DESC
)
SELECT
    a.author_id,
    a.username,
    a.display_name,
    a.verified,
    lam.scraped_at AS author_scraped_at,
    lam.follower_count AS author_followers_count
FROM authors a
LEFT JOIN latest_author_metrics lam ON lam.author_id = a.author_id
WHERE a.author_id = ANY(%(author_ids)s::text[])
ORDER BY a.author_id ASC
""",
                {"as_of_time": as_of_time, "author_ids": author_ids},
            )

            video_snapshots = _fetch_all(
                cursor,
                """
SELECT
    video_snapshot_id,
    video_id,
    scraped_at,
    likes,
    comments_count,
    shares,
    plays,
    position
FROM video_snapshots
WHERE video_id = ANY(%(video_ids)s::text[])
  AND scraped_at <= %(as_of_time)s
ORDER BY scraped_at ASC, video_snapshot_id ASC
""",
                {"video_ids": video_ids, "as_of_time": as_of_time},
            )

            comments = _fetch_all(
                cursor,
                """
WITH comment_created AS (
    SELECT
        c.comment_id,
        MIN(cs.scraped_at) AS first_scraped_at
    FROM comments c
    LEFT JOIN comment_snapshots cs ON cs.comment_id = c.comment_id
    WHERE c.video_id = ANY(%(video_ids)s::text[])
    GROUP BY c.comment_id
)
SELECT
    c.comment_id,
    c.video_id,
    c.author_id,
    c.text,
    c.parent_comment_id,
    c.root_comment_id,
    c.comment_level,
    cc.first_scraped_at,
    v.created_at AS video_created_at
FROM comments c
JOIN videos v ON v.video_id = c.video_id
LEFT JOIN comment_created cc ON cc.comment_id = c.comment_id
WHERE c.video_id = ANY(%(video_ids)s::text[])
ORDER BY c.video_id ASC, c.comment_id ASC
""",
                {"video_ids": video_ids},
            )

            comment_snapshots = _fetch_all(
                cursor,
                """
SELECT
    cs.comment_snapshot_id,
    cs.comment_id,
    c.video_id,
    cs.scraped_at,
    cs.likes,
    cs.reply_count
FROM comment_snapshots cs
JOIN comments c ON c.comment_id = cs.comment_id
WHERE c.video_id = ANY(%(video_ids)s::text[])
  AND cs.scraped_at <= %(as_of_time)s
ORDER BY cs.scraped_at ASC, cs.comment_snapshot_id ASC
""",
                {"video_ids": video_ids, "as_of_time": as_of_time},
            )

    bundle = CanonicalDatasetBundle(
        generated_at=as_of_time,
        authors=[
            {
                "author_id": str(row["author_id"]),
                "username": row.get("username"),
                "display_name": row.get("display_name"),
                "followers_count": _to_non_negative_int(row.get("author_followers_count")),
                "verified": row.get("verified"),
                "scraped_at": _normalize_dt(row.get("author_scraped_at")),
                "source": "supabase_export",
            }
            for row in authors
        ],
        videos=[
            {
                "video_id": str(row["video_id"]),
                "author_id": str(row["author_id"]),
                "caption": str(row.get("caption") or ""),
                "hashtags": list(row.get("hashtags") or []),
                "keywords": [],
                "posted_at": _normalize_dt(row.get("created_at")),
                "video_url": row.get("url"),
                "audio_id": row.get("audio_id"),
                "duration_seconds": _to_non_negative_int(row.get("duration_sec")),
                "language": None,
                "source": "supabase_export",
            }
            for row in selected_videos
        ],
        video_snapshots=[
            {
                "video_snapshot_id": str(row["video_snapshot_id"]),
                "video_id": str(row["video_id"]),
                "scraped_at": _normalize_dt(row.get("scraped_at")),
                "views": _to_non_negative_int(row.get("plays")),
                "likes": _to_non_negative_int(row.get("likes")),
                "comments_count": _to_non_negative_int(row.get("comments_count")),
                "shares": _to_non_negative_int(row.get("shares")),
                "plays": _to_non_negative_int(row.get("plays")),
                "position": row.get("position"),
                "source": "supabase_export",
            }
            for row in video_snapshots
        ],
        comments=[
            {
                "comment_id": str(row["comment_id"]),
                "video_id": str(row["video_id"]),
                "author_id": str(row["author_id"]) if row.get("author_id") is not None else None,
                "text": str(row.get("text") or ""),
                "parent_comment_id": row.get("parent_comment_id"),
                "root_comment_id": row.get("root_comment_id"),
                "comment_level": row.get("comment_level"),
                "created_at": _normalize_dt(row.get("first_scraped_at"))
                or _normalize_dt(row.get("video_created_at"))
                or as_of_time,
                "source": "supabase_export",
            }
            for row in comments
        ],
        comment_snapshots=[
            {
                "comment_snapshot_id": str(row["comment_snapshot_id"]),
                "comment_id": str(row["comment_id"]),
                "video_id": str(row["video_id"]),
                "scraped_at": _normalize_dt(row.get("scraped_at")),
                "likes": _to_non_negative_int(row.get("likes")),
                "reply_count": _to_non_negative_int(row.get("reply_count")),
                "source": "supabase_export",
            }
            for row in comment_snapshots
        ],
    )
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-url",
        default=os.getenv("DATABASE_URL", ""),
        help="Postgres connection string. Defaults to DATABASE_URL.",
    )
    parser.add_argument(
        "--as-of-time",
        type=str,
        default=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        help="As-of timestamp in ISO-8601.",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Optional lower bound for videos.created_at.",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help="Optional cap on exported videos, ordered by videos.created_at ASC.",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("artifacts/contracts"),
        help="Output root for contract manifests.",
    )
    parser.add_argument(
        "--output-bundle-json",
        type=Path,
        default=None,
        help="Optional path to also write the canonical bundle JSON.",
    )
    args = parser.parse_args()

    if not str(args.db_url).strip():
        raise SystemExit("DATABASE_URL or --db-url is required.")

    as_of_time = _parse_iso_datetime(args.as_of_time)
    since = _parse_iso_datetime(args.since) if args.since else None
    bundle = export_bundle_from_db(
        db_url=str(args.db_url).strip(),
        as_of_time=as_of_time,
        limit_videos=args.limit_videos,
        since=since,
    )
    manifest = build_contract_manifest(
        bundle=bundle,
        manifest_root=args.manifest_root,
        source_file_hashes={"source": "supabase_export"},
        as_of_time=as_of_time,
    )
    if args.output_bundle_json is not None:
        args.output_bundle_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_bundle_json.write_text(
            json.dumps(bundle.model_dump(mode="python"), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "manifest_id": manifest["manifest_id"],
                "manifest_dir": manifest["manifest_dir"],
                "bundle_file": manifest["bundle_file"],
                "entity_counts": {
                    "authors": len(bundle.authors),
                    "videos": len(bundle.videos),
                    "video_snapshots": len(bundle.video_snapshots),
                    "comments": len(bundle.comments),
                    "comment_snapshots": len(bundle.comment_snapshots),
                },
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
