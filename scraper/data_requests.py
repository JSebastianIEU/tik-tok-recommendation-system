from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from psycopg.rows import dict_row

from scraper.db.client import connect, get_database_url

DatasetName = Literal["full", "videos", "comments", "authors"]
DEFAULT_LIMIT = 1000
MAX_LIMIT = 10000


@dataclass
class RetrievalPage:
    rows: list[dict[str, Any]]
    count: int
    next_cursor: dict[str, str] | None


def _parse_since(since: str | None) -> datetime | None:
    if since is None:
        return None
    value = since.strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:  # noqa: BLE001
        raise ValueError(f"Invalid --since value '{since}'. Use ISO-8601, e.g. 2026-03-01T00:00:00Z") from exc


def _validate_limit(limit: int) -> int:
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if limit > MAX_LIMIT:
        raise ValueError(f"limit must be <= {MAX_LIMIT}")
    return limit


def _validate_offset(offset: int) -> int:
    if offset < 0:
        raise ValueError("offset must be >= 0")
    return offset


def _normalize_cursor(
    cursor_created_at: str | None,
    cursor_id: str | None,
) -> tuple[datetime, str] | None:
    if not cursor_created_at and not cursor_id:
        return None
    if not cursor_created_at or not cursor_id:
        raise ValueError("cursor requires both cursor_created_at and cursor_id")
    return _parse_since(cursor_created_at), cursor_id


def _build_pagination_clause(
    *,
    timestamp_expr: str,
    id_expr: str,
    cursor: tuple[datetime, str] | None,
    params: list[Any],
) -> str:
    if cursor is None:
        return ""
    cursor_ts, cursor_id = cursor
    params.extend([cursor_ts, cursor_ts, cursor_id])
    return (
        " AND ("
        f"{timestamp_expr} < %s OR "
        f"({timestamp_expr} = %s AND {id_expr} < %s)"
        ")"
    )


def _to_cursor(timestamp: datetime | None, row_id: str | None) -> dict[str, str] | None:
    if timestamp is None or row_id is None:
        return None
    return {
        "cursor_created_at": timestamp.isoformat(),
        "cursor_id": str(row_id),
    }


def _fetch_rows(
    sql: str,
    params: Sequence[Any],
    *,
    db_url: str | None,
) -> list[dict[str, Any]]:
    with connect(db_url) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]


def get_videos_data(
    *,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    since: str | None = None,
    cursor_created_at: str | None = None,
    cursor_id: str | None = None,
    db_url: str | None = None,
) -> RetrievalPage:
    limit = _validate_limit(limit)
    offset = _validate_offset(offset)
    since_dt = _parse_since(since)
    cursor = _normalize_cursor(cursor_created_at, cursor_id)

    params: list[Any] = []
    where = "WHERE TRUE"
    if since_dt is not None:
        where += " AND v.created_at >= %s"
        params.append(since_dt)
    where += _build_pagination_clause(
        timestamp_expr="COALESCE(v.created_at, TIMESTAMPTZ 'epoch')",
        id_expr="v.video_id",
        cursor=cursor,
        params=params,
    )

    sql = f"""
        WITH latest_video_snapshot AS (
            SELECT DISTINCT ON (video_id)
                video_id,
                scraped_at,
                likes,
                comments_count,
                shares,
                plays
            FROM video_snapshots
            ORDER BY video_id, scraped_at DESC, video_snapshot_id DESC
        )
        SELECT
            v.video_id,
            v.url,
            v.caption,
            v.duration_sec,
            v.thumbnail_url,
            v.created_at,
            v.audio_id,
            a.author_id AS video_author_id,
            a.username AS video_author_username,
            a.display_name AS video_author_display_name,
            a.verified AS video_author_verified,
            lvs.scraped_at AS snapshot_scraped_at,
            lvs.likes,
            lvs.comments_count,
            lvs.shares,
            lvs.plays,
            COALESCE(ht.tags, '{{}}'::TEXT[]) AS hashtags
        FROM videos v
        JOIN authors a ON a.author_id = v.author_id
        LEFT JOIN latest_video_snapshot lvs ON lvs.video_id = v.video_id
        LEFT JOIN LATERAL (
            SELECT ARRAY_AGG(h.tag ORDER BY h.tag) AS tags
            FROM video_hashtags vh
            JOIN hashtags h ON h.hashtag_id = vh.hashtag_id
            WHERE vh.video_id = v.video_id
        ) ht ON TRUE
        {where}
        ORDER BY COALESCE(v.created_at, TIMESTAMPTZ 'epoch') DESC, v.video_id DESC
        LIMIT %s OFFSET %s
    """
    params.extend([limit, offset])
    rows = _fetch_rows(sql, params, db_url=db_url)

    next_cursor = None
    if rows:
        last = rows[-1]
        next_cursor = _to_cursor(last.get("created_at"), last.get("video_id"))

    return RetrievalPage(rows=rows, count=len(rows), next_cursor=next_cursor)


def get_comments_data(
    *,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    since: str | None = None,
    cursor_created_at: str | None = None,
    cursor_id: str | None = None,
    db_url: str | None = None,
) -> RetrievalPage:
    limit = _validate_limit(limit)
    offset = _validate_offset(offset)
    since_dt = _parse_since(since)
    cursor = _normalize_cursor(cursor_created_at, cursor_id)

    activity_expr = "COALESCE(lcs.scraped_at, v.created_at, TIMESTAMPTZ 'epoch')"
    params: list[Any] = []
    where = "WHERE TRUE"
    if since_dt is not None:
        where += f" AND {activity_expr} >= %s"
        params.append(since_dt)
    where += _build_pagination_clause(
        timestamp_expr=activity_expr,
        id_expr="c.comment_id",
        cursor=cursor,
        params=params,
    )

    sql = f"""
        WITH latest_comment_snapshot AS (
            SELECT DISTINCT ON (comment_id)
                comment_id,
                scraped_at,
                likes,
                reply_count
            FROM comment_snapshots
            ORDER BY comment_id, scraped_at DESC, comment_snapshot_id DESC
        )
        SELECT
            c.comment_id,
            c.video_id,
            c.parent_comment_id,
            c.root_comment_id,
            c.comment_level,
            c.text,
            c.username AS comment_username,
            ca.author_id AS comment_author_id,
            ca.username AS comment_author_username,
            ca.display_name AS comment_author_display_name,
            ca.verified AS comment_author_verified,
            v.url AS video_url,
            v.caption AS video_caption,
            v.created_at AS video_created_at,
            lcs.scraped_at AS comment_scraped_at,
            lcs.likes AS comment_likes,
            lcs.reply_count,
            {activity_expr} AS row_created_at
        FROM comments c
        JOIN videos v ON v.video_id = c.video_id
        LEFT JOIN authors ca ON ca.author_id = c.author_id
        LEFT JOIN latest_comment_snapshot lcs ON lcs.comment_id = c.comment_id
        {where}
        ORDER BY {activity_expr} DESC, c.comment_id DESC
        LIMIT %s OFFSET %s
    """
    params.extend([limit, offset])
    rows = _fetch_rows(sql, params, db_url=db_url)

    next_cursor = None
    if rows:
        last = rows[-1]
        next_cursor = _to_cursor(last.get("row_created_at"), last.get("comment_id"))

    return RetrievalPage(rows=rows, count=len(rows), next_cursor=next_cursor)


def get_full_data(
    *,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    since: str | None = None,
    cursor_created_at: str | None = None,
    cursor_id: str | None = None,
    db_url: str | None = None,
) -> RetrievalPage:
    limit = _validate_limit(limit)
    offset = _validate_offset(offset)
    since_dt = _parse_since(since)
    cursor = _normalize_cursor(cursor_created_at, cursor_id)

    params: list[Any] = []
    where = "WHERE TRUE"
    if since_dt is not None:
        where += " AND v.created_at >= %s"
        params.append(since_dt)
    where += _build_pagination_clause(
        timestamp_expr="COALESCE(v.created_at, TIMESTAMPTZ 'epoch')",
        id_expr="v.video_id",
        cursor=cursor,
        params=params,
    )

    sql = f"""
        WITH latest_video_snapshot AS (
            SELECT DISTINCT ON (video_id)
                video_id,
                scraped_at,
                likes,
                comments_count,
                shares,
                plays
            FROM video_snapshots
            ORDER BY video_id, scraped_at DESC, video_snapshot_id DESC
        ),
        latest_comment_snapshot AS (
            SELECT DISTINCT ON (comment_id)
                comment_id,
                scraped_at,
                likes,
                reply_count
            FROM comment_snapshots
            ORDER BY comment_id, scraped_at DESC, comment_snapshot_id DESC
        )
        SELECT
            v.video_id,
            v.url,
            v.caption,
            v.duration_sec,
            v.thumbnail_url,
            v.created_at,
            v.audio_id,
            a.author_id AS video_author_id,
            a.username AS video_author_username,
            a.display_name AS video_author_display_name,
            a.verified AS video_author_verified,
            lvs.scraped_at AS snapshot_scraped_at,
            lvs.likes,
            lvs.comments_count,
            lvs.shares,
            lvs.plays,
            COALESCE(ht.tags, '{{}}'::TEXT[]) AS hashtags,
            COALESCE(cm.comments, '[]'::json) AS comments
        FROM videos v
        JOIN authors a ON a.author_id = v.author_id
        LEFT JOIN latest_video_snapshot lvs ON lvs.video_id = v.video_id
        LEFT JOIN LATERAL (
            SELECT ARRAY_AGG(h.tag ORDER BY h.tag) AS tags
            FROM video_hashtags vh
            JOIN hashtags h ON h.hashtag_id = vh.hashtag_id
            WHERE vh.video_id = v.video_id
        ) ht ON TRUE
        LEFT JOIN LATERAL (
            SELECT json_agg(
                json_build_object(
                    'comment_id', c.comment_id,
                    'video_id', c.video_id,
                    'parent_comment_id', c.parent_comment_id,
                    'root_comment_id', c.root_comment_id,
                    'comment_level', c.comment_level,
                    'text', c.text,
                    'username', c.username,
                    'author_id', ca.author_id,
                    'author_username', ca.username,
                    'author_display_name', ca.display_name,
                    'author_verified', ca.verified,
                    'scraped_at', lcs.scraped_at,
                    'likes', lcs.likes,
                    'reply_count', lcs.reply_count
                )
                ORDER BY COALESCE(lcs.scraped_at, TIMESTAMPTZ 'epoch') DESC, c.comment_id DESC
            ) AS comments
            FROM comments c
            LEFT JOIN authors ca ON ca.author_id = c.author_id
            LEFT JOIN latest_comment_snapshot lcs ON lcs.comment_id = c.comment_id
            WHERE c.video_id = v.video_id
        ) cm ON TRUE
        {where}
        ORDER BY COALESCE(v.created_at, TIMESTAMPTZ 'epoch') DESC, v.video_id DESC
        LIMIT %s OFFSET %s
    """
    params.extend([limit, offset])
    rows = _fetch_rows(sql, params, db_url=db_url)

    next_cursor = None
    if rows:
        last = rows[-1]
        next_cursor = _to_cursor(last.get("created_at"), last.get("video_id"))

    return RetrievalPage(rows=rows, count=len(rows), next_cursor=next_cursor)


def get_authors_data(
    *,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    since: str | None = None,
    cursor_created_at: str | None = None,
    cursor_id: str | None = None,
    db_url: str | None = None,
) -> RetrievalPage:
    limit = _validate_limit(limit)
    offset = _validate_offset(offset)
    since_dt = _parse_since(since)
    cursor = _normalize_cursor(cursor_created_at, cursor_id)

    params: list[Any] = []
    activity_expr = "COALESCE(avs.latest_video_created_at, TIMESTAMPTZ 'epoch')"
    where = "WHERE TRUE"
    if since_dt is not None:
        where += f" AND {activity_expr} >= %s"
        params.append(since_dt)
    where += _build_pagination_clause(
        timestamp_expr=activity_expr,
        id_expr="a.author_id",
        cursor=cursor,
        params=params,
    )

    sql = f"""
        WITH latest_video_snapshot AS (
            SELECT DISTINCT ON (video_id)
                video_id,
                plays
            FROM video_snapshots
            ORDER BY video_id, scraped_at DESC, video_snapshot_id DESC
        ),
        author_video_stats AS (
            SELECT
                v.author_id,
                COUNT(*)::BIGINT AS videos_count,
                MAX(v.created_at) AS latest_video_created_at,
                AVG(lvs.plays) AS avg_video_plays
            FROM videos v
            LEFT JOIN latest_video_snapshot lvs ON lvs.video_id = v.video_id
            GROUP BY v.author_id
        ),
        author_comment_stats AS (
            SELECT
                c.author_id,
                COUNT(*)::BIGINT AS comments_count
            FROM comments c
            WHERE c.author_id IS NOT NULL
            GROUP BY c.author_id
        )
        SELECT
            a.author_id,
            a.username,
            a.display_name,
            a.bio,
            a.avatar_url,
            a.verified,
            COALESCE(avs.videos_count, 0) AS videos_count,
            COALESCE(acs.comments_count, 0) AS comments_count,
            avs.avg_video_plays,
            avs.latest_video_created_at,
            {activity_expr} AS row_created_at
        FROM authors a
        LEFT JOIN author_video_stats avs ON avs.author_id = a.author_id
        LEFT JOIN author_comment_stats acs ON acs.author_id = a.author_id
        {where}
        ORDER BY {activity_expr} DESC, a.author_id DESC
        LIMIT %s OFFSET %s
    """
    params.extend([limit, offset])
    rows = _fetch_rows(sql, params, db_url=db_url)

    next_cursor = None
    if rows:
        last = rows[-1]
        next_cursor = _to_cursor(last.get("row_created_at"), last.get("author_id"))

    return RetrievalPage(rows=rows, count=len(rows), next_cursor=next_cursor)


def _serialize_row_for_csv(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (list, dict)):
            out[key] = json.dumps(value, ensure_ascii=False)
        else:
            out[key] = value
    return out


def export_rows(rows: Sequence[Mapping[str, Any]], *, output_path: Path, fmt: Literal["jsonl", "csv"]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str))
                f.write("\n")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_serialize_row_for_csv(row))


DATASET_LOADERS: dict[DatasetName, Callable[..., RetrievalPage]] = {
    "full": get_full_data,
    "videos": get_videos_data,
    "comments": get_comments_data,
    "authors": get_authors_data,
}


def _fetch_all_pages(
    *,
    dataset: DatasetName,
    db_url: str | None,
    page_limit: int,
    since: str | None,
) -> list[dict[str, Any]]:
    loader = DATASET_LOADERS[dataset]
    cursor_ts: str | None = None
    cursor_id: str | None = None
    all_rows: list[dict[str, Any]] = []

    while True:
        page = loader(
            db_url=db_url,
            limit=page_limit,
            offset=0,
            since=since,
            cursor_created_at=cursor_ts,
            cursor_id=cursor_id,
        )
        if not page.rows:
            break
        all_rows.extend(page.rows)
        if not page.next_cursor:
            break
        cursor_ts = page.next_cursor["cursor_created_at"]
        cursor_id = page.next_cursor["cursor_id"]

    return all_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve pre-joined datasets from scraper Postgres/Supabase.")
    parser.add_argument("--dataset", choices=["full", "videos", "comments", "authors"], default="full")
    parser.add_argument("--db-url", default=None, help="Optional DB URL override (else DATABASE_URL).")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"Rows per page (default: {DEFAULT_LIMIT}).")
    parser.add_argument("--offset", type=int, default=0, help="Offset for paged retrieval (default: 0).")
    parser.add_argument("--since", default=None, help="ISO-8601 timestamp lower bound.")
    parser.add_argument("--cursor-created-at", default=None, help="Keyset cursor timestamp from prior page.")
    parser.add_argument("--cursor-id", default=None, help="Keyset cursor id from prior page.")
    parser.add_argument("--all", action="store_true", help="Fetch all pages using keyset pagination.")
    parser.add_argument("--out", default=None, help="Optional output path (.jsonl or .csv).")
    parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl", help="Output format if --out is set.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    db_url = get_database_url(args.db_url)
    dataset: DatasetName = args.dataset

    if args.all:
        rows = _fetch_all_pages(
            dataset=dataset,
            db_url=db_url,
            page_limit=_validate_limit(args.limit),
            since=args.since,
        )
        next_cursor = None
    else:
        if args.offset and (args.cursor_created_at or args.cursor_id):
            raise ValueError("Use either offset pagination or cursor pagination, not both.")
        page = DATASET_LOADERS[dataset](
            db_url=db_url,
            limit=args.limit,
            offset=args.offset,
            since=args.since,
            cursor_created_at=args.cursor_created_at,
            cursor_id=args.cursor_id,
        )
        rows = page.rows
        next_cursor = page.next_cursor

    if args.out:
        output_path = Path(args.out).resolve()
        export_rows(rows, output_path=output_path, fmt=args.format)
        print(f"Wrote {len(rows)} rows to {output_path}")
    else:
        print(json.dumps({"dataset": dataset, "count": len(rows), "next_cursor": next_cursor}, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
