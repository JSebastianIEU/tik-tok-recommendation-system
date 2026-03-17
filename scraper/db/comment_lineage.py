from __future__ import annotations

from typing import Optional

from psycopg import Connection

from .client import connect, get_database_url


def _ensure_columns_with_conn(conn: Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            ALTER TABLE comments ADD COLUMN IF NOT EXISTS root_comment_id TEXT;
            ALTER TABLE comments ADD COLUMN IF NOT EXISTS comment_level SMALLINT NOT NULL DEFAULT 0;
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_comments_root_comment_id ON comments(root_comment_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_comments_comment_level ON comments(comment_level)")


def backfill_comment_lineage(*, db_url: Optional[str] = None, conn: Optional[Connection] = None) -> int:
    """Backfill root_comment_id/comment_level for existing comments.

    Returns number of rows updated.
    """

    sql = """
        WITH RECURSIVE comment_lineage AS (
            SELECT
                c.comment_id,
                c.parent_comment_id,
                c.comment_id AS root_comment_id,
                0::SMALLINT AS comment_level
            FROM comments c
            WHERE c.parent_comment_id IS NULL
            UNION ALL
            SELECT
                child.comment_id,
                child.parent_comment_id,
                lineage.root_comment_id,
                (lineage.comment_level + 1)::SMALLINT AS comment_level
            FROM comments child
            JOIN comment_lineage lineage ON child.parent_comment_id = lineage.comment_id
            WHERE lineage.comment_level < 32
        ),
        resolved_lineage AS (
            SELECT
                c.comment_id,
                COALESCE(
                    l.root_comment_id,
                    CASE
                        WHEN c.parent_comment_id IS NULL THEN c.comment_id
                        ELSE c.parent_comment_id
                    END
                ) AS root_comment_id,
                COALESCE(
                    l.comment_level,
                    CASE
                        WHEN c.parent_comment_id IS NULL THEN 0
                        ELSE 1
                    END
                )::SMALLINT AS comment_level
            FROM comments c
            LEFT JOIN comment_lineage l ON l.comment_id = c.comment_id
        )
        UPDATE comments c
        SET
            root_comment_id = r.root_comment_id,
            comment_level = r.comment_level
        FROM resolved_lineage r
        WHERE c.comment_id = r.comment_id
          AND (
              c.root_comment_id IS DISTINCT FROM r.root_comment_id
              OR c.comment_level IS DISTINCT FROM r.comment_level
          )
    """

    def _run(target_conn: Connection) -> int:
        _ensure_columns_with_conn(target_conn)
        with target_conn.cursor() as cur:
            cur.execute(sql)
            return int(cur.rowcount or 0)

    if conn is not None:
        return _run(conn)

    resolved_db_url = get_database_url(db_url)
    with connect(resolved_db_url) as managed_conn:
        updated = _run(managed_conn)
        managed_conn.commit()
        return updated


def ensure_comment_lineage_columns(*, db_url: Optional[str] = None, conn: Optional[Connection] = None) -> None:
    """Ensure lineage columns/indexes exist; safe to call repeatedly."""

    if conn is not None:
        _ensure_columns_with_conn(conn)
        return

    resolved_db_url = get_database_url(db_url)
    with connect(resolved_db_url) as managed_conn:
        _ensure_columns_with_conn(managed_conn)
        managed_conn.commit()
