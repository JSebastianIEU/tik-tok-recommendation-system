# ruff: noqa: E402
from __future__ import annotations

import pytest

pytest.importorskip("psycopg")

from scraper.db import comment_lineage as cl


class _Cursor:
    def __init__(self):
        self.calls: list[tuple[str, tuple | None]] = []
        self.rowcount = 3

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))


class _Conn:
    def __init__(self):
        self.cur = _Cursor()
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self.cur

    def commit(self):
        self.commits += 1


def test_backfill_comment_lineage_with_existing_connection():
    conn = _Conn()
    updated = cl.backfill_comment_lineage(conn=conn)
    assert updated == 3
    sql_blob = "\n".join(call[0] for call in conn.cur.calls)
    assert "ALTER TABLE comments ADD COLUMN IF NOT EXISTS root_comment_id" in sql_blob
    assert "WITH RECURSIVE comment_lineage" in sql_blob


def test_ensure_columns_uses_connect_when_db_url_given(monkeypatch):
    conn = _Conn()
    monkeypatch.setattr(cl, "connect", lambda _db: conn)
    monkeypatch.setattr(cl, "get_database_url", lambda _db=None: "postgresql://db")

    cl.ensure_comment_lineage_columns(db_url="postgresql://db")

    assert conn.commits == 1
    sql_blob = "\n".join(call[0] for call in conn.cur.calls)
    assert "idx_comments_root_comment_id" in sql_blob
