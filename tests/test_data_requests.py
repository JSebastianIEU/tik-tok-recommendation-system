# ruff: noqa: E402
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("psycopg")

from scraper import data_requests as dr


class _Cursor:
    def __init__(self, rows):
        self._rows = rows
        self.executed: list[tuple[str, tuple]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, tuple(params or ())))

    def fetchall(self):
        return self._rows


class _Conn:
    def __init__(self, rows):
        self.cursor_obj = _Cursor(rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self, row_factory=None):
        return self.cursor_obj


def _dt(v: str) -> datetime:
    return datetime.fromisoformat(v.replace("Z", "+00:00"))


def test_validate_limit_and_offset_guards():
    with pytest.raises(ValueError):
        dr._validate_limit(0)
    with pytest.raises(ValueError):
        dr._validate_limit(dr.MAX_LIMIT + 1)
    with pytest.raises(ValueError):
        dr._validate_offset(-1)


def test_parse_since_and_cursor_parsing():
    since = dr._parse_since("2026-03-01T00:00:00Z")
    assert since is not None
    assert since.tzinfo is not None

    with pytest.raises(ValueError):
        dr._parse_since("not-a-date")
    with pytest.raises(ValueError):
        dr._normalize_cursor("2026-03-01T00:00:00Z", None)


def test_get_videos_data_returns_rows_and_cursor(monkeypatch):
    rows = [
        {
            "video_id": "v2",
            "created_at": _dt("2026-03-10T10:00:00Z"),
            "hashtags": ["fitness"],
        },
        {
            "video_id": "v1",
            "created_at": _dt("2026-03-09T09:00:00Z"),
            "hashtags": ["workout"],
        },
    ]
    conn = _Conn(rows)
    monkeypatch.setattr(dr, "connect", lambda _db=None: conn)

    page = dr.get_videos_data(limit=2, offset=3, since="2026-03-01T00:00:00Z", db_url="postgresql://db")

    assert page.count == 2
    assert page.rows[0]["video_id"] == "v2"
    assert page.next_cursor is not None
    assert page.next_cursor["cursor_id"] == "v1"

    sql, params = conn.cursor_obj.executed[0]
    assert "FROM videos v" in sql
    assert "ORDER BY COALESCE(v.created_at, TIMESTAMPTZ 'epoch') DESC, v.video_id DESC" in sql
    assert params[-2:] == (2, 3)


def test_get_comments_data_applies_cursor(monkeypatch):
    rows = [
        {
            "comment_id": "c1",
            "row_created_at": _dt("2026-03-09T01:00:00Z"),
        }
    ]
    conn = _Conn(rows)
    monkeypatch.setattr(dr, "connect", lambda _db=None: conn)

    page = dr.get_comments_data(
        limit=1,
        cursor_created_at="2026-03-10T00:00:00Z",
        cursor_id="c9",
        db_url="postgresql://db",
    )

    assert page.count == 1
    assert page.next_cursor is not None
    assert page.next_cursor["cursor_id"] == "c1"
    sql, _ = conn.cursor_obj.executed[0]
    assert "FROM comments c" in sql
    assert "c.comment_id < %s" in sql


def test_get_full_and_authors_queries(monkeypatch):
    full_conn = _Conn([{"video_id": "v1", "created_at": _dt("2026-03-10T00:00:00Z"), "comments": []}])
    author_conn = _Conn([{"author_id": "a1", "row_created_at": _dt("2026-03-08T00:00:00Z")}])
    conns = iter([full_conn, author_conn])
    monkeypatch.setattr(dr, "connect", lambda _db=None: next(conns))

    full_page = dr.get_full_data(limit=1)
    author_page = dr.get_authors_data(limit=1)

    assert full_page.count == 1
    assert author_page.count == 1
    assert "json_agg" in full_conn.cursor_obj.executed[0][0]
    assert "FROM authors a" in author_conn.cursor_obj.executed[0][0]


def test_export_rows_jsonl_and_csv(tmp_path: Path):
    rows = [{"id": "1", "tags": ["a", "b"], "meta": {"x": 1}}]

    out_jsonl = tmp_path / "out.jsonl"
    dr.export_rows(rows, output_path=out_jsonl, fmt="jsonl")
    assert out_jsonl.exists()
    assert '"id": "1"' in out_jsonl.read_text(encoding="utf-8")

    out_csv = tmp_path / "out.csv"
    dr.export_rows(rows, output_path=out_csv, fmt="csv")
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        got = list(reader)
    assert got[0]["id"] == "1"
    assert got[0]["tags"] == '["a", "b"]'
    assert got[0]["meta"] == '{"x": 1}'


def test_fetch_all_pages(monkeypatch):
    calls = []

    def _loader(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return dr.RetrievalPage(
                rows=[{"video_id": "v1"}],
                count=1,
                next_cursor={"cursor_created_at": "2026-03-10T00:00:00+00:00", "cursor_id": "v1"},
            )
        return dr.RetrievalPage(rows=[], count=0, next_cursor=None)

    monkeypatch.setitem(dr.DATASET_LOADERS, "videos", _loader)

    rows = dr._fetch_all_pages(dataset="videos", db_url="postgresql://db", page_limit=10, since=None)

    assert rows == [{"video_id": "v1"}]
    assert calls[0]["cursor_created_at"] is None
    assert calls[1]["cursor_id"] == "v1"


def test_main_rejects_offset_with_cursor(monkeypatch):
    monkeypatch.setattr(dr, "get_database_url", lambda _db=None: "postgresql://db")
    with pytest.raises(ValueError):
        dr.main(
            [
                "--dataset",
                "videos",
                "--offset",
                "10",
                "--cursor-created-at",
                "2026-03-01T00:00:00Z",
                "--cursor-id",
                "v1",
            ]
        )
