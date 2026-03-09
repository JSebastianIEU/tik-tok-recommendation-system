# ruff: noqa: E402
from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("psycopg")

from scraper.db import writer


class _Cursor:
    def __init__(self):
        self._last_sql = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):  # noqa: ARG002
        self._last_sql = sql

    def fetchone(self):
        if "SELECT 1 FROM videos" in self._last_sql:
            return None
        if "RETURNING video_snapshot_id" in self._last_sql:
            return (123,)
        return (1,)

    def fetchall(self):
        if "SELECT hashtag_id, tag FROM hashtags" in self._last_sql:
            return [(10, "a"), (11, "b")]
        return []


class _Conn:
    def cursor(self):
        return _Cursor()

    def commit(self):
        return None

    def rollback(self):
        return None

    def close(self):
        return None


def _sample_normalized():
    now = datetime.now(timezone.utc).isoformat()
    return {
        "author": {"author_id": "au1", "username": "u1", "display_name": "n1"},
        "video": {
            "video_id": "v1",
            "author_id": "au1",
            "url": "https://www.tiktok.com/@u1/video/v1",
            "caption": "cap",
            "hashtags": ["#a", "b"],
            "audio_id": "m1",
            "audio_name": "song",
            "duration_sec": 10,
            "thumbnail_url": "http://img",
            "created_at": now,
            "scraped_at": now,
            "likes": 1,
            "comments_count": 2,
            "shares": 3,
            "plays": 4,
        },
        "authorMetricSnapshot": {"author_id": "au1", "video_id": "v1", "follower_count": 100},
        "comments": [{"comment_id": "c1", "author_id": "cu1", "username": "cuser", "text": "t"}],
    }


def test_normalize_hashtag_tags():
    assert writer._normalize_hashtag_tags(["#a", " a ", "#a", "", "b"]) == ["a", "b"]


def test_ensure_hashtags_and_link_video_hashtags():
    conn = _Conn()
    ids = writer._ensure_hashtags(conn, ["#a", "b", "b"])
    assert ids == [10, 11]
    writer._link_video_hashtags(conn, "v1", ids)  # should not raise


def test_write_with_connection_skip_existing_from_set():
    conn = _Conn()
    wrote = writer._write_with_connection(
        conn,
        _sample_normalized(),
        skip_existing=True,
        existing_video_ids={"v1"},
    )
    assert wrote is False


def test_write_with_connection_executes_full_path():
    conn = _Conn()
    wrote = writer._write_with_connection(
        conn,
        _sample_normalized(),
        scrape_ctx=writer.ScrapeContext(
            scrape_run_id=writer.uuid.uuid4(),
            source="test",
            started_at=datetime.now(timezone.utc),
        ),
        position=1,
        skip_existing=False,
    )
    assert wrote is True


def test_batched_writer_commits_and_rolls_back(monkeypatch):
    conn = _Conn()
    monkeypatch.setattr(writer, "connect", lambda _db=None: conn)
    monkeypatch.setattr(writer, "write_normalized_record", lambda *_a, **_k: True)
    with writer.BatchedRecordWriter(db_url="postgresql://x", commit_every=1) as bw:
        assert bw.write(_sample_normalized())

    monkeypatch.setattr(writer, "connect", lambda _db=None: conn)
    with writer.BatchedRecordWriter(db_url="postgresql://x", commit_every=10) as bw:
        bw.rollback()


def test_write_normalized_record_uses_managed_connection(monkeypatch):
    class _Ctx:
        def __enter__(self):
            return _Conn()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(writer, "get_connection", lambda _db=None: _Ctx())
    assert writer.write_normalized_record(_sample_normalized(), db_url="postgresql://x") is True
