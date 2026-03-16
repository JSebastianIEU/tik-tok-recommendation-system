# ruff: noqa: E402
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("TikTokApi")

from scraper import comment_enrichment as ce


class _Cursor:
    def __init__(self, *, fetchall_rows=None, fetchone_row=None, rowcount=0):
        self.fetchall_rows = fetchall_rows or []
        self.fetchone_row = fetchone_row
        self.rowcount = rowcount
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self.fetchall_rows

    def fetchone(self):
        return self.fetchone_row


class _Conn:
    def __init__(self, cursors=None, *, commit_raises=False):
        self._cursors = list(cursors or [_Cursor()])
        self._idx = 0
        self.commits = 0
        self.rollbacks = 0
        self.closed = 0
        self._commit_raises = commit_raises

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        if self._idx < len(self._cursors):
            cur = self._cursors[self._idx]
            self._idx += 1
            return cur
        return self._cursors[-1]

    def commit(self):
        self.commits += 1
        if self._commit_raises:
            self._commit_raises = False
            raise RuntimeError("commit failed")

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed += 1


class _ApiCtx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _args(**overrides):
    base = {
        "db_url": None,
        "ms_token": None,
        "limit": 5,
        "include_existing": False,
        "max_existing_comments": 0,
        "comments": 3,
        "replies": 1,
        "min_likes_for_replies": 0,
        "delay": 0.0,
        "max_attempts_per_video": 5,
        "retry_backoff_base_sec": 60.0,
        "stale_running_minutes": 30,
        "proxies_file": None,
        "summary_path": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_seed_jobs_from_videos_only_missing(monkeypatch):
    cur = _Cursor(rowcount=3)
    conn = _Conn([cur])
    monkeypatch.setattr(ce, "connect", lambda _db_url: conn)

    inserted = ce._seed_jobs_from_videos(
        "postgresql://db",
        limit=10,
        include_existing=False,
        max_existing_comments=0,
    )

    assert inserted == 3
    sql, params = cur.executed[0]
    assert "NOT EXISTS" in sql
    assert params == (10,)


def test_seed_jobs_from_videos_include_existing(monkeypatch):
    cur = _Cursor(rowcount=1)
    conn = _Conn([cur])
    monkeypatch.setattr(ce, "connect", lambda _db_url: conn)

    ce._seed_jobs_from_videos(
        "postgresql://db",
        limit=3,
        include_existing=True,
        max_existing_comments=0,
    )
    sql, _ = cur.executed[0]
    assert "AND TRUE" in sql


def test_seed_jobs_from_videos_low_coverage_mode(monkeypatch):
    cur = _Cursor(rowcount=2)
    conn = _Conn([cur])
    monkeypatch.setattr(ce, "connect", lambda _db_url: conn)

    ce._seed_jobs_from_videos(
        "postgresql://db",
        limit=7,
        include_existing=False,
        max_existing_comments=2,
    )
    sql, params = cur.executed[0]
    assert "COUNT(*)" in sql
    assert params == (2, 7)


def test_select_candidates_from_jobs(monkeypatch):
    cur = _Cursor(fetchall_rows=[("v1",), ("v2",)])
    conn = _Conn([cur])

    out = ce._select_candidates_from_jobs(conn, limit=2, max_attempts_per_video=4)

    assert out == ["v1", "v2"]
    sql, params = cur.executed[0]
    assert "status IN ('pending', 'failed')" in sql
    assert params == (4, 2)


def test_mark_retry_or_exhausted_returns_failed_then_exhausted():
    cur_failed = _Cursor()
    conn_failed = _Conn([cur_failed])
    status_failed = ce._mark_retry_or_exhausted(
        conn_failed,
        video_id="v1",
        attempt_count=1,
        max_attempts_per_video=3,
        last_error="empty",
        retry_backoff_base_sec=60.0,
    )
    assert status_failed == "failed"

    cur_exhausted = _Cursor()
    conn_exhausted = _Conn([cur_exhausted])
    status_exhausted = ce._mark_retry_or_exhausted(
        conn_exhausted,
        video_id="v1",
        attempt_count=3,
        max_attempts_per_video=3,
        last_error="empty",
        retry_backoff_base_sec=60.0,
    )
    assert status_exhausted == "exhausted"


def test_run_enrichment_no_candidates(monkeypatch):
    conn = _Conn()
    monkeypatch.setattr(ce, "get_database_url", lambda _db=None: "postgresql://db")
    monkeypatch.setattr(ce, "get_ms_token", lambda _ms=None: "tok")
    monkeypatch.setattr(ce, "_ensure_jobs_table", lambda *_a, **_k: None)
    monkeypatch.setattr(ce, "_recover_stale_running_jobs", lambda *_a, **_k: None)
    monkeypatch.setattr(ce, "_seed_jobs_from_videos", lambda *_a, **_k: 0)
    monkeypatch.setattr(ce, "TikTokApi", lambda: _ApiCtx())
    monkeypatch.setattr(ce, "connect", lambda _db: conn)

    async def _fake_create_session(*_a, **_k):
        return None

    monkeypatch.setattr(ce, "_create_api_session", _fake_create_session)
    monkeypatch.setattr(ce, "_select_candidates_from_jobs", lambda *_a, **_k: [])

    code = asyncio.run(ce._run_enrichment(_args()))
    assert code == 0
    assert conn.closed == 1


def test_run_enrichment_success_with_session_recovery(monkeypatch):
    conn = _Conn()
    monkeypatch.setattr(ce, "get_database_url", lambda _db=None: "postgresql://db")
    monkeypatch.setattr(ce, "get_ms_token", lambda _ms=None: "tok")
    monkeypatch.setattr(ce, "_ensure_jobs_table", lambda *_a, **_k: None)
    monkeypatch.setattr(ce, "_recover_stale_running_jobs", lambda *_a, **_k: None)
    monkeypatch.setattr(ce, "_seed_jobs_from_videos", lambda *_a, **_k: 0)
    monkeypatch.setattr(ce, "TikTokApi", lambda: _ApiCtx())
    monkeypatch.setattr(ce, "connect", lambda _db: conn)
    monkeypatch.setattr(ce, "_select_candidates_from_jobs", lambda *_a, **_k: ["v1", "v2"])
    monkeypatch.setattr(ce, "_mark_running", lambda *_a, **_k: 1)

    calls = {"session": 0, "writes": 0, "done": 0, "retry": 0}

    async def _fake_create_session(*_a, **_k):
        calls["session"] += 1

    async def _fake_fetch(_api, video_id, *_a, **_k):
        if video_id == "v1" and calls["session"] == 1:
            return ([{"comment_id": "c1"}], 0, 0, True)
        if video_id == "v1":
            return ([{"comment_id": "c1r"}], 0, 0, False)
        return ([], 1, 0, False)

    def _fake_write(*_a, **_k):
        calls["writes"] += 1
        return 2

    def _fake_done(*_a, **_k):
        calls["done"] += 1

    def _fake_retry(*_a, **_k):
        calls["retry"] += 1
        return "failed"

    monkeypatch.setattr(ce, "_create_api_session", _fake_create_session)
    monkeypatch.setattr(ce, "fetch_comments_for_video", _fake_fetch)
    monkeypatch.setattr(ce, "write_comments_for_existing_video", _fake_write)
    monkeypatch.setattr(ce, "_mark_done", _fake_done)
    monkeypatch.setattr(ce, "_mark_retry_or_exhausted", _fake_retry)

    code = asyncio.run(ce._run_enrichment(_args(limit=2)))
    assert code == 0
    assert calls["session"] >= 2  # initial + recovery
    assert calls["writes"] == 1
    assert calls["done"] == 1
    assert calls["retry"] == 1  # second video had empty comments
    assert conn.commits >= 2
    assert conn.closed == 1


def test_run_enrichment_reconnects_on_commit_failure(monkeypatch):
    conn1 = _Conn(commit_raises=True)
    conn2 = _Conn()
    conns = iter([conn1, conn2])

    monkeypatch.setattr(ce, "get_database_url", lambda _db=None: "postgresql://db")
    monkeypatch.setattr(ce, "get_ms_token", lambda _ms=None: "tok")
    monkeypatch.setattr(ce, "_ensure_jobs_table", lambda *_a, **_k: None)
    monkeypatch.setattr(ce, "_recover_stale_running_jobs", lambda *_a, **_k: None)
    monkeypatch.setattr(ce, "_seed_jobs_from_videos", lambda *_a, **_k: 0)
    monkeypatch.setattr(ce, "TikTokApi", lambda: _ApiCtx())
    monkeypatch.setattr(ce, "connect", lambda _db: next(conns))
    monkeypatch.setattr(ce, "_select_candidates_from_jobs", lambda *_a, **_k: ["v1"])
    monkeypatch.setattr(ce, "_mark_running", lambda *_a, **_k: 1)
    monkeypatch.setattr(ce, "_mark_retry_or_exhausted", lambda *_a, **_k: "failed")

    async def _fake_create_session(*_a, **_k):
        return None

    async def _fake_fetch(*_a, **_k):
        return ([], 1, 0, False)

    monkeypatch.setattr(ce, "_create_api_session", _fake_create_session)
    monkeypatch.setattr(ce, "fetch_comments_for_video", _fake_fetch)

    code = asyncio.run(ce._run_enrichment(_args(limit=1)))
    assert code == 0
    assert conn1.closed == 1
    assert conn2.closed == 1


def test_run_enrichment_writes_summary_path(monkeypatch, tmp_path):
    conn = _Conn()
    summary_path = tmp_path / "comment_summary.json"

    monkeypatch.setattr(ce, "get_database_url", lambda _db=None: "postgresql://db")
    monkeypatch.setattr(ce, "get_ms_token", lambda _ms=None: "tok")
    monkeypatch.setattr(ce, "_ensure_jobs_table", lambda *_a, **_k: None)
    monkeypatch.setattr(ce, "_recover_stale_running_jobs", lambda *_a, **_k: None)
    monkeypatch.setattr(ce, "_seed_jobs_from_videos", lambda *_a, **_k: 4)
    monkeypatch.setattr(ce, "TikTokApi", lambda: _ApiCtx())
    monkeypatch.setattr(ce, "connect", lambda _db: conn)
    monkeypatch.setattr(ce, "_select_candidates_from_jobs", lambda *_a, **_k: [])

    async def _fake_create_session(*_a, **_k):
        return None

    monkeypatch.setattr(ce, "_create_api_session", _fake_create_session)

    code = asyncio.run(ce._run_enrichment(_args(summary_path=str(summary_path))))
    assert code == 0
    assert summary_path.exists()
    payload = summary_path.read_text(encoding="utf-8")
    assert "processed" in payload


def test_main_invokes_async_run(monkeypatch):
    called = {"ok": False}

    async def _fake_run(_args):
        called["ok"] = True
        return 0

    monkeypatch.setattr(ce, "_run_enrichment", _fake_run)
    assert ce.main(["--limit", "1"]) == 0
    assert called["ok"] is True
