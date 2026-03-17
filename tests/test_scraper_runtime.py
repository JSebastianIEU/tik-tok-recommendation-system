# ruff: noqa: E402
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("yaml")

from scraper.db import writer as db_writer
from scraper.run_full_scale import SourceJob, _run_scrape, main as run_full_scale_main


class _FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, _sql, _params=None):
        return None


class _FakeConn:
    def cursor(self):
        return _FakeCursor()


def test_writer_upserts_comment_author_before_comment_insert(monkeypatch):
    seen_author_ids: list[str] = []

    def _capture_author(_conn, author):
        seen_author_ids.append(str(author.get("author_id")))

    monkeypatch.setattr(db_writer, "_upsert_author", _capture_author)

    db_writer._upsert_comments_and_snapshots(
        _FakeConn(),
        video_id="video-1",
        comments=[
            {
                "comment_id": "comment-1",
                "author_id": "author-42",
                "username": "user42",
                "text": "hello",
            }
        ],
        video_snapshot_id=1,
        scraped_at=SimpleNamespace(),
    )

    assert "author-42" in seen_author_ids


def test_run_scrape_returns_rows_written(monkeypatch, tmp_path: Path):
    out_path = tmp_path / "out.jsonl"
    job = SourceJob(mode="hashtag", name="fitness", count=3, output_path=out_path)

    def _fake_run(_cmd, env, cwd):
        assert env["DATABASE_URL"] == "postgresql://db"
        assert cwd
        out_path.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("scraper.run_full_scale.subprocess.run", _fake_run)

    code, rows = _run_scrape(
        job,
        db_url="postgresql://db",
        env={"DATABASE_URL": "postgresql://db"},
    )
    assert code == 0
    assert rows == 2


def test_main_marks_empty_results_as_failed(monkeypatch, tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("hashtags: [{name: fitness, count: 10}]\nkeywords: []\n", encoding="utf-8")
    summary_path = tmp_path / "summary.json"

    monkeypatch.setenv("MS_TOKEN", "token")
    monkeypatch.setattr("scraper.run_full_scale._resolve_db_url", lambda *_: "postgresql://db")
    monkeypatch.setattr("scraper.run_full_scale.ensure_comment_lineage_columns", lambda **_kwargs: None)
    monkeypatch.setattr("scraper.run_full_scale._ensure_job_state_table", lambda *_: None)
    monkeypatch.setattr("scraper.run_full_scale._load_completed_jobs", lambda *_: set())
    monkeypatch.setattr("scraper.run_full_scale._mark_job_running", lambda *_: None)

    finished: list[tuple[str, str | None]] = []

    def _capture_finished(_db_url, _job, *, status, last_error=None):
        finished.append((status, last_error))

    monkeypatch.setattr("scraper.run_full_scale._mark_job_finished", _capture_finished)

    def _fake_run_scrape(job, **_kwargs):
        job.output_path.write_text("", encoding="utf-8")
        return 0, 0

    monkeypatch.setattr("scraper.run_full_scale._run_scrape", _fake_run_scrape)

    exit_code = run_full_scale_main(
        [
            str(cfg),
            "--retry-empty",
            "0",
            "--retry-delay",
            "0",
            "--delay",
            "0",
            "--summary-path",
            str(summary_path),
            "--no-resume",
        ]
    )

    assert exit_code == 1
    assert finished and finished[0][0] == "failed"
    assert finished[0][1] == "empty_result_set_after_retries"

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["jobs_failed"] == 1
    assert payload["sources"][0]["status"] == "failed_empty"
    assert payload["sources"][0]["rows_written"] == 0


def test_main_aborts_after_max_consecutive_empty(monkeypatch, tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "hashtags: [{name: fitness, count: 10}, {name: cooking, count: 10}]\nkeywords: []\n",
        encoding="utf-8",
    )
    summary_path = tmp_path / "summary.json"

    monkeypatch.setenv("MS_TOKEN", "token")
    monkeypatch.setattr("scraper.run_full_scale._resolve_db_url", lambda *_: "postgresql://db")
    monkeypatch.setattr("scraper.run_full_scale.ensure_comment_lineage_columns", lambda **_kwargs: None)
    monkeypatch.setattr("scraper.run_full_scale._ensure_job_state_table", lambda *_: None)
    monkeypatch.setattr("scraper.run_full_scale._load_completed_jobs", lambda *_: set())
    monkeypatch.setattr("scraper.run_full_scale._mark_job_running", lambda *_: None)
    monkeypatch.setattr("scraper.run_full_scale._mark_job_finished", lambda *_a, **_k: None)

    executed: list[str] = []

    def _fake_run_scrape(job, **_kwargs):
        executed.append(job.key)
        job.output_path.write_text("", encoding="utf-8")
        return 0, 0

    monkeypatch.setattr("scraper.run_full_scale._run_scrape", _fake_run_scrape)

    exit_code = run_full_scale_main(
        [
            str(cfg),
            "--retry-empty",
            "0",
            "--retry-delay",
            "0",
            "--delay",
            "0",
            "--summary-path",
            str(summary_path),
            "--max-consecutive-empty",
            "1",
            "--no-resume",
        ]
    )

    assert exit_code == 1
    assert executed == ["hashtag:fitness"]
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["aborted_early"] is True
    assert payload["abort_reason"] == "aborted_after_1_consecutive_empty_sources"


def test_main_uses_config_defaults_when_cli_omits(monkeypatch, tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "\n".join(
            [
                "hashtags: [{name: fitness}]",
                "keywords: []",
                "per_query_video_limit: 40",
                "comments: 7",
                "replies: 3",
                "min_likes_for_replies: 15",
                "retry_empty: 0",
                "retry_delay: 0",
                "delay: 0",
                "skip_existing: true",
                "no_resume: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = tmp_path / "summary.json"

    monkeypatch.setenv("MS_TOKEN", "token")
    monkeypatch.setattr("scraper.run_full_scale._resolve_db_url", lambda *_: "postgresql://db")
    monkeypatch.setattr("scraper.run_full_scale.ensure_comment_lineage_columns", lambda **_kwargs: None)
    monkeypatch.setattr("scraper.run_full_scale._ensure_job_state_table", lambda *_: None)
    monkeypatch.setattr("scraper.run_full_scale._load_completed_jobs", lambda *_: set())
    monkeypatch.setattr("scraper.run_full_scale._mark_job_running", lambda *_: None)
    monkeypatch.setattr("scraper.run_full_scale._mark_job_finished", lambda *_a, **_k: None)

    seen: dict[str, int | bool | None] = {}

    def _fake_run_scrape(job, **kwargs):
        seen["count"] = job.count
        seen["comments"] = kwargs.get("comments")
        seen["replies"] = kwargs.get("replies")
        seen["min_likes_for_replies"] = kwargs.get("min_likes_for_replies")
        seen["skip_existing"] = kwargs.get("skip_existing")
        job.output_path.write_text('{"ok":1}\n', encoding="utf-8")
        return 0, 1

    monkeypatch.setattr("scraper.run_full_scale._run_scrape", _fake_run_scrape)

    exit_code = run_full_scale_main([str(cfg), "--summary-path", str(summary_path)])

    assert exit_code == 0
    assert seen["count"] == 40
    assert seen["comments"] == 7
    assert seen["replies"] == 3
    assert seen["min_likes_for_replies"] == 15
    assert seen["skip_existing"] is True
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["resume_enabled"] is False
    assert payload["jobs_succeeded"] == 1


def test_fetch_comments_flags_session_invalid(monkeypatch):
    pytest.importorskip("TikTokApi")
    from scraper import scrape_tiktok_sample as sts

    class _BrokenVideo:
        async def comments(self, count=0):  # noqa: ARG002
            raise RuntimeError("No sessions created, please create sessions first")
            yield {}

    class _Api:
        def video(self, id):  # noqa: A002, ARG002
            return _BrokenVideo()

    comments, video_errs, reply_errs, session_invalid = asyncio.run(
        sts.fetch_comments_for_video(_Api(), "video-1", count=5, replies_per_comment=1)
    )
    assert comments == []
    assert video_errs == 1
    assert reply_errs == 0
    assert session_invalid is True


def test_fetch_comments_counts_reply_errors(monkeypatch):
    pytest.importorskip("TikTokApi")
    from scraper import scrape_tiktok_sample as sts

    class _Comment:
        id = "c1"
        text = "parent"
        likes_count = 999
        as_dict = {"cid": "c1", "user": {"uid": "u1", "unique_id": "usr"}, "digg_count": 999}

        async def replies(self, count=0):  # noqa: ARG002
            raise RuntimeError("reply api failed")
            yield {}

    class _Video:
        async def comments(self, count=0):  # noqa: ARG002
            yield _Comment()

    class _Api:
        def video(self, id):  # noqa: A002, ARG002
            return _Video()

    comments, video_errs, reply_errs, session_invalid = asyncio.run(
        sts.fetch_comments_for_video(_Api(), "video-1", count=5, replies_per_comment=1)
    )
    assert len(comments) == 1
    assert comments[0]["comment_id"] == "c1"
    assert video_errs == 0
    assert reply_errs == 1
    assert session_invalid is False


def test_scrape_batch_writes_output_jsonl(monkeypatch, tmp_path: Path):
    pytest.importorskip("selenium")
    from scraper import tiktok_post_scraper as sps

    def _fake_worker(urls, *_args, **_kwargs):
        return [{"success": True, "url": url, "normalized": {"video": {"video_id": url}}} for url in urls]

    monkeypatch.setattr(sps, "_worker_scrape_chunk", _fake_worker)

    out_path = tmp_path / "batch.jsonl"
    results = list(
        sps.scrape_tiktok_batch(
            ["u1", "u2", "u3"],
            workers=2,
            output_jsonl=str(out_path),
        )
    )
    assert len(results) == 3
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
