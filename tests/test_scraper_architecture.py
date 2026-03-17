# ruff: noqa: E402
from pathlib import Path

import pytest


psycopg = pytest.importorskip("psycopg")
pytest.importorskip("yaml")
pytest.importorskip("selenium")

from scraper.config import PipelineConfig
from scraper import cli
from scraper.pipeline import _resolve_db_url as resolve_pipeline_db_url
from scraper.run_full_scale import _build_jobs, _resolve_db_url as resolve_full_scale_db_url


def test_full_scale_db_url_precedence(monkeypatch):
    config = {"db_url": "postgresql://from-config"}
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-env")

    assert resolve_full_scale_db_url("postgresql://from-arg", config) == "postgresql://from-arg"
    assert resolve_full_scale_db_url(None, config) == "postgresql://from-env"

    monkeypatch.delenv("DATABASE_URL")
    assert resolve_full_scale_db_url(None, config) == "postgresql://from-config"


def test_pipeline_db_url_precedence(monkeypatch):
    cfg = PipelineConfig(
        keywords=["a"],
        hashtags=["b"],
        per_query_video_limit=1,
        max_comments_per_video=0,
        comment_sort="top",
        modes_enabled=["keyword"],
        concurrency=1,
        headless=True,
        db_url="postgresql://from-config",
        output_raw_jsonl=False,
        output_raw_jsonl_path=str(Path("tmp.jsonl").resolve()),
        source_label="test",
        ms_token=None,
        tiktok_browser="webkit",
    )
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-env")

    assert (
        resolve_pipeline_db_url(cfg, db_url_override="postgresql://from-override")
        == "postgresql://from-override"
    )
    assert resolve_pipeline_db_url(cfg) == "postgresql://from-env"

    monkeypatch.delenv("DATABASE_URL")
    assert resolve_pipeline_db_url(cfg) == "postgresql://from-config"


def test_build_jobs_creates_expected_sources(tmp_path: Path):
    config = {
        "hashtags": [{"name": "fitness life", "count": 25}],
        "keywords": [{"name": "morning routine", "count": 10}],
    }
    jobs = _build_jobs(config, tmp_path)
    assert len(jobs) == 2
    assert jobs[0].key == "hashtag:fitness life"
    assert jobs[1].key == "keyword:morning routine"
    assert jobs[0].output_path.name == "full_scale_hashtag_fitness_life.jsonl"
    assert jobs[1].output_path.name == "full_scale_keyword_morning_routine.jsonl"


def test_build_jobs_honors_member_style_defaults_and_modes(tmp_path: Path):
    config = {
        "hashtags": ["fitness", "cooking"],
        "keywords": ["morning routine"],
        "per_query_video_limit": 40,
        "modes_enabled": ["keyword"],
    }
    jobs = _build_jobs(config, tmp_path)
    assert [job.key for job in jobs] == ["keyword:morning routine"]
    assert jobs[0].count == 40


def test_cli_parser_supports_scrape_comments():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "scrape-comments",
            "--limit",
            "10",
            "--comments",
            "3",
            "--replies",
            "1",
            "--max-attempts-per-video",
            "4",
            "--retry-backoff-base-sec",
            "120",
            "--stale-running-minutes",
            "15",
            "--max-existing-comments",
            "2",
            "--summary-path",
            "/tmp/comment_summary.json",
        ]
    )
    assert args.command == "scrape-comments"
    assert args.limit == 10
    assert args.comments == 3
    assert args.replies == 1
    assert args.max_attempts_per_video == 4
    assert args.retry_backoff_base_sec == 120
    assert args.stale_running_minutes == 15
    assert args.max_existing_comments == 2
    assert args.summary_path == "/tmp/comment_summary.json"


def test_cli_parser_supports_export_data():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "export-data",
            "--dataset",
            "comments",
            "--limit",
            "200",
            "--since",
            "2026-03-01T00:00:00Z",
            "--all",
            "--format",
            "csv",
            "--out",
            "/tmp/comments.csv",
        ]
    )
    assert args.command == "export-data"
    assert args.dataset == "comments"
    assert args.limit == 200
    assert args.since == "2026-03-01T00:00:00Z"
    assert args.all is True
    assert args.format == "csv"


def test_cli_parser_supports_comment_lineage_backfill():
    parser = cli.build_parser()
    args = parser.parse_args(["backfill-comment-lineage", "--db-url", "postgresql://db"])
    assert args.command == "backfill-comment-lineage"
    assert args.db_url == "postgresql://db"
