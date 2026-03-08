from pathlib import Path

import pytest


psycopg = pytest.importorskip("psycopg")
pytest.importorskip("yaml")
pytest.importorskip("selenium")

from scraper.config import PipelineConfig
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
        proxies_file=None,
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
