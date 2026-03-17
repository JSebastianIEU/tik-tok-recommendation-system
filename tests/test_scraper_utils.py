# ruff: noqa: E402
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("yaml")
pytest.importorskip("selenium")

from scraper import config as scraper_config
from scraper import json_merge
from scraper import pipeline
from scraper import tiktok_post_scraper as post_scraper
from scraper.db import client as db_client
from scraper.db import schema as db_schema


def test_json_merge_key_and_score():
    row = {
        "success": True,
        "normalized": {
            "video": {"video_id": "v1", "likes": 1, "comments_count": 2, "shares": 3, "plays": 4},
            "comments": [{"a": 1}, {"b": 2}],
        },
    }
    assert json_merge._record_key(row) == "video_id:v1"
    assert json_merge._record_score(row) >= 1000


def test_merge_jsonl_files_prefers_richer_record(tmp_path: Path):
    in1 = tmp_path / "a.jsonl"
    in2 = tmp_path / "b.jsonl"
    out = tmp_path / "out.jsonl"
    row_low = {"success": True, "url": "u1", "normalized": {"video": {"video_id": "v1"}, "comments": []}}
    row_high = {"success": True, "url": "u1", "normalized": {"video": {"video_id": "v1", "likes": 10}, "comments": [1, 2, 3]}}
    in1.write_text(json.dumps(row_low) + "\n", encoding="utf-8")
    in2.write_text(json.dumps(row_high) + "\n", encoding="utf-8")

    summary = json_merge.merge_jsonl_files([str(in1), str(in2)], str(out))
    assert summary.rows_read == 2
    assert summary.rows_written == 1
    saved = json.loads(out.read_text(encoding="utf-8").strip())
    assert len(saved["normalized"]["comments"]) == 3


def test_merge_jsonl_files_errors(tmp_path: Path):
    with pytest.raises(ValueError):
        json_merge.merge_jsonl_files([], str(tmp_path / "x.jsonl"))
    with pytest.raises(FileNotFoundError):
        json_merge.merge_jsonl_files([str(tmp_path / "missing.jsonl")], str(tmp_path / "x.jsonl"))


def test_config_helpers_and_load(tmp_path: Path):
    assert scraper_config._str_list([" a ", "b"], "x") == ["a", "b"]
    with pytest.raises(ValueError):
        scraper_config._str_list("bad", "x")
    with pytest.raises(ValueError):
        scraper_config._get_int({"a": "1"}, "a", 1)
    with pytest.raises(ValueError):
        scraper_config._get_bool({"a": "true"}, "a", True)
    assert scraper_config._normalize_modes([]) == ["keyword", "hashtag"]
    assert scraper_config._normalize_hashtags([" #x ", "#x", "y"]) == ["x", "y"]
    assert scraper_config._parse_output_jsonl({"output_raw_jsonl": True})[0] is True

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
keywords: ["k1"]
hashtags: ["#h1"]
modes_enabled: ["keyword", "hashtag"]
output_raw_jsonl:
  enabled: true
  path: "raw/out.jsonl"
""",
        encoding="utf-8",
    )
    loaded = scraper_config.load_pipeline_config(cfg)
    assert loaded.keywords == ["k1"]
    assert loaded.hashtags == ["h1"]
    assert loaded.output_raw_jsonl is True


def test_read_structured_config_json(tmp_path: Path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"keywords":["a"],"hashtags":["b"]}', encoding="utf-8")
    out = scraper_config._read_structured_config(cfg)
    assert out["keywords"] == ["a"]


def test_db_client_url_and_connection_context(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError):
        db_client.get_database_url()
    monkeypatch.setenv("DATABASE_URL", "postgresql://env")
    assert db_client.get_database_url(None) == "postgresql://env"
    assert db_client.get_database_url("postgresql://override") == "postgresql://override"

    class FakeConn:
        def __init__(self):
            self.committed = False
            self.rolled_back = False
            self.closed = False

        def commit(self):
            self.committed = True

        def rollback(self):
            self.rolled_back = True

        def close(self):
            self.closed = True

    conn = FakeConn()
    monkeypatch.setattr(db_client, "connect", lambda _db=None: conn)
    with db_client.get_connection():
        pass
    assert conn.committed and conn.closed

    conn2 = FakeConn()
    monkeypatch.setattr(db_client, "connect", lambda _db=None: conn2)
    with pytest.raises(ValueError):
        with db_client.get_connection():
            raise ValueError("boom")
    assert conn2.rolled_back and conn2.closed


def test_apply_schema_reads_sql_and_executes(monkeypatch, tmp_path: Path):
    sql_path = tmp_path / "schema.sql"
    sql_path.write_text("SELECT 1;", encoding="utf-8")
    seen: dict[str, Any] = {}

    class Cur:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute(self, sql):
            seen["sql"] = sql

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def cursor(self):
            return Cur()

    monkeypatch.setattr(db_schema, "get_connection", lambda _db=None: Conn())
    out = db_schema.apply_schema(schema_path=str(sql_path))
    assert out == sql_path
    assert seen["sql"] == "SELECT 1;"


def test_pipeline_helpers_cover_edge_cases(tmp_path: Path):
    assert pipeline._safe_int("1") == 1
    assert pipeline._safe_int("bad") is None
    assert pipeline._normalize_video_url("/@a/video/1?x=1") == "https://www.tiktok.com/@a/video/1"
    assert pipeline._normalize_video_url("https://x.com") is None
    assert pipeline._build_video_url_from_raw({"id": "1", "author": {"uniqueId": "abc"}}) == "https://www.tiktok.com/@abc/video/1"
    assert pipeline._video_metric_tuple({"stats": {"playCount": "3", "diggCount": "2"}}) == (3, 2)

    ranked = pipeline._rank_raw_videos(
        [
            {"id": "1", "author": {"uniqueId": "a"}, "stats": {"playCount": 1, "diggCount": 1}},
            {"id": "2", "author": {"uniqueId": "a"}, "stats": {"playCount": 10, "diggCount": 1}},
        ]
    )
    assert ranked[0]["id"] == "2"

    candidates = pipeline._to_candidates(
        [
            {"id": "1", "author": {"uniqueId": "a"}, "stats": {"playCount": 2, "diggCount": 1}},
            {"id": "1", "author": {"uniqueId": "a"}, "stats": {"playCount": 2, "diggCount": 1}},
        ],
        mode="hashtag",
        query="q",
        limit=10,
    )
    assert len(candidates) == 1

    class Obj:
        def as_dict(self):
            return {"x": 1}

    assert pipeline._video_as_dict({"a": 1}) == {"a": 1}
    assert pipeline._video_as_dict(Obj()) == {"x": 1}

    async def _collect():
        async def gen():
            yield {"a": 1}
            yield {"b": 2}

        return await pipeline._collect_video_dicts(gen(), limit=1)

    out = asyncio.run(_collect())
    assert out == [{"a": 1}]


def test_discover_keyword_with_api_prefers_search_type():
    class Search:
        async def search_type(self, term, entity, count=10):
            assert term == "fitness"
            assert entity == "item"
            assert count == 2
            for row in [{"id": "1"}, {"id": "2"}]:
                yield row

    class Api:
        search = Search()

    rows = asyncio.run(pipeline._discover_keyword_with_api(Api(), keyword="fitness", limit=2))
    assert rows == [{"id": "1"}, {"id": "2"}]


def test_discover_keyword_with_api_falls_back_to_videos():
    class Search:
        def videos(self, query=None, count=10, **kwargs):
            assert query == "fitness" or kwargs.get("keyword") == "fitness" or kwargs.get("keywords") == "fitness"
            assert count == 2

            async def gen():
                yield {"id": "10"}
                yield {"id": "11"}

            return gen()

    class Api:
        search = Search()

    rows = asyncio.run(pipeline._discover_keyword_with_api(Api(), keyword="fitness", limit=2))
    assert rows == [{"id": "10"}, {"id": "11"}]


def test_discover_keyword_with_api_requires_supported_search_api():
    class Search:
        pass

    class Api:
        search = Search()

    with pytest.raises(RuntimeError, match="search\\.videos and search\\.search_type are not available"):
        asyncio.run(pipeline._discover_keyword_with_api(Api(), keyword="fitness", limit=2))


def test_tiktok_post_scraper_url_file_helpers(tmp_path: Path, monkeypatch):
    line = '{"id":"123","author":{"uniqueId":"abc"}}'
    assert post_scraper._video_url_from_api_jsonl_line(line) == "https://www.tiktok.com/@abc/video/123"
    assert post_scraper._video_url_from_api_jsonl_line("{bad") is None

    txt = tmp_path / "urls.txt"
    txt.write_text("https://x/video/1\n#comment\nhttps://x/video/1\n", encoding="utf-8")
    urls = post_scraper._read_urls_from_file(txt, jsonl_format=False)
    assert urls == ["https://x/video/1"]

    js = tmp_path / "urls.jsonl"
    js.write_text(line + "\n" + line + "\n", encoding="utf-8")
    urls2 = post_scraper._read_urls_from_file(js, jsonl_format=True)
    assert urls2 == ["https://www.tiktok.com/@abc/video/123"]

    monkeypatch.setenv("DATABASE_URL", "postgresql://env")
    assert post_scraper._resolve_db_url(None) == "postgresql://env"
