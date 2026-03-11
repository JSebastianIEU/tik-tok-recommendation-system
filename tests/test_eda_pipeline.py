from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from eda import pipeline
def test_build_run_id_and_dir(tmp_path: Path):
    rid = pipeline.build_run_id(datetime(2026, 3, 11, 10, 0, 0, tzinfo=timezone.utc))
    assert rid == "20260311_100000"

    out = pipeline.build_run_dir(tmp_path, run_id=rid)
    assert out.exists()
    assert out.name == rid


def test_load_plan_validates(tmp_path: Path):
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        """
        datasets:
          - name: videos
            limit: 100
            format: jsonl
        """,
        encoding="utf-8",
    )
    reqs = pipeline.load_plan(plan)
    assert len(reqs) == 1
    assert reqs[0].name == "videos"

    bad = tmp_path / "bad.yaml"
    bad.write_text("datasets: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        pipeline.load_plan(bad)


def test_run_plan_single_page(monkeypatch, tmp_path: Path):
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        """
        datasets:
          - name: videos
            limit: 2
            all_pages: false
            format: jsonl
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(pipeline, "resolve_db_url", lambda **_k: "postgresql://db")
    monkeypatch.setattr(
        pipeline,
        "fetch_dataset",
        lambda **_k: ([{"video_id": "v1", "created_at": "2026-03-10", "video_author_id": "a1"}, {"video_id": "v2", "created_at": "2026-03-09", "video_author_id": "a2"}], {"cursor_id": "v2", "cursor_created_at": "2026-03-09T00:00:00+00:00"}),
    )

    manifest = pipeline.run_plan(
        plan_path=plan,
        output_root=tmp_path / "raw",
        run_id="rid1",
        registry_path=tmp_path / "runs.jsonl",
    )
    assert manifest["run_id"] == "rid1"
    assert manifest["datasets"][0]["rows"] == 2

    out_file = Path(manifest["datasets"][0]["path"])
    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    manifest_file = Path(manifest["output_dir"]) / "manifest.json"
    assert manifest_file.exists()
    parsed = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert parsed["run_id"] == "rid1"
    assert "lineage" in parsed
    assert (Path(manifest["output_dir"]) / "quality_report.json").exists()


def test_run_plan_all_pages(monkeypatch, tmp_path: Path):
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        """
        datasets:
          - name: comments
            limit: 3
            all_pages: true
            format: csv
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(pipeline, "resolve_db_url", lambda **_k: "postgresql://db")
    monkeypatch.setattr(
        pipeline,
        "fetch_dataset",
        lambda **_k: ([{"comment_id": "c1", "video_id": "v1", "text": "hello", "row_created_at": "2026-03-10"}], None),
    )

    manifest = pipeline.run_plan(
        plan_path=plan,
        output_root=tmp_path / "raw",
        run_id="rid2",
        registry_path=tmp_path / "runs.jsonl",
    )
    assert manifest["datasets"][0]["rows"] == 1
    assert manifest["datasets"][0]["format"] == "csv"
    assert manifest["datasets"][0]["next_cursor"] is None

    out_file = Path(manifest["datasets"][0]["path"])
    assert out_file.exists()
    assert out_file.suffix == ".csv"


def test_resolve_db_url_from_profile(monkeypatch, tmp_path: Path):
    profiles = tmp_path / "profiles.yaml"
    profiles.write_text(
        """
profiles:
  staging:
    db_env_var: DATABASE_URL_STAGING
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_URL_STAGING", "postgresql://profile-db")

    resolved = pipeline.resolve_db_url(db_url=None, profile="staging", profiles_path=profiles)
    assert resolved == "postgresql://profile-db"


def test_run_plan_builds_silver(monkeypatch, tmp_path: Path):
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        """
        datasets:
          - name: comments
            limit: 2
            all_pages: false
            format: jsonl
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(pipeline, "resolve_db_url", lambda **_k: "postgresql://db")
    monkeypatch.setattr(
        pipeline,
        "fetch_dataset",
        lambda **_k: (
            [
                {"comment_id": "c1", "video_id": "v1", "text": " hi ", "row_created_at": "2026-03-10"},
                {"comment_id": "c1", "video_id": "v1", "text": " hi ", "row_created_at": "2026-03-11"},
            ],
            None,
        ),
    )

    manifest = pipeline.run_plan(
        plan_path=plan,
        output_root=tmp_path / "bronze",
        silver_output_root=tmp_path / "silver",
        run_id="rid3",
        registry_path=tmp_path / "runs.jsonl",
        build_silver=True,
    )
    assert manifest["silver_output_dir"] is not None
    ds = manifest["datasets"][0]
    assert ds["silver"] is not None
    assert ds["silver"]["rows"] == 1
    assert Path(ds["silver"]["path"]).exists()
    assert (Path(manifest["silver_output_dir"]) / "quality_report.json").exists()
