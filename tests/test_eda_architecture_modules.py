from __future__ import annotations

from pathlib import Path

from eda.src.contracts import validate_dataset_rows, validate_silver_rows
from eda.src.lineage import build_lineage
from eda.src.metadata_store import append_run_registry
from eda.src.quality import build_quality_scorecard


def test_validate_dataset_rows_contracts():
    ok = validate_dataset_rows(
        "videos",
        [{"video_id": "v1", "created_at": "2026-03-10", "video_author_id": "a1"}],
    )
    assert ok.valid is True

    bad = validate_dataset_rows("comments", [{"comment_id": "c1", "video_id": "v1"}])
    assert bad.valid is False
    assert "row_created_at" in bad.missing_required_columns or bad.row_missing_required_count >= 1

    silver_ok = validate_silver_rows(
        "comments",
        [
            {
                "comment_id": "c1",
                "video_id": "v1",
                "text": "x",
                "row_created_at": "2026-03-11",
                "text_length": 1,
                "is_reply": False,
            }
        ],
    )
    assert silver_ok.valid is True


def test_build_quality_scorecard():
    rows = [
        {"comment_id": "c1", "text": "a"},
        {"comment_id": "c1", "text": None},
    ]
    q = build_quality_scorecard("comments", rows)
    assert q["rows"] == 2
    assert q["duplicate_rate"] > 0.0
    assert "text" in q["null_rates"]


def test_build_lineage_and_registry(tmp_path: Path):
    lineage = build_lineage(db_url="postgresql://user:pw@host:5432/postgres", plan_path="/tmp/plan.yaml")
    assert lineage["db_fingerprint"]["host"] == "host"
    assert "generated_at" in lineage

    reg = tmp_path / "runs.jsonl"
    append_run_registry({"run_id": "r1"}, registry_path=reg)
    append_run_registry({"run_id": "r2"}, registry_path=reg)
    lines = reg.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
