from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.recommendation import (
    CanonicalDatasetBundle,
    build_comment_intelligence_snapshot_manifest,
    build_comment_transfer_priors,
    evaluate_comment_shadow_promotion,
    load_comment_intelligence_snapshot_manifest,
    load_comment_transfer_priors_manifest,
)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _bundle() -> CanonicalDatasetBundle:
    return CanonicalDatasetBundle(
        version="contract.v2",
        generated_at=_dt("2026-03-24T12:00:00Z"),
        authors=[{"author_id": "a1", "followers_count": 32000}],
        videos=[
            {
                "video_id": "v1",
                "author_id": "a1",
                "caption": "How to batch cook faster",
                "hashtags": ["#tutorial", "#food"],
                "keywords": ["batch", "cook"],
                "search_query": "meal prep",
                "posted_at": _dt("2026-03-20T00:00:00Z"),
            }
        ],
        video_snapshots=[
            {
                "video_snapshot_id": "v1-s1",
                "video_id": "v1",
                "scraped_at": _dt("2026-03-21T00:00:00Z"),
                "views": 1000,
                "likes": 120,
                "comments_count": 12,
                "shares": 10,
            }
        ],
        comments=[
            {
                "comment_id": "v1::c1",
                "video_id": "v1",
                "text": "How much rice exactly?",
                "created_at": _dt("2026-03-20T01:00:00Z"),
                "ingested_at": _dt("2026-03-20T01:30:00Z"),
                "root_comment_id": "v1::c1",
                "comment_level": 0,
            },
            {
                "comment_id": "v1::c2",
                "video_id": "v1",
                "text": "Saved this, super helpful",
                "created_at": _dt("2026-03-21T08:00:00Z"),
                "ingested_at": _dt("2026-03-21T08:30:00Z"),
                "root_comment_id": "v1::c1",
                "parent_comment_id": "v1::c1",
                "comment_level": 1,
            },
            {
                "comment_id": "v1::future",
                "video_id": "v1",
                "text": "future event must be excluded",
                "created_at": _dt("2026-03-28T08:00:00Z"),
                "ingested_at": _dt("2026-03-28T08:30:00Z"),
                "root_comment_id": "v1::future",
                "comment_level": 0,
            },
        ],
        comment_snapshots=[
            {
                "comment_snapshot_id": "v1::c1::s1",
                "comment_id": "v1::c1",
                "video_id": "v1",
                "scraped_at": _dt("2026-03-21T01:00:00Z"),
                "reply_count": 1,
                "likes": 10,
            }
        ],
    )


def test_comment_snapshot_manifest_enforces_as_of_filtering(tmp_path: Path) -> None:
    payload = build_comment_intelligence_snapshot_manifest(
        bundle=_bundle(),
        as_of_time=_dt("2026-03-24T12:00:00Z"),
        output_root=tmp_path,
        mode="full",
    )
    assert payload["comment_feature_manifest_id"]
    assert payload["stats"]["rows_total"] == 1
    _, rows = load_comment_intelligence_snapshot_manifest(
        Path(tmp_path) / payload["comment_feature_manifest_id"] / "manifest.json"
    )
    row = rows[0]
    assert row["features"]["comment_count_total"] == 2
    assert row["features"]["confusion_index"] > 0
    assert row["features"]["help_seeking_index"] >= 0
    assert 0.0 <= row["features"]["alignment_score"] <= 1.0
    assert 0.0 <= row["features"]["value_prop_coverage"] <= 1.0
    assert 0.0 <= row["features"]["on_topic_ratio"] <= 1.0
    assert 0.0 <= row["features"]["artifact_drift_ratio"] <= 1.0
    assert -1.0 <= row["features"]["alignment_shift_early_late"] <= 1.0
    assert 0.0 <= row["features"]["alignment_confidence"] <= 1.0
    assert str(row["features"]["alignment_method_version"]).startswith("alignment.v2.hybrid")


def test_comment_transfer_priors_manifest_build_and_load(tmp_path: Path) -> None:
    snapshot_manifest = build_comment_intelligence_snapshot_manifest(
        bundle=_bundle(),
        as_of_time=_dt("2026-03-24T12:00:00Z"),
        output_root=tmp_path / "features",
        mode="full",
    )
    priors_manifest = build_comment_transfer_priors(
        snapshot_manifest=Path(tmp_path)
        / "features"
        / snapshot_manifest["comment_feature_manifest_id"]
        / "manifest.json",
        output_root=tmp_path / "priors",
        min_support=1,
        shrinkage_alpha=1.0,
    )
    manifest_payload, priors = load_comment_transfer_priors_manifest(
        Path(tmp_path)
        / "priors"
        / priors_manifest["comment_priors_manifest_id"]
        / "manifest.json"
    )
    assert manifest_payload["rows_total"] >= 1
    assert len(priors.entries) >= 1
    entry = priors.entries[0]
    assert 0.0 <= entry.confusion_index <= 1.0
    assert 0.0 <= entry.help_seeking_index <= 1.0
    assert 0.0 <= entry.prior_alignment_score <= 1.0
    assert 0.0 <= entry.prior_value_prop_coverage <= 1.0
    assert 0.0 <= entry.prior_artifact_drift_ratio <= 1.0
    assert 0.0 <= entry.prior_alignment_confidence <= 1.0


def test_comment_rollout_gate_flags_regression() -> None:
    ok, failures = evaluate_comment_shadow_promotion(
        baseline={
            "mandatory_comment_coverage_rate": 0.98,
            "offline_quality_metric": 0.8,
        },
        shadow={
            "mandatory_comment_coverage_rate": 0.95,
            "latency_p95_ms": 180.0,
            "offline_quality_metric": 0.7,
        },
    )
    assert ok is False
    assert "mandatory_comment_coverage" in failures
    assert "latency_p95_ms" in failures
    assert "offline_quality_metric" in failures


def test_comment_alignment_scores_are_deterministic(tmp_path: Path) -> None:
    bundle = _bundle()
    manifest_a = build_comment_intelligence_snapshot_manifest(
        bundle=bundle,
        as_of_time=_dt("2026-03-24T12:00:00Z"),
        output_root=tmp_path / "a",
        mode="full",
    )
    manifest_b = build_comment_intelligence_snapshot_manifest(
        bundle=bundle,
        as_of_time=_dt("2026-03-24T12:00:00Z"),
        output_root=tmp_path / "b",
        mode="full",
    )
    _, rows_a = load_comment_intelligence_snapshot_manifest(
        Path(tmp_path) / "a" / manifest_a["comment_feature_manifest_id"] / "manifest.json"
    )
    _, rows_b = load_comment_intelligence_snapshot_manifest(
        Path(tmp_path) / "b" / manifest_b["comment_feature_manifest_id"] / "manifest.json"
    )
    assert rows_a[0]["features"]["alignment_score"] == rows_b[0]["features"]["alignment_score"]
    assert (
        rows_a[0]["features"]["alignment_shift_early_late"]
        == rows_b[0]["features"]["alignment_shift_early_late"]
    )
