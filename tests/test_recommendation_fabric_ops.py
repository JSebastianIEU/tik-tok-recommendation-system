from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.recommendation import (
    CanonicalDatasetBundle,
    PromotionThresholds,
    build_feature_snapshot_manifest,
    evaluate_shadow_promotion,
)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _bundle() -> CanonicalDatasetBundle:
    return CanonicalDatasetBundle(
        version="contract.v2",
        generated_at=_dt("2026-03-24T12:00:00Z"),
        authors=[{"author_id": "a1", "followers_count": 1000}],
        videos=[
            {
                "video_id": "v1",
                "author_id": "a1",
                "caption": "Hook then payoff",
                "hashtags": ["#growth"],
                "keywords": ["growth"],
                "posted_at": _dt("2026-03-20T12:00:00Z"),
                "duration_seconds": 30,
                "language": "en",
            }
        ],
        video_snapshots=[
            {
                "video_snapshot_id": "v1-s1",
                "video_id": "v1",
                "scraped_at": _dt("2026-03-21T12:00:00Z"),
                "views": 1000,
                "likes": 90,
                "comments_count": 14,
                "shares": 9,
            }
        ],
        comments=[],
        comment_snapshots=[],
    )


def test_build_feature_snapshot_manifest(tmp_path: Path) -> None:
    bundle = _bundle()
    payload = build_feature_snapshot_manifest(
        bundle=bundle,
        output_root=tmp_path,
        mode="full",
        as_of_time=_dt("2026-03-24T12:00:00Z"),
    )
    assert payload["feature_manifest_id"]
    assert payload["fabric_version"] == "fabric.v2"
    assert payload["stats"]["rows_total"] == 1
    manifest_path = Path(tmp_path) / payload["feature_manifest_id"] / "manifest.json"
    assert manifest_path.exists()


def test_evaluate_shadow_promotion_flags_regressions() -> None:
    ok, failures = evaluate_shadow_promotion(
        baseline={
            "mandatory_coverage_rate": 0.99,
            "offline_quality_metric": 0.81,
        },
        shadow={
            "mandatory_coverage_rate": 0.95,
            "latency_p95_ms": 220.0,
            "psi_mandatory": 0.31,
            "psi_optional": 0.35,
            "offline_quality_metric": 0.7,
        },
        thresholds=PromotionThresholds(),
    )
    assert ok is False
    assert "mandatory_coverage" in failures
    assert "latency_p95_ms" in failures
    assert "psi_mandatory" in failures
    assert "psi_optional" in failures
    assert "offline_quality_metric" in failures
