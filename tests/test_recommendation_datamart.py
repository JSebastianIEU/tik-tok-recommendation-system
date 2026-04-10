import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.recommendation import (
    BuildTrainingDataMartConfig,
    CanonicalDatasetBundle,
    TrainingDataMart,
    build_comment_intelligence_snapshot_manifest,
    build_comment_transfer_priors,
    build_contract_manifest,
    build_training_data_mart,
    build_training_data_mart_from_manifest,
    build_training_data_mart_from_jsonl,
    validate_feature_access_policy,
    validate_raw_dataset_jsonl_against_contract,
)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _snapshot_id(video_id: str, idx: int) -> str:
    return f"{video_id}-s{idx}"


def make_bundle() -> CanonicalDatasetBundle:
    return CanonicalDatasetBundle(
        **{
            "version": "contract.v2",
            "generated_at": "2026-01-20T00:00:00Z",
            "authors": [
                {"author_id": "a1", "followers_count": 1000},
                {"author_id": "a2", "followers_count": 900},
            ],
            "videos": [
                {
                    "video_id": "v1",
                    "author_id": "a1",
                    "caption": "Quick meal prep guide",
                    "hashtags": ["#mealprep", "#tutorial"],
                    "keywords": ["meal prep", "guide"],
                    "search_query": "meal prep",
                    "posted_at": "2026-01-01T00:00:00Z",
                    "duration_seconds": 40,
                    "language": "en",
                },
                {
                    "video_id": "v2",
                    "author_id": "a1",
                    "caption": "How to batch cook",
                    "hashtags": ["#mealprep", "#food"],
                    "keywords": ["batch", "cook"],
                    "search_query": "meal prep",
                    "posted_at": "2026-01-02T00:00:00Z",
                    "duration_seconds": 50,
                    "language": "en",
                },
                {
                    "video_id": "v3",
                    "author_id": "a2",
                    "caption": "Startup founder story",
                    "hashtags": ["#startup", "#story"],
                    "keywords": ["founder", "story"],
                    "search_query": "startup",
                    "posted_at": "2026-01-03T00:00:00Z",
                },
                {
                    "video_id": "v4",
                    "author_id": "a2",
                    "caption": "Recent upload should be censored",
                    "hashtags": ["#recent"],
                    "keywords": ["recent"],
                    "search_query": "startup",
                    "posted_at": "2026-01-18T00:00:00Z",
                    "duration_seconds": 35,
                },
            ],
            "video_snapshots": [
                {
                    "video_snapshot_id": "v1-s1",
                    "video_id": "v1",
                    "scraped_at": "2026-01-02T00:00:00Z",
                    "views": 1000,
                    "likes": 80,
                    "comments_count": 10,
                    "shares": 6,
                },
                {
                    "video_snapshot_id": "v1-s2",
                    "video_id": "v1",
                    "scraped_at": "2026-01-05T00:00:00Z",
                    "views": 5000,
                    "likes": 480,
                    "comments_count": 80,
                    "shares": 42,
                },
                {
                    "video_snapshot_id": "v2-s1",
                    "video_id": "v2",
                    "scraped_at": "2026-01-03T00:00:00Z",
                    "views": 1200,
                    "likes": 100,
                    "comments_count": 12,
                    "shares": 8,
                },
                {
                    "video_snapshot_id": "v2-s2",
                    "video_id": "v2",
                    "scraped_at": "2026-01-06T00:00:00Z",
                    "views": 6200,
                    "likes": 620,
                    "comments_count": 110,
                    "shares": 55,
                },
                {
                    "video_snapshot_id": "v3-s1",
                    "video_id": "v3",
                    "scraped_at": "2026-01-04T00:00:00Z",
                    "views": 900,
                    "likes": 30,
                    "comments_count": 9,
                    "shares": 2,
                },
                {
                    "video_snapshot_id": "v3-s2",
                    "video_id": "v3",
                    "scraped_at": "2026-01-07T00:00:00Z",
                    "views": 2400,
                    "likes": 90,
                    "comments_count": 33,
                    "shares": 14,
                },
            ],
            "comments": [
                {
                    "comment_id": "v1::c1",
                    "video_id": "v1",
                    "text": "Can you share exact steps?",
                    "created_at": "2026-01-01T12:00:00Z",
                },
                {
                    "comment_id": "v1::c2",
                    "video_id": "v1",
                    "text": "Saved this",
                    "created_at": "2026-01-03T12:00:00Z",
                },
                {
                    "comment_id": "v2::c1",
                    "video_id": "v2",
                    "text": "How much rice?",
                    "created_at": "2026-01-02T12:00:00Z",
                },
            ],
            "comment_snapshots": [
                {
                    "comment_snapshot_id": "v1::c1::s1",
                    "comment_id": "v1::c1",
                    "video_id": "v1",
                    "scraped_at": "2026-01-05T00:00:00Z",
                    "likes": 10,
                    "reply_count": 1,
                }
            ],
        }
    )


def make_split_bundle() -> CanonicalDatasetBundle:
    generated_at = _dt("2026-01-30T00:00:00Z")
    videos = []
    snapshots = []
    for idx in range(1, 6):
        video_id = f"v{idx}"
        posted_at = _dt(f"2026-01-0{idx}T00:00:00Z")
        videos.append(
            {
                "video_id": video_id,
                "author_id": "author-main",
                "caption": f"Video {idx}",
                "hashtags": ["#topic"],
                "keywords": ["topic"],
                "search_query": "topic",
                "posted_at": posted_at.isoformat().replace("+00:00", "Z"),
                "duration_seconds": 30 + idx,
                "language": "en",
            }
        )
        snapshots.append(
            {
                "video_snapshot_id": _snapshot_id(video_id, 1),
                "video_id": video_id,
                "scraped_at": (posted_at + timedelta(hours=24)).isoformat().replace("+00:00", "Z"),
                "views": 100 * idx,
                "likes": 5 * idx,
                "comments_count": 2 * idx,
                "shares": idx,
            }
        )
        snapshots.append(
            {
                "video_snapshot_id": _snapshot_id(video_id, 2),
                "video_id": video_id,
                "scraped_at": (posted_at + timedelta(hours=96)).isoformat().replace("+00:00", "Z"),
                "views": 1000 * idx,
                "likes": 90 * idx,
                "comments_count": 12 * idx,
                "shares": 9 * idx,
            }
        )

    return CanonicalDatasetBundle(
        version="contract.v2",
        generated_at=generated_at,
        authors=[{"author_id": "author-main", "followers_count": 2000}],
        videos=videos,
        video_snapshots=snapshots,
        comments=[],
        comment_snapshots=[],
    )


def make_partial_trajectory_bundle() -> CanonicalDatasetBundle:
    return CanonicalDatasetBundle(
        version="contract.v2",
        generated_at=_dt("2026-01-20T00:00:00Z"),
        authors=[{"author_id": "a1", "followers_count": 1000}],
        videos=[
            {
                "video_id": "v1",
                "author_id": "a1",
                "caption": "Trajectory partial",
                "hashtags": ["#traj"],
                "keywords": ["traj"],
                "search_query": "traj",
                "posted_at": "2026-01-01T00:00:00Z",
                "duration_seconds": 30,
                "language": "en",
            }
        ],
        video_snapshots=[
            {
                "video_snapshot_id": "v1-s1",
                "video_id": "v1",
                "scraped_at": "2026-01-02T04:00:00Z",  # after t0 (as_of), before t6
                "event_time": "2026-01-02T04:00:00Z",
                "ingested_at": "2026-01-02T04:05:00Z",
                "source_watermark_time": "2026-01-02T04:00:00Z",
                "views": 120,
                "likes": 12,
                "comments_count": 4,
                "shares": 2,
            },
            {
                "video_snapshot_id": "v1-s2",
                "video_id": "v1",
                "scraped_at": "2026-01-06T00:00:00Z",
                "event_time": "2026-01-06T00:00:00Z",
                "ingested_at": "2026-01-06T00:05:00Z",
                "source_watermark_time": "2026-01-06T00:00:00Z",
                "views": 500,
                "likes": 50,
                "comments_count": 20,
                "shares": 10,
            },
        ],
        comments=[],
        comment_snapshots=[],
    )


def make_ingest_cutoff_trajectory_bundle() -> CanonicalDatasetBundle:
    return CanonicalDatasetBundle(
        version="contract.v2",
        generated_at=_dt("2026-01-10T00:00:00Z"),
        authors=[{"author_id": "a1", "followers_count": 1000}],
        videos=[
            {
                "video_id": "vcut",
                "author_id": "a1",
                "caption": "Ingest cutoff trajectory",
                "hashtags": ["#traj"],
                "keywords": ["traj"],
                "search_query": "traj",
                "posted_at": "2026-01-01T00:00:00Z",
                "duration_seconds": 30,
                "language": "en",
            }
        ],
        video_snapshots=[
            {
                "video_snapshot_id": "vcut-s1",
                "video_id": "vcut",
                "scraped_at": "2026-01-02T00:00:00Z",
                "event_time": "2026-01-02T00:00:00Z",
                "ingested_at": "2026-01-08T00:00:00Z",
                "source_watermark_time": "2026-01-02T00:00:00Z",
                "views": 100,
                "likes": 10,
                "comments_count": 2,
                "shares": 1,
            },
            {
                "video_snapshot_id": "vcut-s2",
                "video_id": "vcut",
                "scraped_at": "2026-01-05T00:00:00Z",
                "event_time": "2026-01-05T00:00:00Z",
                "ingested_at": "2026-01-05T01:00:00Z",
                "source_watermark_time": "2026-01-05T00:00:00Z",
                "views": 800,
                "likes": 90,
                "comments_count": 25,
                "shares": 9,
            },
        ],
        comments=[],
        comment_snapshots=[],
    )


def make_late_out_anchor_bundle() -> CanonicalDatasetBundle:
    return CanonicalDatasetBundle(
        version="contract.v2",
        generated_at=_dt("2026-01-10T00:00:00Z"),
        authors=[{"author_id": "a1", "followers_count": 1000}],
        videos=[
            {
                "video_id": "vlate",
                "author_id": "a1",
                "caption": "Late out anchor",
                "hashtags": ["#traj"],
                "keywords": ["traj"],
                "search_query": "traj",
                "posted_at": "2026-01-01T00:00:00Z",
                "duration_seconds": 30,
                "language": "en",
            }
        ],
        video_snapshots=[
            {
                "video_snapshot_id": "vlate-s1",
                "video_id": "vlate",
                "scraped_at": "2026-01-02T00:00:00Z",
                "event_time": "2026-01-02T00:00:00Z",
                "ingested_at": "2026-01-02T00:10:00Z",
                "source_watermark_time": "2026-01-02T00:00:00Z",
                "lateness_class": "late_out_of_watermark",
                "views": 120,
                "likes": 12,
                "comments_count": 4,
                "shares": 1,
            },
            {
                "video_snapshot_id": "vlate-s2",
                "video_id": "vlate",
                "scraped_at": "2026-01-05T00:00:00Z",
                "event_time": "2026-01-05T00:00:00Z",
                "ingested_at": "2026-01-05T00:10:00Z",
                "source_watermark_time": "2026-01-05T00:00:00Z",
                "lateness_class": "on_time",
                "views": 900,
                "likes": 100,
                "comments_count": 30,
                "shares": 12,
            },
        ],
        comments=[],
        comment_snapshots=[],
    )


def _mutate_snapshot_views(
    bundle: CanonicalDatasetBundle, video_snapshot_id: str, views: int
) -> CanonicalDatasetBundle:
    payload = bundle.model_dump(mode="python")
    for item in payload["video_snapshots"]:
        if item["video_snapshot_id"] == video_snapshot_id:
            item["views"] = views
            break
    return CanonicalDatasetBundle.model_validate(payload)


def _as_of_row_map(mart: dict) -> dict:
    return {row["video_id"]: row for row in mart["rows"]}


def test_validate_raw_dataset_jsonl_against_contract_supports_time_series_duplicates():
    raw = "\n".join(
        [
            '{"video_id":"x1","caption":"First caption","hashtags":["#a"],"keywords":["a"],"posted_at":"2026-01-01T00:00:00Z","likes":10,"comments_count":2,"shares":1,"views":100,"scraped_at":"2026-01-01T10:00:00Z","author":{"author_id":"ax"}}',
            '{"video_id":"x1","caption":"Second caption","hashtags":["#b"],"keywords":["b"],"posted_at":"2026-01-01T00:00:00Z","likes":11,"comments_count":3,"shares":1,"views":110,"scraped_at":"2026-01-02T10:00:00Z","author":{"author_id":"ax"}}',
        ]
    )
    result = validate_raw_dataset_jsonl_against_contract(
        raw_jsonl=raw,
        as_of_time=_dt("2026-01-20T00:00:00Z"),
    )
    assert result.ok is True
    assert result.bundle is not None
    assert len(result.bundle.videos) == 1
    assert len(result.bundle.video_snapshots) == 2
    assert result.bundle.videos[0].caption == "Second caption"
    assert not any("dropped by contract normalizer" in warning for warning in result.warnings)


def test_validate_raw_dataset_jsonl_against_contract_rejects_conflicting_video_author():
    raw = "\n".join(
        [
            '{"video_id":"x1","caption":"A","hashtags":["#a"],"keywords":["a"],"posted_at":"2026-01-01T00:00:00Z","likes":10,"comments_count":2,"shares":1,"views":100,"scraped_at":"2026-01-01T10:00:00Z","author":{"author_id":"ax"}}',
            '{"video_id":"x1","caption":"B","hashtags":["#b"],"keywords":["b"],"posted_at":"2026-01-01T00:00:00Z","likes":11,"comments_count":3,"shares":1,"views":110,"scraped_at":"2026-01-02T10:00:00Z","author":{"author_id":"ay"}}',
        ]
    )
    result = validate_raw_dataset_jsonl_against_contract(
        raw_jsonl=raw,
        as_of_time=_dt("2026-01-20T00:00:00Z"),
    )
    assert result.ok is False
    assert any("conflicting author_id for video 'x1'" in err for err in result.errors)


def test_timestamp_precedence_and_fallback_behavior():
    as_of = _dt("2026-01-20T00:00:00Z")
    raw = "\n".join(
        [
            '{"video_id":"p1","caption":"A","hashtags":["#a"],"keywords":["a"],"posted_at":"2026-01-01T00:00:00Z","likes":10,"comments_count":1,"shares":1,"views":100,"scraped_at":"2026-01-03T01:00:00Z","metrics_scraped_at":"2026-01-04T01:00:00Z","author":{"author_id":"a"}}',
            '{"video_id":"p2","caption":"B","hashtags":["#b"],"keywords":["b"],"posted_at":"2026-01-01T00:00:00Z","likes":11,"comments_count":1,"shares":1,"views":110,"scraped_at":"bad-ts","metrics_scraped_at":"2026-01-05T01:00:00Z","author":{"author_id":"b"}}',
            '{"video_id":"p3","caption":"C","hashtags":["#c"],"keywords":["c"],"posted_at":"2026-01-01T00:00:00Z","likes":12,"comments_count":1,"shares":1,"views":120,"author":{"author_id":"c"}}',
        ]
    )
    result = validate_raw_dataset_jsonl_against_contract(raw_jsonl=raw, as_of_time=as_of)
    assert result.ok is True
    assert result.bundle is not None
    times = {
        snap.video_id: snap.scraped_at.astimezone(timezone.utc)
        for snap in result.bundle.video_snapshots
    }
    assert times["p1"] == _dt("2026-01-03T01:00:00Z")
    assert times["p2"] == _dt("2026-01-05T01:00:00Z")
    assert times["p3"] == as_of
    assert any("fallback to as_of_time" in warning for warning in result.warnings)


def test_strict_timestamps_rejects_fallback():
    raw = (
        '{"video_id":"p3","caption":"C","hashtags":["#c"],"keywords":["c"],"posted_at":"2026-01-01T00:00:00Z","likes":12,"comments_count":1,"shares":1,"views":120,"author":{"author_id":"c"}}'
    )
    result = validate_raw_dataset_jsonl_against_contract(
        raw_jsonl=raw,
        as_of_time=_dt("2026-01-20T00:00:00Z"),
        strict_timestamps=True,
    )
    assert result.ok is False
    assert any("missing valid snapshot timestamp" in err for err in result.errors)


def test_build_training_data_mart_applies_censoring_and_reason_stats():
    bundle = make_bundle()
    mart = build_training_data_mart(
        bundle,
        config=BuildTrainingDataMartConfig(
            track="pre_publication",
            min_history_hours=24,
            label_window_hours=72,
            train_ratio=0.67,
            validation_ratio=0.16,
        ),
    )
    assert mart["version"] == "datamart.v1"
    assert mart["stats"]["rows_total"] == 3
    assert mart["stats"]["rows_censored"] == len(mart["excluded_video_records"])
    assert "v4" in mart["excluded_video_ids"]
    assert len(mart["excluded_video_records"]) > 0
    assert len(mart["stats"]["excluded_by_reason"]) > 0


def test_build_training_data_mart_post_publication_uses_only_pre_as_of_comments():
    bundle = make_bundle()
    mart = build_training_data_mart(
        bundle,
        config=BuildTrainingDataMartConfig(
            track="post_publication",
            min_history_hours=24,
            label_window_hours=72,
        ),
    )
    row_v1 = next(row for row in mart["rows"] if row["video_id"] == "v1")
    assert row_v1["features"]["comment_features"]["available"] is True
    # For v1, as_of_time is 2026-01-02T00:00:00Z so only c1 is eligible.
    assert row_v1["features"]["comment_features"]["count_pre"] == 1


def test_datamart_post_publication_uses_comment_snapshot_manifest(tmp_path: Path):
    bundle = make_bundle()
    snapshot_manifest = build_comment_intelligence_snapshot_manifest(
        bundle=bundle,
        as_of_time=_dt("2026-01-02T00:00:00Z"),
        output_root=tmp_path / "comment_features",
        mode="full",
    )
    manifest_path = (
        tmp_path
        / "comment_features"
        / snapshot_manifest["comment_feature_manifest_id"]
        / "manifest.json"
    )
    mart = build_training_data_mart(
        bundle,
        config=BuildTrainingDataMartConfig(
            track="post_publication",
            min_history_hours=24,
            label_window_hours=72,
            comment_feature_manifest_path=str(manifest_path),
        ),
    )
    row_v1 = next(row for row in mart["rows"] if row["video_id"] == "v1")
    ci = row_v1["features"]["comment_intelligence"]
    assert ci["source"] == "video_snapshot"
    assert ci["available"] is True
    assert ci["confusion_index"] >= 0.0
    assert 0.0 <= ci["alignment_score"] <= 1.0
    assert 0.0 <= ci["value_prop_coverage"] <= 1.0
    assert 0.0 <= ci["on_topic_ratio"] <= 1.0
    assert 0.0 <= ci["artifact_drift_ratio"] <= 1.0
    assert isinstance(ci.get("z_features"), dict)
    assert "alignment_score" in ci["z_features"]


def test_datamart_pre_publication_uses_transfer_priors_only(tmp_path: Path):
    bundle = make_bundle()
    snapshot_manifest = build_comment_intelligence_snapshot_manifest(
        bundle=bundle,
        as_of_time=_dt("2026-01-20T00:00:00Z"),
        output_root=tmp_path / "comment_features",
        mode="full",
    )
    snapshot_manifest_path = (
        tmp_path
        / "comment_features"
        / snapshot_manifest["comment_feature_manifest_id"]
        / "manifest.json"
    )
    priors_manifest = build_comment_transfer_priors(
        snapshot_manifest=snapshot_manifest_path,
        output_root=tmp_path / "comment_priors",
        min_support=1,
        shrinkage_alpha=1.0,
    )
    priors_manifest_path = (
        tmp_path / "comment_priors" / priors_manifest["comment_priors_manifest_id"] / "manifest.json"
    )
    mart = build_training_data_mart(
        bundle,
        config=BuildTrainingDataMartConfig(
            track="pre_publication",
            min_history_hours=24,
            label_window_hours=72,
            comment_feature_manifest_path=str(snapshot_manifest_path),
            comment_priors_manifest_path=str(priors_manifest_path),
        ),
    )
    row_v1 = next(row for row in mart["rows"] if row["video_id"] == "v1")
    ci = row_v1["features"]["comment_intelligence"]
    assert ci["source"] == "transfer_prior"
    assert ci["available"] is True
    assert 0.0 <= ci["alignment_score"] <= 1.0
    assert 0.0 <= ci["value_prop_coverage"] <= 1.0
    assert 0.0 <= ci["artifact_drift_ratio"] <= 1.0
    assert isinstance(ci.get("z_features"), dict)


def test_build_training_data_mart_uses_train_only_stats_for_zscores():
    bundle = make_split_bundle()
    mart = build_training_data_mart(
        bundle,
        config=BuildTrainingDataMartConfig(
            min_history_hours=24,
            label_window_hours=72,
            train_ratio=0.6,
            validation_ratio=0.2,
        ),
    )
    train_rows = [row for row in mart["rows"] if row["split"] == "train"]
    assert len(train_rows) == 3

    train_reach = [row["labels"]["future_reach_log_delta"] for row in train_rows]
    train_engagement = [row["labels"]["future_engagement_rate"] for row in train_rows]
    train_conversion = [row["labels"]["future_shares_per_1k_views"] for row in train_rows]

    reach_mean = sum(train_reach) / len(train_reach)
    engagement_mean = sum(train_engagement) / len(train_engagement)
    conversion_mean = sum(train_conversion) / len(train_conversion)
    reach_std = math.sqrt(
        sum((v - reach_mean) ** 2 for v in train_reach) / max(1, len(train_reach) - 1)
    )
    engagement_std = math.sqrt(
        sum((v - engagement_mean) ** 2 for v in train_engagement)
        / max(1, len(train_engagement) - 1)
    )
    conversion_std = math.sqrt(
        sum((v - conversion_mean) ** 2 for v in train_conversion)
        / max(1, len(train_conversion) - 1)
    )

    def expected(value: float, mean: float, std: float) -> float:
        return 0.0 if std == 0 else (value - mean) / std

    for row in mart["rows"]:
        assert row["targets_z"]["reach"] == pytest.approx(
            round(expected(row["labels"]["future_reach_log_delta"], reach_mean, reach_std), 6)
        )
        assert row["targets_z"]["engagement"] == pytest.approx(
            round(
                expected(
                    row["labels"]["future_engagement_rate"],
                    engagement_mean,
                    engagement_std,
                ),
                6,
            )
        )
        assert row["targets_z"]["conversion"] == pytest.approx(
            round(
                expected(
                    row["labels"]["future_shares_per_1k_views"],
                    conversion_mean,
                    conversion_std,
                ),
                6,
            )
        )


def test_build_training_data_mart_adds_trajectory_fields_and_formulas():
    bundle = make_split_bundle()
    mart = build_training_data_mart(
        bundle,
        config=BuildTrainingDataMartConfig(
            min_history_hours=24,
            label_window_hours=72,
            train_ratio=0.6,
            validation_ratio=0.2,
        ),
    )
    row = next(item for item in mart["rows"] if item["video_id"] == "v1")
    reach_labels = row["labels_trajectory"]["reach"]
    assert set(reach_labels["series"].keys()) == {"t0", "t6", "t24", "t96"}
    assert set(reach_labels["components"].keys()) == {
        "early_velocity",
        "core_velocity",
        "late_lift",
        "stability",
    }
    assert row["target_availability"]["reach"]["components"]["early_velocity"] is True
    assert row["target_availability"]["reach"]["components"]["late_lift"] is True
    expected_late_lift = (
        reach_labels["series"]["t96"] - reach_labels["series"]["t24"]
    )
    assert reach_labels["components"]["late_lift"] == pytest.approx(round(expected_late_lift, 6))
    assert "trajectory_unavailable_by_reason" in mart["stats"]
    assert "trajectory_unavailable_by_component" in mart["stats"]


def test_build_training_data_mart_trajectory_partial_mask_keeps_row():
    bundle = make_partial_trajectory_bundle()
    mart = build_training_data_mart(
        bundle,
        config=BuildTrainingDataMartConfig(
            min_history_hours=24,
            label_window_hours=72,
            train_ratio=0.8,
            validation_ratio=0.1,
        ),
    )
    assert mart["stats"]["rows_total"] == 1
    row = mart["rows"][0]
    availability = row["target_availability"]["reach"]
    assert availability["components"]["early_velocity"] is False
    assert availability["objective_available"] is False
    reasons = row["target_unavailable_reasons"]["reach"]
    assert "missing_anchor_snapshot" in reasons
    component_reasons = row["target_unavailable_component_reasons"]["reach"]
    assert "missing_anchor_snapshot" in component_reasons["early_velocity"]
    assert component_reasons["late_lift"] == []


def test_build_training_data_mart_respects_explicit_as_of_run_time():
    bundle = make_ingest_cutoff_trajectory_bundle()
    run_cutoff = _dt("2026-01-06T00:00:00Z")
    mart = build_training_data_mart(
        bundle,
        config=BuildTrainingDataMartConfig(
            min_history_hours=24,
            label_window_hours=72,
            as_of_run_time=run_cutoff,
        ),
    )
    assert mart["stats"]["rows_total"] == 1
    assert mart["config"]["as_of_run_time"] == run_cutoff
    row = mart["rows"][0]
    assert "ingested_after_cutoff" in row["target_unavailable_reasons"]["reach"]
    assert "ingested_after_cutoff" in row["target_unavailable_component_reasons"]["reach"][
        "early_velocity"
    ]
    assert mart["stats"]["trajectory_unavailable_by_reason"]["ingested_after_cutoff"] > 0


def test_build_training_data_mart_marks_late_out_of_watermark_in_trajectory():
    bundle = make_late_out_anchor_bundle()
    mart = build_training_data_mart(
        bundle,
        config=BuildTrainingDataMartConfig(
            min_history_hours=24,
            label_window_hours=72,
        ),
    )
    row = mart["rows"][0]
    assert "late_out_of_watermark" in row["target_unavailable_reasons"]["reach"]
    assert "late_out_of_watermark" in row["target_unavailable_component_reasons"]["reach"][
        "early_velocity"
    ]
    assert mart["stats"]["trajectory_unavailable_by_reason"]["late_out_of_watermark"] > 0


def test_trajectory_train_rows_not_affected_by_test_distribution_shift():
    base_bundle = make_split_bundle()
    shifted_test_bundle = _mutate_snapshot_views(base_bundle, "v5-s2", 500_000)

    cfg = BuildTrainingDataMartConfig(
        min_history_hours=24,
        label_window_hours=72,
        train_ratio=0.6,
        validation_ratio=0.2,
        pair_target_source="trajectory_v2_composite",
    )
    mart_base = build_training_data_mart(base_bundle, config=cfg)
    mart_shift = build_training_data_mart(shifted_test_bundle, config=cfg)
    rows_base = _as_of_row_map(mart_base)
    rows_shift = _as_of_row_map(mart_shift)

    for video_id in ("v1", "v2", "v3", "v4"):
        assert (
            rows_base[video_id]["targets_trajectory_z"]
            == rows_shift[video_id]["targets_trajectory_z"]
        )
    assert rows_base["v5"]["targets_trajectory_z"] != rows_shift["v5"]["targets_trajectory_z"]


def test_train_rows_not_affected_by_test_distribution_shift():
    base_bundle = make_split_bundle()
    shifted_test_bundle = _mutate_snapshot_views(base_bundle, "v5-s2", 250000)

    cfg = BuildTrainingDataMartConfig(
        min_history_hours=24,
        label_window_hours=72,
        train_ratio=0.6,
        validation_ratio=0.2,
    )
    mart_base = build_training_data_mart(base_bundle, config=cfg)
    mart_shift = build_training_data_mart(shifted_test_bundle, config=cfg)

    rows_base = _as_of_row_map(mart_base)
    rows_shift = _as_of_row_map(mart_shift)

    for video_id in ("v1", "v2", "v3", "v4"):
        assert rows_base[video_id]["targets_z"] == rows_shift[video_id]["targets_z"]


def test_validation_and_test_change_when_train_distribution_changes():
    base_bundle = make_split_bundle()
    shifted_train_bundle = _mutate_snapshot_views(base_bundle, "v1-s2", 10000)

    cfg = BuildTrainingDataMartConfig(
        min_history_hours=24,
        label_window_hours=72,
        train_ratio=0.6,
        validation_ratio=0.2,
    )
    mart_base = build_training_data_mart(base_bundle, config=cfg)
    mart_shift = build_training_data_mart(shifted_train_bundle, config=cfg)

    rows_base = _as_of_row_map(mart_base)
    rows_shift = _as_of_row_map(mart_shift)
    assert rows_base["v4"]["targets_z"] != rows_shift["v4"]["targets_z"]
    assert rows_base["v5"]["targets_z"] != rows_shift["v5"]["targets_z"]


def test_author_baseline_is_leave_one_out_for_train_and_train_only_for_non_train():
    bundle = make_split_bundle()
    cfg = BuildTrainingDataMartConfig(
        min_history_hours=24,
        label_window_hours=72,
        train_ratio=0.6,
        validation_ratio=0.2,
        min_author_rows_for_baseline=2,
    )
    mart = build_training_data_mart(bundle, config=cfg)
    train_rows = [row for row in mart["rows"] if row["split"] == "train"]
    non_train_rows = [row for row in mart["rows"] if row["split"] != "train"]

    train_log_views = [math.log1p(row["labels"]["future_views"]) for row in train_rows]
    train_mean = sum(train_log_views) / len(train_log_views)

    for row in train_rows:
        current = math.log1p(row["labels"]["future_views"])
        expected = (sum(train_log_views) - current) / (len(train_log_views) - 1)
        assert row["labels"]["author_expected_log_views"] == pytest.approx(round(expected, 6))

    for row in non_train_rows:
        assert row["labels"]["author_expected_log_views"] == pytest.approx(round(train_mean, 6))


def test_build_training_data_mart_builds_pair_rows_with_time_order():
    bundle = make_bundle()
    mart = build_training_data_mart(
        bundle,
        config=BuildTrainingDataMartConfig(
            include_pair_rows=True,
            pair_objective="engagement",
            pair_candidates_per_query=3,
        ),
    )
    assert len(mart["pair_rows"]) > 0
    for pair in mart["pair_rows"]:
        assert pair["candidate_as_of_time"] < pair["query_as_of_time"]
        assert 0 <= pair["relevance_label"] <= 3


def test_build_training_data_mart_pair_target_source_trajectory_skips_unavailable():
    bundle = make_split_bundle()
    payload = bundle.model_dump(mode="python")
    for item in payload["video_snapshots"]:
        if item["video_snapshot_id"] == "v2-s1":
            item["scraped_at"] = "2026-01-03T06:00:00Z"  # as_of(v2)=2026-01-03T00:00:00Z
            item["event_time"] = "2026-01-03T06:00:00Z"
            item["ingested_at"] = "2026-01-03T06:10:00Z"
            item["source_watermark_time"] = "2026-01-03T06:00:00Z"
            break
    adjusted = CanonicalDatasetBundle.model_validate(payload)
    mart = build_training_data_mart(
        adjusted,
        config=BuildTrainingDataMartConfig(
            include_pair_rows=True,
            pair_objective="engagement",
            pair_target_source="trajectory_v2_composite",
            pair_candidates_per_query=5,
        ),
    )
    assert len(mart["pair_rows"]) > 0
    assert all(pair["target_source"] == "trajectory_v2_composite" for pair in mart["pair_rows"])
    assert all(pair["query_video_id"] != "v2" for pair in mart["pair_rows"])
    assert all(pair["candidate_video_id"] != "v2" for pair in mart["pair_rows"])
    assert all(pair["availability_mask"]["query_objective_available"] is True for pair in mart["pair_rows"])
    assert mart["stats"]["pair_rows_dropped_unavailable_target"] > 0
    assert mart["stats"]["pair_rows_dropped_by_reason"]["query_target_unavailable"] > 0
    assert mart["stats"]["pair_rows_dropped_by_reason"]["candidate_target_unavailable"] > 0


def test_build_training_data_mart_output_validates_against_schema():
    bundle = make_bundle()
    mart = build_training_data_mart(bundle)
    validated = TrainingDataMart.model_validate(mart)
    assert validated.version == "datamart.v1"
    assert validated.source_contract_version == "contract.v2"


def test_build_training_data_mart_config_validation():
    with pytest.raises(ValueError):
        BuildTrainingDataMartConfig(track="bad")
    with pytest.raises(ValueError):
        BuildTrainingDataMartConfig(pair_objective="bad")
    with pytest.raises(ValueError):
        BuildTrainingDataMartConfig(min_history_hours=0)
    with pytest.raises(ValueError):
        BuildTrainingDataMartConfig(label_window_hours=0)
    with pytest.raises(ValueError):
        BuildTrainingDataMartConfig(pair_candidates_per_query=0)
    with pytest.raises(ValueError):
        BuildTrainingDataMartConfig(pair_target_source="bad")
    with pytest.raises(ValueError):
        BuildTrainingDataMartConfig(trajectory_windows_hours=(24, 6, 96))
    with pytest.raises(ValueError):
        BuildTrainingDataMartConfig(
            pair_target_source="trajectory_v2_composite",
            enable_trajectory_labels=False,
        )


def test_build_training_data_mart_from_jsonl_integration_and_warning_gating():
    raw = "\n".join(
        [
            '{"video_id":"a","caption":"A","hashtags":["#a"],"keywords":["a"],"posted_at":"2026-01-01T00:00:00Z","likes":50,"comments_count":5,"shares":2,"views":1000,"author":{"author_id":"aa"}}',
            '{"video_id":"b","caption":"B","hashtags":["#b"],"keywords":["b"],"posted_at":"2026-01-02T00:00:00Z","likes":70,"comments_count":6,"shares":3,"views":1200,"author":{"author_id":"bb"}}',
        ]
    )
    mart = build_training_data_mart_from_jsonl(
        raw_jsonl=raw,
        as_of_time=_dt("2026-01-20T00:00:00Z"),
        config=BuildTrainingDataMartConfig(
            min_history_hours=24,
            label_window_hours=72,
        ),
    )
    assert mart["version"] == "datamart.v1"
    assert mart["stats"]["rows_total"] >= 1
    assert len(mart["warnings"]) > 0

    with pytest.raises(ValueError):
        build_training_data_mart_from_jsonl(
            raw_jsonl=raw,
            as_of_time=_dt("2026-01-20T00:00:00Z"),
            config=BuildTrainingDataMartConfig(
                min_history_hours=24,
                label_window_hours=72,
            ),
            fail_on_warnings=True,
        )


def test_feature_access_policy_blocks_comment_namespace_for_pre_publication():
    errors = validate_feature_access_policy(
        "pre_publication",
        ["video_metadata", "comments"],
    )
    assert len(errors) == 1


def test_build_training_data_mart_from_jsonl_writes_manifest(tmp_path):
    raw = (
        '{"video_id":"a","caption":"A","hashtags":["#a"],"keywords":["a"],'
        '"posted_at":"2026-01-01T00:00:00Z","likes":50,"comments_count":5,"shares":2,'
        '"views":1000,"author":{"author_id":"aa"}}'
    )
    mart = build_training_data_mart_from_jsonl(
        raw_jsonl=raw,
        as_of_time=_dt("2026-01-20T00:00:00Z"),
        config=BuildTrainingDataMartConfig(
            min_history_hours=24,
            label_window_hours=72,
        ),
        manifest_root=tmp_path,
    )
    assert mart["source_manifest_id"] is not None
    assert mart["source_manifest_path"] is not None


def test_build_training_data_mart_from_manifest(tmp_path):
    bundle = make_bundle()
    manifest = build_contract_manifest(
        bundle=bundle,
        manifest_root=tmp_path,
        source_file_hashes={"unit": "hash"},
        as_of_time=_dt("2026-01-20T00:00:00Z"),
    )
    mart = build_training_data_mart_from_manifest(
        manifest_ref=manifest["manifest_dir"],
        config=BuildTrainingDataMartConfig(
            min_history_hours=24,
            label_window_hours=72,
        ),
    )
    assert mart["source_manifest_id"] == manifest["manifest_id"]


def test_build_training_data_mart_respects_bitemporal_ingest_cutoff():
    bundle = CanonicalDatasetBundle(
        version="contract.v2",
        generated_at=_dt("2026-01-30T00:00:00Z"),
        authors=[{"author_id": "a1", "followers_count": 1000}],
        videos=[
            {
                "video_id": "v1",
                "author_id": "a1",
                "caption": "Temporal test",
                "hashtags": ["#x"],
                "keywords": ["x"],
                "search_query": "x",
                "posted_at": "2026-01-01T00:00:00Z",
                "duration_seconds": 30,
                "language": "en",
            }
        ],
        video_snapshots=[
            {
                "video_snapshot_id": "v1-s1",
                "video_id": "v1",
                "scraped_at": "2026-01-02T00:00:00Z",
                "views": 100,
                "likes": 10,
                "comments_count": 2,
                "shares": 1,
                "ingested_at": "2026-01-04T00:00:00Z",
                "event_time": "2026-01-02T00:00:00Z",
                "source_watermark_time": "2026-01-02T00:00:00Z",
            },
            {
                "video_snapshot_id": "v1-s2",
                "video_id": "v1",
                "scraped_at": "2026-01-06T00:00:00Z",
                "views": 600,
                "likes": 60,
                "comments_count": 12,
                "shares": 5,
                "ingested_at": "2026-01-07T00:00:00Z",
                "event_time": "2026-01-06T00:00:00Z",
                "source_watermark_time": "2026-01-06T00:00:00Z",
            },
        ],
        comments=[],
        comment_snapshots=[],
    )
    mart = build_training_data_mart(
        bundle,
        config=BuildTrainingDataMartConfig(
            min_history_hours=24,
            label_window_hours=72,
        ),
    )
    assert mart["stats"]["rows_total"] == 1
    row = mart["rows"][0]
    assert row["features"]["pre_metrics"]["has_snapshot"] is False
    assert "missing_pre_snapshot" in row["features"]["missingness_flags"]
