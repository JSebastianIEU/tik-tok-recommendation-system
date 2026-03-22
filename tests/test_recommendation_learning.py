from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.recommendation import (
    BuildTrainingDataMartConfig,
    CanonicalDatasetBundle,
    build_training_data_mart,
)
from src.recommendation.learning import (
    ArtifactRegistry,
    NegativeSampler,
    NegativeSamplerConfig,
    RecommenderRuntime,
    RecommenderTrainingConfig,
    TemporalCandidatePool,
    TemporalCandidatePoolConfig,
    map_objective,
    train_recommender_from_datamart,
)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def make_datamart() -> dict:
    generated_at = _dt("2026-03-01T00:00:00Z")
    authors = [{"author_id": "a1", "followers_count": 1500}]
    videos = []
    snapshots = []
    comments = []
    comment_snapshots = []

    base_time = _dt("2026-01-01T00:00:00Z")
    for idx in range(1, 11):
        video_id = f"v{idx}"
        posted_at = base_time + timedelta(days=idx)
        videos.append(
            {
                "video_id": video_id,
                "author_id": "a1",
                "caption": f"video {idx} tutorial growth",
                "hashtags": ["#growth", f"#topic{idx%3}"],
                "keywords": ["growth", "tutorial"],
                "search_query": "growth",
                "posted_at": posted_at,
                "duration_seconds": 30 + idx,
                "language": "en",
            }
        )
        snapshots.append(
            {
                "video_snapshot_id": f"{video_id}-s1",
                "video_id": video_id,
                "scraped_at": posted_at + timedelta(hours=24),
                "views": 100 * idx,
                "likes": 10 * idx,
                "comments_count": 2 * idx,
                "shares": idx,
            }
        )
        snapshots.append(
            {
                "video_snapshot_id": f"{video_id}-s2",
                "video_id": video_id,
                "scraped_at": posted_at + timedelta(hours=96),
                "views": 900 * idx,
                "likes": 80 * idx,
                "comments_count": 15 * idx,
                "shares": 6 * idx,
            }
        )

    bundle = CanonicalDatasetBundle(
        version="contract.v1",
        generated_at=generated_at,
        authors=authors,
        videos=videos,
        video_snapshots=snapshots,
        comments=comments,
        comment_snapshots=comment_snapshots,
    )
    return build_training_data_mart(
        bundle,
        config=BuildTrainingDataMartConfig(
            track="pre_publication",
            min_history_hours=24,
            label_window_hours=72,
            train_ratio=0.6,
            validation_ratio=0.2,
            include_pair_rows=True,
            pair_objective="engagement",
            pair_candidates_per_query=6,
        ),
    )


def test_map_objective_supports_community_alias():
    requested, effective = map_objective("community")
    assert requested == "community"
    assert effective == "engagement"


def test_temporal_candidate_pool_enforces_past_only():
    rows = make_datamart()["rows"]
    split = {row["split"]: row for row in rows}
    query = split["test"]
    pool = TemporalCandidatePool(
        TemporalCandidatePoolConfig(max_age_days=365, min_pool_size=1)
    ).for_query(query_row=query, candidate_rows=rows, index_cutoff_time=query["as_of_time"])
    assert len(pool) > 0
    assert all(item["as_of_time"] < query["as_of_time"] for item in pool)


def test_temporal_candidate_pool_accepts_string_index_cutoff():
    rows = make_datamart()["rows"]
    split = {row["split"]: row for row in rows}
    query = split["test"]
    pool = TemporalCandidatePool(
        TemporalCandidatePoolConfig(max_age_days=365, min_pool_size=1)
    ).for_query(
        query_row=query,
        candidate_rows=rows,
        index_cutoff_time=str(query["as_of_time"]),
    )
    assert len(pool) > 0
    assert all(item["as_of_time"] < query["as_of_time"] for item in pool)


def test_negative_sampler_is_deterministic_and_non_empty():
    rows = make_datamart()["rows"]
    query = next(row for row in rows if row["split"] == "validation")
    pool = [row for row in rows if row["as_of_time"] < query["as_of_time"]]
    positives = pool[:2]

    sampler_a = NegativeSampler(NegativeSamplerConfig(seed=7))
    sampler_b = NegativeSampler(NegativeSamplerConfig(seed=7))
    sample_a = sampler_a.sample(query, positives, pool)
    sample_b = sampler_b.sample(query, positives, pool)
    assert len(sample_a) > 0
    assert [row["row_id"] for row in sample_a] == [row["row_id"] for row in sample_b]


def test_train_recommender_from_datamart_creates_artifacts(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(
            objectives=("reach", "engagement", "conversion"),
            retrieve_k=50,
            run_name="test-recommender",
            dense_model_name="sentence-transformers/all-MiniLM-L6-v2",
        ),
    )
    bundle_dir = Path(result["bundle_dir"])
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "retriever" / "manifest.json").exists()
    assert (bundle_dir / "rankers" / "reach" / "metadata.json").exists()
    assert (bundle_dir / "rankers" / "engagement" / "metadata.json").exists()
    assert (bundle_dir / "rankers" / "conversion" / "metadata.json").exists()


def test_artifact_registry_compatibility_check(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="compat"),
    )
    bundle_dir = Path(result["bundle_dir"])
    registry = ArtifactRegistry(tmp_path)
    manifest = registry.load_manifest(bundle_dir)
    registry.assert_compatible(
        bundle_dir,
        {"feature_schema_hash": manifest["feature_schema_hash"]},
    )
    with pytest.raises(ValueError):
        registry.assert_compatible(
            bundle_dir,
            {"feature_schema_hash": "bad-hash"},
        )


def test_recommender_runtime_community_objective_returns_effective_model(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="runtime"),
    )
    runtime = RecommenderRuntime(bundle_dir=Path(result["bundle_dir"]))
    rows = mart["rows"]
    query = next(row for row in rows if row["split"] == "test")
    candidates = [
        {
            "candidate_id": row["row_id"],
            "text": f"{row['topic_key']} {row['row_id']}",
            "topic_key": row["topic_key"],
            "author_id": row["author_id"],
            "as_of_time": row["as_of_time"],
        }
        for row in rows
        if row["as_of_time"] < query["as_of_time"]
    ]
    response = runtime.recommend(
        objective="community",
        as_of_time=query["as_of_time"],
        query={
            "query_id": query["row_id"],
            "text": f"{query['topic_key']} {query['row_id']}",
            "topic_key": query["topic_key"],
            "author_id": query["author_id"],
        },
        candidates=candidates,
        top_k=5,
        retrieve_k=20,
        debug=True,
    )
    assert response["objective"] == "community"
    assert response["objective_effective"] == "engagement"
    assert isinstance(response["items"], list)
    assert len(response["items"]) > 0


def test_recommender_runtime_rejects_untrained_objective(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(
            run_name="runtime-reach-only",
            objectives=("reach",),
        ),
    )
    runtime = RecommenderRuntime(bundle_dir=Path(result["bundle_dir"]))
    rows = mart["rows"]
    query = next(row for row in rows if row["split"] == "test")
    candidates = [
        {
            "candidate_id": row["row_id"],
            "text": f"{row['topic_key']} {row['row_id']}",
            "topic_key": row["topic_key"],
            "author_id": row["author_id"],
            "as_of_time": row["as_of_time"],
        }
        for row in rows
        if row["as_of_time"] < query["as_of_time"]
    ]
    with pytest.raises(ValueError):
        runtime.recommend(
            objective="engagement",
            as_of_time=query["as_of_time"],
            query={
                "query_id": query["row_id"],
                "text": f"{query['topic_key']} {query['row_id']}",
                "topic_key": query["topic_key"],
                "author_id": query["author_id"],
            },
            candidates=candidates,
            top_k=5,
            retrieve_k=20,
        )
