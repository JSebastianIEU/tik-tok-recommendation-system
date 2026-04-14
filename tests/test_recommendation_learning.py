from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.recommendation import BuildTrainingDataMartConfig, CanonicalDatasetBundle, build_training_data_mart
from src.recommendation.learning import (
    ArtifactRegistry,
    NegativeSampler,
    NegativeSamplerConfig,
    RecommenderRuntime,
    RecommenderRuntimeConfig,
    RecommenderTrainingConfig,
    TemporalCandidatePool,
    TemporalCandidatePoolConfig,
    _branch_dropout_weights,
    _select_retriever_weight_variant,
    map_objective,
    train_recommender_from_datamart,
)
from src.recommendation.learning.inference import ArtifactCompatibilityError


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def make_datamart() -> dict:
    generated_at = _dt("2026-03-01T00:00:00Z")
    authors = [{"author_id": "a1", "followers_count": 1500}]
    videos = []
    snapshots = []

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
        version="contract.v2",
        generated_at=generated_at,
        authors=authors,
        videos=videos,
        video_snapshots=snapshots,
        comments=[],
        comment_snapshots=[],
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


def test_branch_dropout_keeps_dropped_branch_zero_when_single_branch_base():
    weights = _branch_dropout_weights(
        {
            "lexical": 0.0,
            "dense_text": 0.0,
            "multimodal": 0.0,
            "graph_dense": 1.0,
            "trajectory_dense": 0.0,
        },
        dropped_branches=["graph_dense"],
    )
    assert float(weights["graph_dense"]) == 0.0
    assert abs(sum(float(v) for v in weights.values()) - 1.0) < 1e-6
    assert float(weights["lexical"]) > 0.0
    assert float(weights["dense_text"]) > 0.0
    assert float(weights["multimodal"]) > 0.0
    assert float(weights["trajectory_dense"]) > 0.0


def test_retriever_weight_gate_falls_back_to_sparse_when_validation_worse():
    gate = _select_retriever_weight_variant(
        learned_weights={
            "lexical": 0.1,
            "dense_text": 0.6,
            "multimodal": 0.2,
            "graph_dense": 0.1,
            "trajectory_dense": 0.0,
        },
        learned_validation={"recall@50": 0.20, "recall@100": 0.25, "recall@200": 0.30},
        sparse_baseline_weights={
            "lexical": 1.0,
            "dense_text": 0.0,
            "multimodal": 0.0,
            "graph_dense": 0.0,
            "trajectory_dense": 0.0,
        },
        sparse_validation={"recall@50": 0.25, "recall@100": 0.30, "recall@200": 0.35},
        learned_test={"recall@50": 0.30, "recall@100": 0.35, "recall@200": 0.40},
        sparse_test={"recall@50": 0.28, "recall@100": 0.31, "recall@200": 0.38},
    )
    assert gate["selected_variant"] == "sparse_baseline"
    selected = gate["selected_weights"]
    assert float(selected["lexical"]) == 1.0
    assert float(selected["dense_text"]) == 0.0
    assert gate["validation"]["selected_not_worse_than_competitor"] is True
    assert gate["test"]["selected_not_worse_than_competitor"] is False


def test_train_recommender_from_datamart_creates_baseline_artifacts(tmp_path: Path):
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
    assert (bundle_dir / "rankers" / "reach" / "baseline_manifest.json").exists()
    assert (bundle_dir / "rankers" / "engagement" / "baseline_manifest.json").exists()
    assert (bundle_dir / "rankers" / "conversion" / "baseline_manifest.json").exists()
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    objective_metrics = json.loads(
        (bundle_dir / "metrics" / "objective_metrics.json").read_text(encoding="utf-8")
    )
    assert manifest["component"] == "recommender-learning-v1"
    assert manifest["ranker_family_version"] == "recommender.ranker.baseline.v1"
    assert "fabric_registry_signature" in manifest
    assert "fabric_schema_hashes" in manifest
    assert "policy_reranker" in manifest
    assert "objective_ablation_reports" in manifest
    for objective in ("reach", "engagement", "conversion"):
        assert objective in objective_metrics
        assert "retriever" in objective_metrics[objective]
        assert "ranker" in objective_metrics[objective]
        assert "ndcg@10" in objective_metrics[objective]["ranker"]
        assert "mrr@20" in objective_metrics[objective]["ranker"]
        assert "recall@100" in objective_metrics[objective]["retriever"]
        ablation_meta = manifest["objective_ablation_reports"][objective]
        ablation_path = bundle_dir / str(ablation_meta["path"])
        assert ablation_path.exists()


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
            "video_id": row["video_id"],
            "text": f"{row['topic_key']} {row['row_id']}",
            "topic_key": row["topic_key"],
            "author_id": row["author_id"],
            "as_of_time": row["as_of_time"],
            "signal_hints": {
                "comment_intelligence": {
                    "source": "unit_test",
                    "available": True,
                    "taxonomy_version": "comment_taxonomy.v2.0.0",
                    "dominant_intents": ["help_seeking"],
                    "confusion_index": 0.3,
                    "help_seeking_index": 0.6,
                    "sentiment_volatility": 0.2,
                    "sentiment_shift_early_late": 0.0,
                    "reply_depth_max": 1.0,
                    "reply_branch_factor": 0.5,
                    "reply_ratio": 0.2,
                    "root_thread_concentration": 1.0,
                    "confidence": 0.8,
                    "missingness_flags": [],
                }
            },
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
        routing={"stage_budgets_ms": {"retrieval": 5000, "ranking": 5000}},
        debug=True,
    )
    assert response["objective"] == "community"
    assert response["objective_effective"] == "engagement"
    assert isinstance(response["items"], list)
    assert len(response["items"]) > 0
    first_item = response["items"][0]
    assert isinstance(first_item.get("comment_trace"), dict)
    assert "alignment_score" in first_item["comment_trace"]
    assert "retrieval_branch_scores" in first_item
    assert "score_components" in first_item
    assert response["calibration_version"] == "calibration.baseline.v1"
    assert response["policy_version"] == "policy.baseline.v1"


def test_recommender_runtime_can_retrieve_from_bundle_index_without_candidates(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="runtime-bundle-retrieval"),
    )
    runtime = RecommenderRuntime(bundle_dir=Path(result["bundle_dir"]))
    query = next(row for row in mart["rows"] if row["split"] == "test")
    response = runtime.recommend(
        objective="engagement",
        as_of_time=query["as_of_time"],
        query={
            "query_id": query["row_id"],
            "description": query["caption"],
            "hashtags": list(query["hashtags"]),
            "topic_key": query["topic_key"],
            "language": query["language"],
            "content_type": query["content_type"],
            "author_id": query["author_id"],
        },
        candidates=[],
        top_k=5,
        retrieve_k=20,
        routing={"stage_budgets_ms": {"retrieval": 5000, "ranking": 5000}},
        debug=True,
    )
    assert len(response["items"]) > 0
    assert response["retriever_artifact_version"] == "retriever.v2.0"
    debug_retrieval = response["debug"]["config"]["retrieval"]
    assert debug_retrieval["retriever_loaded"] is True
    assert isinstance(debug_retrieval["stats"], dict)
    first_item = response["items"][0]
    assert first_item["retrieval_branch_scores"]["dense_text"] >= 0.0


def test_runtime_rejects_incompatible_feature_hash(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="runtime-compat"),
    )
    bundle_dir = Path(result["bundle_dir"])
    with pytest.raises(ArtifactCompatibilityError):
        RecommenderRuntime(
            bundle_dir=bundle_dir,
            config=RecommenderRuntimeConfig(feature_schema_hash="bad-hash"),
        )
