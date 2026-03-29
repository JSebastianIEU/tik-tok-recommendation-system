from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.recommendation import (
    BuildTrainingDataMartConfig,
    CanonicalDatasetBundle,
    build_comment_intelligence_snapshot_manifest,
    build_training_data_mart,
)
from src.recommendation.learning import (
    AdaptiveNegativeMiner,
    AdaptiveNegativeMiningConfig,
    ArtifactRegistry,
    GraphBuildConfig,
    TrajectoryBuildConfig,
    NegativeSampler,
    NegativeSamplerConfig,
    ObjectiveRankerTrainer,
    RecommenderRuntime,
    RecommenderTrainingConfig,
    TemporalCandidatePool,
    TemporalCandidatePoolConfig,
    build_creator_video_dna_graph,
    build_trajectory_bundle,
    map_objective,
    train_recommender_from_datamart,
)
from src.recommendation.learning.inference import ArtifactCompatibilityError, RoutingContractError
from src.recommendation.learning.ranker import (
    SEGMENT_CREATOR_COLD_START,
    SEGMENT_CREATOR_MATURE,
    SEGMENT_FORMAT_ENTERTAINMENT,
    SEGMENT_FORMAT_TUTORIAL,
    segment_candidates_for_pair,
)
from src.recommendation.learning.retriever import HybridRetriever


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
        version="contract.v2",
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


def make_datamart_with_comment_manifest(tmp_path: Path) -> dict:
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
            comment_feature_manifest_path=str(snapshot_manifest_path),
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


def test_segment_candidates_respect_creator_and_format_mapping():
    cold_query = {
        "author_id": "a1",
        "features": {"creator_prior_video_count": 3},
    }
    tutorial_candidate = {
        "author_id": "a2",
        "content_type": "tutorial",
    }
    cold_segments = segment_candidates_for_pair(
        query_row=cold_query,
        candidate_row=tutorial_candidate,
        creator_cold_threshold=10,
    )
    assert cold_segments == [SEGMENT_CREATOR_COLD_START, SEGMENT_FORMAT_TUTORIAL]

    mature_query = {
        "author_id": "a1",
        "features": {"creator_prior_video_count": 12},
    }
    entertainment_candidate = {
        "author_id": "a3",
        "content_type": "reaction",
    }
    mature_segments = segment_candidates_for_pair(
        query_row=mature_query,
        candidate_row=entertainment_candidate,
        creator_cold_threshold=10,
    )
    assert mature_segments == [SEGMENT_CREATOR_MATURE, SEGMENT_FORMAT_ENTERTAINMENT]


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
    assert (bundle_dir / "rankers" / "reach" / "family_manifest.json").exists()
    assert (bundle_dir / "rankers" / "reach" / "global" / "ensemble_manifest.json").exists()
    assert (bundle_dir / "rankers" / "reach" / "calibration.json").exists()
    assert (bundle_dir / "rankers" / "engagement" / "family_manifest.json").exists()
    assert (bundle_dir / "rankers" / "engagement" / "global" / "ensemble_manifest.json").exists()
    assert (bundle_dir / "rankers" / "engagement" / "calibration.json").exists()
    assert (bundle_dir / "rankers" / "conversion" / "family_manifest.json").exists()
    assert (bundle_dir / "rankers" / "conversion" / "global" / "ensemble_manifest.json").exists()
    assert (bundle_dir / "rankers" / "conversion" / "calibration.json").exists()
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "fabric_registry_signature" in manifest
    assert "fabric_schema_hashes" in manifest
    assert "comment_intelligence_version" in manifest
    assert "ranker_family" in manifest
    assert manifest["ranker_family"]["ensemble_size"] == 5
    assert "ranker_calibration" in manifest
    assert "policy_reranker" in manifest
    assert "objective_calibration_reports" in manifest
    for objective in ("reach", "engagement", "conversion"):
        assert objective in manifest["ranker_family"]["segment_promotion_by_objective"]
        calibration_meta = manifest["objective_calibration_reports"][objective]
        calibration_path = bundle_dir / str(calibration_meta["path"])
        assert calibration_path.exists()
        calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        assert calibration_payload["target_definition"] == "p_relevance_ge_2"
        assert "quality" in calibration_payload["summary"]
        assert "overall" in calibration_payload["summary"]["quality"]
        assert "brier" in calibration_payload["summary"]["quality"]["overall"]
        assert "logloss" in calibration_payload["summary"]["quality"]["overall"]
        assert "ece" in calibration_payload["summary"]["quality"]["overall"]


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
        debug=True,
    )
    assert response["objective"] == "community"
    assert response["objective_effective"] == "engagement"
    assert isinstance(response["items"], list)
    assert len(response["items"]) > 0
    assert "comment_intelligence_version" in response
    first_item = response["items"][0]
    assert isinstance(first_item.get("comment_trace"), dict)
    assert "alignment_score" in first_item["comment_trace"]
    assert "value_prop_coverage" in first_item["comment_trace"]
    assert "artifact_drift_ratio" in first_item["comment_trace"]
    assert "graph_bundle_id" in response
    assert "graph_version" in response
    assert "graph_fallback_mode" in response
    assert "calibration_version" in response
    assert "policy_version" in response
    assert isinstance(response.get("policy_metadata"), dict)
    assert isinstance(response.get("calibration_metadata"), dict)
    assert "score_raw" in first_item
    assert "score_calibrated" in first_item
    assert "score_mean" in first_item
    assert "score_std" in first_item
    assert "confidence" in first_item
    assert "selected_ranker_id" in first_item
    assert "global_score_mean" in first_item
    assert "segment_blend_weight" in first_item
    assert "policy_penalty" in first_item
    assert "policy_bonus" in first_item
    assert "policy_adjusted_score" in first_item
    assert isinstance(first_item.get("calibration_trace"), dict)
    assert isinstance(first_item.get("policy_trace"), dict)
    assert isinstance(first_item.get("retrieval_branch_scores"), dict)
    assert "graph_dense" in first_item["retrieval_branch_scores"]


def test_recommender_runtime_portfolio_mode_adds_trace_fields(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="runtime-portfolio"),
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
        objective="engagement",
        as_of_time=query["as_of_time"],
        query={
            "query_id": query["row_id"],
            "text": f"{query['topic_key']} {query['row_id']}",
            "topic_key": query["topic_key"],
            "author_id": query["author_id"],
        },
        candidates=candidates,
        portfolio={
            "enabled": True,
            "weights": {"reach": 0.45, "conversion": 0.35, "durability": 0.20},
            "risk_aversion": 0.10,
            "candidate_pool_cap": 120,
        },
        top_k=5,
        retrieve_k=20,
    )
    assert response["portfolio_mode"] is True
    assert isinstance(response.get("portfolio_metadata"), dict)
    first_item = response["items"][0]
    assert isinstance(first_item.get("portfolio_trace"), dict)
    assert "utility_before_policy" in first_item["portfolio_trace"]
    assert "utility_after_policy" in first_item["portfolio_trace"]


def test_recommender_runtime_portfolio_mode_degrades_when_required_objectives_missing(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(
            run_name="runtime-portfolio-fallback",
            objectives=("engagement",),
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
    response = runtime.recommend(
        objective="engagement",
        as_of_time=query["as_of_time"],
        query={
            "query_id": query["row_id"],
            "text": f"{query['topic_key']} {query['row_id']}",
            "topic_key": query["topic_key"],
            "author_id": query["author_id"],
        },
        candidates=candidates,
        portfolio={"enabled": True},
        top_k=5,
        retrieve_k=20,
    )
    assert response["portfolio_mode"] is False
    metadata = response.get("portfolio_metadata")
    assert isinstance(metadata, dict)
    assert isinstance(metadata.get("fallback_reason"), str)
    assert str(metadata.get("fallback_reason", "")).startswith("missing_ranker_")


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


def test_recommender_runtime_allows_disabling_graph_branch(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="runtime-graph-disable"),
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
        objective="engagement",
        as_of_time=query["as_of_time"],
        query={
            "query_id": query["row_id"],
            "text": f"{query['topic_key']} {query['row_id']}",
            "topic_key": query["topic_key"],
            "author_id": query["author_id"],
        },
        candidates=candidates,
        graph_controls={"enable_graph_branch": False},
        top_k=5,
        retrieve_k=20,
    )
    assert response["graph_fallback_mode"] == "graph_branch_disabled"
    assert len(response["items"]) > 0


def test_recommender_runtime_uses_manifest_comment_features_with_hint_fallback(tmp_path: Path):
    mart = make_datamart_with_comment_manifest(tmp_path)
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="runtime-comment-manifest"),
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
            "as_of_time": "2026-03-01T00:00:00Z",
            "signal_hints": {
                "comment_intelligence": {
                    "source": "request_hint",
                    "available": True,
                    "taxonomy_version": "comment_taxonomy.v2.0.0",
                    "dominant_intents": ["confusion"],
                    "confusion_index": 0.99,
                    "help_seeking_index": 0.01,
                    "sentiment_volatility": 0.99,
                    "sentiment_shift_early_late": 0.0,
                    "reply_depth_max": 9.0,
                    "reply_branch_factor": 9.0,
                    "reply_ratio": 0.9,
                    "root_thread_concentration": 0.9,
                    "confidence": 0.1,
                    "missingness_flags": [],
                }
            },
        }
        for row in rows
        if row["row_id"] != query["row_id"]
    ]
    response = runtime.recommend(
        objective="engagement",
        as_of_time="2026-03-02T00:00:00Z",
        query={
            "query_id": query["row_id"],
            "text": f"{query['topic_key']} {query['row_id']}",
            "topic_key": query["topic_key"],
            "author_id": query["author_id"],
            "as_of_time": "2026-03-02T00:00:00Z",
        },
        candidates=candidates,
        top_k=5,
        retrieve_k=20,
        debug=True,
    )
    assert response["debug"]["comment_index_loaded"] is True
    assert len(response["items"]) > 0
    first_trace = response["items"][0].get("comment_trace")
    assert isinstance(first_trace, dict)
    assert first_trace.get("source") == "manifest_snapshot"
    assert "alignment_score" in first_trace
    assert "alignment_confidence" in first_trace


def test_recommender_runtime_rejects_fabric_schema_mismatch(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="fabric-mismatch"),
    )
    bundle_dir = Path(result["bundle_dir"])
    manifest_path = bundle_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["fabric_registry_signature"] = "bad-signature"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with pytest.raises(ValueError):
        RecommenderRuntime(bundle_dir=bundle_dir)


def test_recommender_training_config_rejects_bad_pair_target_source():
    with pytest.raises(ValueError):
        RecommenderTrainingConfig(pair_target_source="bad")


def test_train_recommender_from_datamart_supports_trajectory_pair_target_source(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(
            run_name="trajectory-target-source",
            pair_target_source="trajectory_v2_composite",
            dense_model_name="sentence-transformers/all-MiniLM-L6-v2",
        ),
    )
    bundle_dir = Path(result["bundle_dir"])
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pair_target_source"] == "trajectory_v2_composite"
    assert (bundle_dir / "rankers" / "engagement" / "family_manifest.json").exists()


def _adaptive_mining_fixture():
    mart = make_datamart()
    rows = list(mart["rows"])
    pair_rows = list(mart["pair_rows"])
    rows_by_id = {str(row["row_id"]): row for row in rows}
    train_rows = [row for row in rows if str(row.get("split")) == "train"]
    retriever = HybridRetriever.train(rows=train_rows)
    objective = "engagement"
    objective_pair_rows = [
        pair
        for pair in pair_rows
        if str(pair.get("objective")) == objective
        and str(pair.get("target_source", "scalar_v1")) == "scalar_v1"
    ]
    baseline_ranker = ObjectiveRankerTrainer(
        objective=objective,
        random_seed=13,
    ).train(rows_by_id=rows_by_id, pair_rows=objective_pair_rows)
    return {
        "mart": mart,
        "rows": rows,
        "rows_by_id": rows_by_id,
        "train_rows": train_rows,
        "pair_rows": objective_pair_rows,
        "retriever": retriever,
        "baseline_ranker": baseline_ranker,
        "objective": objective,
    }


def test_recommender_training_config_rejects_bad_negative_mining_mode():
    with pytest.raises(ValueError):
        RecommenderTrainingConfig(negative_mining_mode="bad")


def test_adaptive_negative_miner_is_deterministic():
    fixture = _adaptive_mining_fixture()
    config = AdaptiveNegativeMiningConfig(
        enabled=True,
        mining_candidate_k=60,
        seed=17,
    )
    miner_a = AdaptiveNegativeMiner(config=config)
    miner_b = AdaptiveNegativeMiner(config=config)
    mined_a, diagnostics_a = miner_a.mine(
        objective=fixture["objective"],
        target_source="scalar_v1",
        rows_by_id=fixture["rows_by_id"],
        train_rows=fixture["train_rows"],
        base_pair_rows=fixture["pair_rows"],
        retriever=fixture["retriever"],
        baseline_ranker=fixture["baseline_ranker"],
        max_age_days=180,
    )
    mined_b, diagnostics_b = miner_b.mine(
        objective=fixture["objective"],
        target_source="scalar_v1",
        rows_by_id=fixture["rows_by_id"],
        train_rows=fixture["train_rows"],
        base_pair_rows=fixture["pair_rows"],
        retriever=fixture["retriever"],
        baseline_ranker=fixture["baseline_ranker"],
        max_age_days=180,
    )
    assert [row["pair_id"] for row in mined_a] == [row["pair_id"] for row in mined_b]
    assert diagnostics_a["confusion_rate"] == diagnostics_b["confusion_rate"]
    assert diagnostics_a["ratios"] == diagnostics_b["ratios"]


def test_adaptive_negative_miner_false_friend_threshold_behavior():
    fixture = _adaptive_mining_fixture()
    miner = AdaptiveNegativeMiner(
        AdaptiveNegativeMiningConfig(
            enabled=True,
            mining_candidate_k=80,
            false_friend_similarity_pct=0.0,
            false_friend_prediction_pct=0.0,
            seed=5,
        )
    )
    mined_rows, diagnostics = miner.mine(
        objective=fixture["objective"],
        target_source="scalar_v1",
        rows_by_id=fixture["rows_by_id"],
        train_rows=fixture["train_rows"],
        base_pair_rows=fixture["pair_rows"],
        retriever=fixture["retriever"],
        baseline_ranker=fixture["baseline_ranker"],
        max_age_days=180,
    )
    assert len(mined_rows) > 0
    assert all(row.get("mining_reason") == "false_friend" for row in mined_rows)
    assert diagnostics["false_friend_count"] >= len(mined_rows)


def test_adaptive_negative_miner_enforces_debias_caps_and_temporal_safety():
    fixture = _adaptive_mining_fixture()
    miner = AdaptiveNegativeMiner(
        AdaptiveNegativeMiningConfig(
            enabled=True,
            mining_candidate_k=80,
            max_per_author=1,
            max_per_topic=1,
            max_per_era=1,
            seed=11,
        )
    )
    mined_rows, diagnostics = miner.mine(
        objective=fixture["objective"],
        target_source="scalar_v1",
        rows_by_id=fixture["rows_by_id"],
        train_rows=fixture["train_rows"],
        base_pair_rows=fixture["pair_rows"],
        retriever=fixture["retriever"],
        baseline_ranker=fixture["baseline_ranker"],
        max_age_days=180,
    )
    assert all(row.get("candidate_as_of_time") < row.get("query_as_of_time") for row in mined_rows)
    query_author_counts: dict[tuple[str, str], int] = {}
    for row in mined_rows:
        candidate_row = fixture["rows_by_id"][str(row["candidate_row_id"])]
        key = (str(row["query_row_id"]), str(candidate_row.get("author_id") or "unknown"))
        query_author_counts[key] = query_author_counts.get(key, 0) + 1
    assert all(count <= 1 for count in query_author_counts.values())
    assert diagnostics["dropped_by_cap"].get("author_cap", 0) >= 0


def test_train_recommender_with_adaptive_negative_mining_writes_manifest_and_diagnostics(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(
            objectives=("engagement",),
            run_name="adaptive-mining",
            negative_mining_mode="adaptive_v2",
            adaptive_negative_mining=AdaptiveNegativeMiningConfig(
                enabled=True,
                mining_candidate_k=80,
                false_friend_similarity_pct=0.0,
                false_friend_prediction_pct=0.0,
            ),
        ),
    )
    bundle_dir = Path(result["bundle_dir"])
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["negative_mining_mode"] == "adaptive_v2"
    assert isinstance(manifest.get("adaptive_selected_by_objective"), dict)
    diag_meta = manifest["objective_diagnostics"]["engagement"]
    diag_path = bundle_dir / str(diag_meta["path"])
    assert diag_path.exists()
    diag_payload = json.loads(diag_path.read_text(encoding="utf-8"))
    assert "gate" in diag_payload
    assert "mining" in diag_payload


def test_train_recommender_gate_prefers_baseline_on_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.recommendation.learning import pipeline as learning_pipeline

    def fake_mine(self, **kwargs):
        base_pair_rows = list(kwargs["base_pair_rows"])
        rows_by_id = kwargs["rows_by_id"]
        for pair in base_pair_rows:
            qid = str(pair["query_row_id"])
            cid = str(pair["candidate_row_id"])
            if str(rows_by_id[qid].get("split")) == "train" and str(rows_by_id[cid].get("split")) == "train":
                mined = dict(pair)
                mined["pair_id"] = f"{pair['pair_id']}::adaptive"
                mined["mined"] = True
                mined["mining_policy_version"] = "adaptive_v2"
                mined["mining_reason"] = "hard"
                mined["hardness_score"] = 0.8
                mined["debias_era_bucket"] = "2026-01"
                return [mined], {"enabled": True, "mode": "adaptive_v2", "mined_rows_total": 1}
        return [], {"enabled": True, "mode": "adaptive_v2", "mined_rows_total": 0}

    eval_calls = {"count": 0}

    def fake_rank_eval(*args, **kwargs):
        eval_calls["count"] += 1
        if eval_calls["count"] == 1:
            return {"ndcg@10": 0.5, "ndcg@20": 0.5, "mrr@10": 0.5, "mrr@20": 0.5}
        return {"ndcg@10": 0.49, "ndcg@20": 0.49, "mrr@10": 0.49, "mrr@20": 0.49}

    monkeypatch.setattr(learning_pipeline.AdaptiveNegativeMiner, "mine", fake_mine)
    monkeypatch.setattr(learning_pipeline, "_evaluate_ranker_objective", fake_rank_eval)

    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(
            objectives=("engagement",),
            run_name="adaptive-gate-baseline",
            negative_mining_mode="adaptive_v2",
            adaptive_negative_mining=AdaptiveNegativeMiningConfig(enabled=True),
        ),
    )
    metrics = result["objective_metrics"]["engagement"]
    assert metrics["ranker_selected_variant"] == "baseline"
    assert metrics["adaptive_gate"]["selected"] is False


def test_train_recommender_gate_promotes_adaptive_on_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.recommendation.learning import pipeline as learning_pipeline

    def fake_mine(self, **kwargs):
        base_pair_rows = list(kwargs["base_pair_rows"])
        rows_by_id = kwargs["rows_by_id"]
        for pair in base_pair_rows:
            qid = str(pair["query_row_id"])
            cid = str(pair["candidate_row_id"])
            if str(rows_by_id[qid].get("split")) == "train" and str(rows_by_id[cid].get("split")) == "train":
                mined = dict(pair)
                mined["pair_id"] = f"{pair['pair_id']}::adaptive"
                mined["mined"] = True
                mined["mining_policy_version"] = "adaptive_v2"
                mined["mining_reason"] = "hard"
                mined["hardness_score"] = 0.9
                mined["debias_era_bucket"] = "2026-01"
                return [mined], {"enabled": True, "mode": "adaptive_v2", "mined_rows_total": 1}
        return [], {"enabled": True, "mode": "adaptive_v2", "mined_rows_total": 0}

    eval_calls = {"count": 0}

    def fake_rank_eval(*args, **kwargs):
        eval_calls["count"] += 1
        if eval_calls["count"] == 1:
            return {"ndcg@10": 0.5, "ndcg@20": 0.5, "mrr@10": 0.5, "mrr@20": 0.5}
        return {"ndcg@10": 0.6, "ndcg@20": 0.6, "mrr@10": 0.6, "mrr@20": 0.6}

    monkeypatch.setattr(learning_pipeline.AdaptiveNegativeMiner, "mine", fake_mine)
    monkeypatch.setattr(learning_pipeline, "_evaluate_ranker_objective", fake_rank_eval)

    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(
            objectives=("engagement",),
            run_name="adaptive-gate-promote",
            negative_mining_mode="adaptive_v2",
            adaptive_negative_mining=AdaptiveNegativeMiningConfig(enabled=True),
        ),
    )
    metrics = result["objective_metrics"]["engagement"]
    assert metrics["ranker_selected_variant"] == "adaptive"
    assert metrics["adaptive_gate"]["selected"] is True


def test_retriever_candidate_id_intersection_and_metadata():
    rows = [
        {
            "row_id": "v1::2026-03-10T00:00:00Z",
            "video_id": "v1",
            "author_id": "a1",
            "topic_key": "alpha",
            "content_type": "tutorial",
            "locale": "en-us",
            "as_of_time": "2026-03-10T00:00:00Z",
            "features": {"language": "en", "missingness_flags": []},
        },
        {
            "row_id": "v2::2026-03-11T00:00:00Z",
            "video_id": "v2",
            "author_id": "a2",
            "topic_key": "beta",
            "content_type": "tutorial",
            "locale": "en-gb",
            "as_of_time": "2026-03-11T00:00:00Z",
            "features": {"language": "en", "missingness_flags": []},
        },
    ]
    retriever = HybridRetriever.train(rows=rows)
    query = {
        "row_id": "q1",
        "topic_key": "alpha",
        "as_of_time": "2026-03-20T00:00:00Z",
        "features": {"missingness_flags": []},
    }
    items, meta = retriever.retrieve(
        query_row=query,
        top_k=5,
        objective="engagement",
        candidate_ids=["v2"],
        retrieval_constraints={"language": "en", "locale": "en-us", "content_type": "tutorial"},
        return_metadata=True,
    )
    assert len(items) == 1
    assert items[0]["candidate_id"] == "v2"
    assert meta["retrieval_mode"] == "intersected"
    assert meta["constraint_tier_used"] in {0, 1, 2, 3}


def test_creator_video_dna_graph_is_deterministic():
    rows = [
        {
            "row_id": "v1::2026-03-10T00:00:00Z",
            "video_id": "v1",
            "author_id": "a1",
            "topic_key": "#growth tutorial",
            "content_type": "tutorial",
            "as_of_time": "2026-03-10T00:00:00Z",
            "_fabric_output": {
                "audio": {"speech_ratio": 0.8, "tempo": 120, "energy": 0.7, "music_presence": True},
                "visual": {"style_tags": ["close_up", "kitchen"]},
            },
            "features": {"keyword_count": 3, "missingness_flags": []},
        },
        {
            "row_id": "v2::2026-03-12T00:00:00Z",
            "video_id": "v2",
            "author_id": "a2",
            "topic_key": "#growth #mealprep",
            "content_type": "tutorial",
            "as_of_time": "2026-03-12T00:00:00Z",
            "_fabric_output": {
                "audio": {"speech_ratio": 0.6, "tempo": 100, "energy": 0.5, "music_presence": True},
                "visual": {"style_tags": ["close_up", "counter"]},
            },
            "features": {"keyword_count": 2, "missingness_flags": []},
        },
    ]
    cfg = GraphBuildConfig(embedding_dim=16, seed=9)
    graph_a = build_creator_video_dna_graph(
        rows=rows,
        as_of_time="2026-03-20T00:00:00Z",
        run_cutoff_time="2026-03-20T00:00:00Z",
        config=cfg,
    )
    graph_b = build_creator_video_dna_graph(
        rows=rows,
        as_of_time="2026-03-20T00:00:00Z",
        run_cutoff_time="2026-03-20T00:00:00Z",
        config=cfg,
    )
    assert graph_a.graph_bundle_id == graph_b.graph_bundle_id
    assert graph_a.graph_schema_hash == graph_b.graph_schema_hash
    assert len(graph_a.video_embeddings) >= 2


def test_retriever_graph_branch_exposes_graph_dense_scores():
    rows = [
        {
            "row_id": "v1::2026-03-10T00:00:00Z",
            "video_id": "v1",
            "author_id": "a1",
            "topic_key": "#growth tutorial",
            "content_type": "tutorial",
            "as_of_time": "2026-03-10T00:00:00Z",
            "features": {"language": "en", "missingness_flags": []},
        },
        {
            "row_id": "v2::2026-03-11T00:00:00Z",
            "video_id": "v2",
            "author_id": "a2",
            "topic_key": "#cooking tips",
            "content_type": "tutorial",
            "as_of_time": "2026-03-11T00:00:00Z",
            "features": {"language": "en", "missingness_flags": []},
        },
    ]
    retriever = HybridRetriever.train(
        rows=rows,
        graph_vectors={
            "v1::2026-03-10T00:00:00Z": [1.0, 0.0, 0.0, 0.0],
            "v2::2026-03-11T00:00:00Z": [0.0, 1.0, 0.0, 0.0],
        },
        graph_lookup={
            "video": {"q1": [1.0, 0.0, 0.0, 0.0]},
            "creator": {"a1": [1.0, 0.0, 0.0, 0.0]},
            "hashtag": {"#growth": [1.0, 0.0, 0.0, 0.0]},
            "audio_motif": {},
            "style_signature": {},
        },
        graph_metadata={
            "graph_bundle_id": "graph-test",
            "graph_version": "creator_video_dna_graph.v2",
            "graph_schema_hash": "x",
        },
    )
    items, metadata = retriever.retrieve(
        query_row={
            "row_id": "q1",
            "author_id": "a1",
            "topic_key": "#growth tutorial",
            "as_of_time": "2026-03-20T00:00:00Z",
            "features": {"missingness_flags": []},
        },
        top_k=2,
        objective="engagement",
        return_metadata=True,
    )
    assert len(items) == 2
    assert "graph_dense" in items[0]["retrieval_branch_scores"]
    assert metadata["graph_bundle_id"] == "graph-test"
    assert metadata["graph_version"] == "creator_video_dna_graph.v2"


def test_trajectory_bundle_is_deterministic_for_same_rows():
    rows = make_datamart()["rows"]
    cfg = TrajectoryBuildConfig(embedding_dim=12, feature_version="trajectory_features.v2")
    bundle_a = build_trajectory_bundle(
        rows=rows,
        as_of_time="2026-03-01T00:00:00Z",
        run_cutoff_time="2026-03-01T00:00:00Z",
        config=cfg,
    )
    bundle_b = build_trajectory_bundle(
        rows=rows,
        as_of_time="2026-03-01T00:00:00Z",
        run_cutoff_time="2026-03-01T00:00:00Z",
        config=cfg,
    )
    assert bundle_a.trajectory_manifest_id == bundle_b.trajectory_manifest_id
    assert bundle_a.trajectory_schema_hash == bundle_b.trajectory_schema_hash
    assert bundle_a.embeddings_by_video == bundle_b.embeddings_by_video


def test_retriever_trajectory_branch_exposes_scores_and_metadata():
    rows = [
        {
            "row_id": "v1::2026-03-10T00:00:00Z",
            "video_id": "v1",
            "author_id": "a1",
            "topic_key": "#growth tutorial",
            "content_type": "tutorial",
            "as_of_time": "2026-03-10T00:00:00Z",
            "features": {"language": "en", "missingness_flags": []},
            "labels_trajectory": {
                "reach": {"series": {"t0": 1.0, "t6": 2.0, "t24": 3.5, "t96": 4.0}, "components": {"early_velocity": 0.2, "core_velocity": 0.1, "late_lift": 0.5, "stability": -0.2}},
                "engagement": {"series": {"t0": 0.1, "t6": 0.2, "t24": 0.3, "t96": 0.4}, "components": {"early_velocity": 0.1, "core_velocity": 0.05, "late_lift": 0.1, "stability": -0.1}},
                "conversion": {"series": {"t0": 0.1, "t6": 0.2, "t24": 0.3, "t96": 0.4}, "components": {"early_velocity": 0.1, "core_velocity": 0.05, "late_lift": 0.1, "stability": -0.1}},
            },
            "target_availability": {
                "reach": {"objective_available": True, "components": {"early_velocity": True, "core_velocity": True, "late_lift": True, "stability": True}},
                "engagement": {"objective_available": True, "components": {"early_velocity": True, "core_velocity": True, "late_lift": True, "stability": True}},
                "conversion": {"objective_available": True, "components": {"early_velocity": True, "core_velocity": True, "late_lift": True, "stability": True}},
            },
        },
        {
            "row_id": "v2::2026-03-11T00:00:00Z",
            "video_id": "v2",
            "author_id": "a2",
            "topic_key": "#cooking tips",
            "content_type": "tutorial",
            "as_of_time": "2026-03-11T00:00:00Z",
            "features": {"language": "en", "missingness_flags": []},
            "labels_trajectory": {
                "reach": {"series": {"t0": 1.0, "t6": 1.8, "t24": 2.2, "t96": 2.3}, "components": {"early_velocity": 0.15, "core_velocity": 0.02, "late_lift": 0.1, "stability": -0.4}},
                "engagement": {"series": {"t0": 0.1, "t6": 0.2, "t24": 0.25, "t96": 0.26}, "components": {"early_velocity": 0.1, "core_velocity": 0.01, "late_lift": 0.01, "stability": -0.2}},
                "conversion": {"series": {"t0": 0.1, "t6": 0.2, "t24": 0.25, "t96": 0.26}, "components": {"early_velocity": 0.1, "core_velocity": 0.01, "late_lift": 0.01, "stability": -0.2}},
            },
            "target_availability": {
                "reach": {"objective_available": True, "components": {"early_velocity": True, "core_velocity": True, "late_lift": True, "stability": True}},
                "engagement": {"objective_available": True, "components": {"early_velocity": True, "core_velocity": True, "late_lift": True, "stability": True}},
                "conversion": {"objective_available": True, "components": {"early_velocity": True, "core_velocity": True, "late_lift": True, "stability": True}},
            },
        },
    ]
    trajectory_bundle = build_trajectory_bundle(rows=rows, config=TrajectoryBuildConfig(embedding_dim=8))
    retriever = HybridRetriever.train(
        rows=rows,
        trajectory_vectors={
            "v1::2026-03-10T00:00:00Z": trajectory_bundle.embeddings_by_video["v1"],
            "v2::2026-03-11T00:00:00Z": trajectory_bundle.embeddings_by_video["v2"],
        },
        trajectory_lookup={"video": trajectory_bundle.embeddings_by_video},
        trajectory_metadata={
            "trajectory_manifest_id": trajectory_bundle.trajectory_manifest_id,
            "trajectory_version": trajectory_bundle.version,
            "trajectory_schema_hash": trajectory_bundle.trajectory_schema_hash,
        },
        objective_blend={
            "engagement": {
                "lexical": 0.0,
                "dense_text": 0.0,
                "multimodal": 0.0,
                "graph_dense": 0.0,
                "trajectory_dense": 1.0,
            }
        },
    )
    items, metadata = retriever.retrieve(
        query_row={
            "row_id": "q1",
            "video_id": "v1",
            "author_id": "a1",
            "topic_key": "#growth tutorial",
            "as_of_time": "2026-03-20T00:00:00Z",
            "features": {
                "trajectory_features": trajectory_bundle.profile_by_video["v1"]["features"],
                "missingness_flags": [],
            },
        },
        top_k=2,
        objective="engagement",
        return_metadata=True,
    )
    assert len(items) == 2
    assert "trajectory_dense" in items[0]["retrieval_branch_scores"]
    assert metadata["trajectory_manifest_id"] == trajectory_bundle.trajectory_manifest_id


def test_retriever_constraint_relaxation_uses_tier_order():
    rows = [
        {
            "row_id": "v1::2026-03-10T00:00:00Z",
            "video_id": "v1",
            "author_id": "a1",
            "topic_key": "alpha",
            "content_type": "tutorial",
            "locale": "en-gb",
            "as_of_time": "2026-03-10T00:00:00Z",
            "features": {"language": "en", "missingness_flags": []},
        }
    ]
    retriever = HybridRetriever.train(rows=rows)
    query = {
        "row_id": "q1",
        "topic_key": "alpha",
        "as_of_time": "2026-03-20T00:00:00Z",
        "features": {"missingness_flags": []},
    }
    _, meta = retriever.retrieve(
        query_row=query,
        top_k=1,
        objective="engagement",
        retrieval_constraints={"language": "en", "locale": "en-us", "content_type": "tutorial"},
        return_metadata=True,
    )
    assert meta["constraint_tier_used"] == 1


def test_retriever_enforces_temporal_cutoff_candidate_before_query():
    rows = [
        {
            "row_id": "v1::2026-03-10T00:00:00Z",
            "video_id": "v1",
            "author_id": "a1",
            "topic_key": "alpha",
            "content_type": "tutorial",
            "as_of_time": "2026-03-10T00:00:00Z",
            "features": {"language": "en", "missingness_flags": []},
        },
        {
            "row_id": "v2::2026-03-22T00:00:00Z",
            "video_id": "v2",
            "author_id": "a2",
            "topic_key": "alpha",
            "content_type": "tutorial",
            "as_of_time": "2026-03-22T00:00:00Z",
            "features": {"language": "en", "missingness_flags": []},
        },
    ]
    retriever = HybridRetriever.train(rows=rows)
    query = {
        "row_id": "q1",
        "topic_key": "alpha",
        "as_of_time": "2026-03-20T00:00:00Z",
        "features": {"missingness_flags": []},
    }
    items = retriever.retrieve(query_row=query, top_k=5, objective="engagement")
    assert len(items) == 1
    assert items[0]["candidate_id"] == "v1"


def test_retriever_objective_blend_changes_rank_order_with_multimodal_signal():
    rows = [
        {
            "row_id": "v1::2026-03-10T00:00:00Z",
            "video_id": "v1",
            "author_id": "a1",
            "topic_key": "alpha tutorial",
            "content_type": "tutorial",
            "as_of_time": "2026-03-10T00:00:00Z",
            "features": {"language": "en", "missingness_flags": []},
        },
        {
            "row_id": "v2::2026-03-10T00:00:00Z",
            "video_id": "v2",
            "author_id": "a2",
            "topic_key": "beta tutorial",
            "content_type": "tutorial",
            "as_of_time": "2026-03-10T00:00:00Z",
            "features": {"language": "en", "missingness_flags": []},
        },
    ]
    retriever = HybridRetriever.train(
        rows=rows,
        multimodal_vectors={
            "v1::2026-03-10T00:00:00Z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "v2::2026-03-10T00:00:00Z": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        },
        objective_blend={
            "reach": {"lexical": 1.0, "dense_text": 0.0, "multimodal": 0.0},
            "conversion": {"lexical": 0.0, "dense_text": 0.0, "multimodal": 1.0},
        },
    )
    query = {
        "row_id": "q1",
        "topic_key": "alpha tutorial",
        "as_of_time": "2026-03-20T00:00:00Z",
        "features": {"missingness_flags": []},
        "_fabric_output": {
            "text": {"clarity_score": 1.0, "token_count": 1.0, "hashtag_count": 1.0},
            "structure": {"hook_timing_seconds": 1.0, "payoff_timing_seconds": 1.0},
            "visual": {"visual_motion_score": 1.0},
            "audio": {"speech_ratio": 1.0},
        },
    }
    reach_items = retriever.retrieve(query_row=query, top_k=2, objective="reach")
    conversion_items = retriever.retrieve(query_row=query, top_k=2, objective="conversion")
    assert len(reach_items) == 2
    assert len(conversion_items) == 2
    assert reach_items[0]["candidate_id"] != conversion_items[0]["candidate_id"]


def test_runtime_accepts_candidate_video_ids_for_intersection(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="runtime-candidate-intersection"),
    )
    runtime = RecommenderRuntime(bundle_dir=Path(result["bundle_dir"]))
    rows = mart["rows"]
    query = next(row for row in rows if row["split"] == "test")
    candidates = [
        {
            "candidate_id": row["video_id"],
            "text": f"{row['topic_key']} {row['video_id']}",
            "topic_key": row["topic_key"],
            "author_id": row["author_id"],
            "as_of_time": row["as_of_time"],
        }
        for row in rows
        if row["as_of_time"] < query["as_of_time"]
    ]
    response = runtime.recommend(
        objective="engagement",
        as_of_time=query["as_of_time"],
        query={
            "query_id": query["row_id"],
            "text": f"{query['topic_key']} {query['row_id']}",
            "topic_key": query["topic_key"],
            "author_id": query["author_id"],
        },
        candidates=candidates,
        candidate_ids=[candidate["candidate_id"] for candidate in candidates],
        top_k=5,
        retrieve_k=20,
    )
    assert len(response["items"]) > 0
    assert response.get("retrieval_mode") in {"global", "intersected"}
    assert isinstance(response["items"][0].get("retrieval_branch_scores"), dict)


def test_runtime_trajectory_controls_disable_branch(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="runtime-trajectory-controls"),
    )
    runtime = RecommenderRuntime(bundle_dir=Path(result["bundle_dir"]))
    rows = mart["rows"]
    query = next(row for row in rows if row["split"] == "test")
    candidates = [
        {
            "candidate_id": row["video_id"],
            "text": f"{row['topic_key']} {row['video_id']}",
            "topic_key": row["topic_key"],
            "author_id": row["author_id"],
            "as_of_time": row["as_of_time"],
        }
        for row in rows
        if row["as_of_time"] < query["as_of_time"]
    ]
    response = runtime.recommend(
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
        trajectory_controls={"enabled": False},
    )
    assert response["trajectory_mode"] == "trajectory_branch_disabled"
    assert "trajectory_prediction" in response
    assert "trajectory_dense" in response["items"][0]["retrieval_branch_scores"]


def test_runtime_policy_reranker_enforces_author_cap_from_manifest(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="runtime-policy-cap"),
    )
    bundle_dir = Path(result["bundle_dir"])
    manifest_path = bundle_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = payload.get("policy_reranker") if isinstance(payload.get("policy_reranker"), dict) else {}
    policy["max_items_per_author"] = 1
    payload["policy_reranker"] = policy
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    runtime = RecommenderRuntime(bundle_dir=bundle_dir)
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
    assert len(response["items"]) == 1
    assert response["policy_metadata"]["max_items_per_author"] == 1


def test_runtime_policy_overrides_support_strict_language(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="runtime-policy-override"),
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
            "language": "es",
        }
        for row in rows
        if row["as_of_time"] < query["as_of_time"]
    ]
    default_response = runtime.recommend(
        objective="engagement",
        as_of_time=query["as_of_time"],
        query={
            "query_id": query["row_id"],
            "text": f"{query['topic_key']} {query['row_id']}",
            "topic_key": query["topic_key"],
            "author_id": query["author_id"],
            "language": "en",
        },
        candidates=candidates,
        top_k=5,
        retrieve_k=20,
    )
    strict_response = runtime.recommend(
        objective="engagement",
        as_of_time=query["as_of_time"],
        query={
            "query_id": query["row_id"],
            "text": f"{query['topic_key']} {query['row_id']}",
            "topic_key": query["topic_key"],
            "author_id": query["author_id"],
            "language": "en",
        },
        candidates=candidates,
        top_k=5,
        retrieve_k=20,
        policy_overrides={"strict_language": True},
    )
    assert len(default_response["items"]) > 0
    assert strict_response["policy_metadata"]["strict_language"] is True
    assert strict_response["policy_metadata"]["dropped_by_rule"]["language_mismatch"] >= 1
    assert len(strict_response["items"]) == 0


def test_runtime_calibrator_compatibility_guard_falls_back_to_identity(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="runtime-calibration-guard"),
    )
    bundle_dir = Path(result["bundle_dir"])
    calibration_path = bundle_dir / "rankers" / "engagement" / "calibration.json"
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    compatibility = payload.get("compatibility") if isinstance(payload.get("compatibility"), dict) else {}
    compatibility["feature_schema_hash"] = "bad-schema"
    payload["compatibility"] = compatibility
    calibration_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    runtime = RecommenderRuntime(bundle_dir=bundle_dir)
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
        objective="engagement",
        as_of_time=query["as_of_time"],
        query={
            "query_id": query["row_id"],
            "text": f"{query['topic_key']} {query['row_id']}",
            "topic_key": query["topic_key"],
            "author_id": query["author_id"],
        },
        candidates=candidates,
        top_k=3,
        retrieve_k=20,
    )
    assert response["calibration_metadata"]["calibrator_available"] is False
    assert "incompatible" in str(response["calibration_metadata"]["load_warning"])


def test_runtime_explainability_enabled_returns_cards_and_counterfactuals(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="runtime-explainability-enabled"),
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
        objective="engagement",
        as_of_time=query["as_of_time"],
        query={
            "query_id": query["row_id"],
            "text": f"{query['topic_key']} {query['row_id']}",
            "topic_key": query["topic_key"],
            "author_id": query["author_id"],
            "language": "en",
        },
        candidates=candidates,
        top_k=3,
        retrieve_k=20,
        explainability={
            "enabled": True,
            "top_features": 4,
            "neighbor_k": 2,
            "run_counterfactuals": True,
        },
    )
    assert isinstance(response.get("explainability_metadata"), dict)
    assert response["items"]
    first_item = response["items"][0]
    assert isinstance(first_item.get("evidence_cards"), dict)
    assert isinstance(first_item.get("temporal_confidence_band"), dict)
    assert isinstance(first_item.get("counterfactual_scenarios"), list)
    assert len(first_item["counterfactual_scenarios"]) == 10


def test_runtime_explainability_deterministic_for_same_input(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="runtime-explainability-deterministic"),
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
    kwargs = {
        "objective": "engagement",
        "as_of_time": query["as_of_time"],
        "query": {
            "query_id": query["row_id"],
            "text": f"{query['topic_key']} {query['row_id']}",
            "topic_key": query["topic_key"],
            "author_id": query["author_id"],
            "language": "en",
        },
        "candidates": candidates,
        "top_k": 3,
        "retrieve_k": 20,
        "explainability": {
            "enabled": True,
            "top_features": 4,
            "neighbor_k": 2,
            "run_counterfactuals": True,
        },
    }
    first = runtime.recommend(**kwargs)
    second = runtime.recommend(**kwargs)
    assert first["items"][0]["evidence_cards"] == second["items"][0]["evidence_cards"]
    assert first["items"][0]["counterfactual_scenarios"] == second["items"][0]["counterfactual_scenarios"]
    baseline = runtime.recommend(
        objective="engagement",
        as_of_time=query["as_of_time"],
        query={
            "query_id": query["row_id"],
            "text": f"{query['topic_key']} {query['row_id']}",
            "topic_key": query["topic_key"],
            "author_id": query["author_id"],
            "language": "en",
        },
        candidates=candidates,
        top_k=3,
        retrieve_k=20,
    )
    assert "explainability_metadata" not in baseline
    assert "evidence_cards" not in baseline["items"][0]


def test_runtime_routing_contract_mismatch_raises(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="runtime-routing-mismatch"),
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
    with pytest.raises(RoutingContractError):
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
            top_k=3,
            retrieve_k=20,
            routing={
                "objective_requested": "engagement",
                "objective_effective": "conversion",
                "track": "post_publication",
            },
        )


def test_runtime_required_compat_mismatch_raises(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="runtime-compat-mismatch"),
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
    with pytest.raises(ArtifactCompatibilityError):
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
            top_k=3,
            retrieve_k=20,
            routing={
                "objective_requested": "engagement",
                "objective_effective": "engagement",
                "track": "post_publication",
                "required_compat": {"feature_schema_hash": "bad"},
            },
        )


def test_runtime_routing_invalid_request_id_raises(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="runtime-routing-invalid-request-id"),
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
    with pytest.raises(RoutingContractError):
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
            top_k=3,
            retrieve_k=20,
            routing={
                "objective_requested": "engagement",
                "objective_effective": "engagement",
                "track": "post_publication",
                "request_id": "bad-request-id",
            },
        )


def test_runtime_explainability_timeout_degrades_with_reason(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="runtime-explain-timeout"),
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
        objective="engagement",
        as_of_time=query["as_of_time"],
        query={
            "query_id": query["row_id"],
            "text": f"{query['topic_key']} {query['row_id']}",
            "topic_key": query["topic_key"],
            "author_id": query["author_id"],
        },
        candidates=candidates,
        top_k=3,
        retrieve_k=20,
        explainability={"enabled": True, "run_counterfactuals": True},
        routing={
            "objective_requested": "engagement",
            "objective_effective": "engagement",
            "track": "post_publication",
            "stage_budgets_ms": {"explainability": 0.0001},
        },
    )
    assert response["fallback_mode"] is True
    assert response["fallback_reason"] == "explainability_timeout"


def test_runtime_routing_request_id_and_experiment_echo(tmp_path: Path):
    mart = make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="runtime-routing-request-id"),
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
        objective="engagement",
        as_of_time=query["as_of_time"],
        query={
            "query_id": query["row_id"],
            "text": f"{query['topic_key']} {query['row_id']}",
            "topic_key": query["topic_key"],
            "author_id": query["author_id"],
        },
        candidates=candidates,
        top_k=3,
        retrieve_k=20,
        routing={
            "objective_requested": "engagement",
            "objective_effective": "engagement",
            "track": "post_publication",
            "request_id": "018f0f57-21cb-7f81-8d17-6efec2b5f2be",
            "experiment": {
                "id": "rec_v2_default",
                "variant": "treatment",
                "unit_hash": "abc123",
            },
        },
        debug=True,
    )
    assert response["request_id"] == "018f0f57-21cb-7f81-8d17-6efec2b5f2be"
    assert response["experiment_id"] == "rec_v2_default"
    assert response["variant"] == "treatment"
    assert isinstance(response["debug"]["retrieved_universe"], list)
    assert isinstance(response["debug"]["ranking_universe"], list)
