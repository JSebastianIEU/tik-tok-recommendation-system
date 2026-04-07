from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from src.recommendation import BuildTrainingDataMartConfig, CanonicalDatasetBundle, build_training_data_mart
from src.recommendation.learning import RecommenderRuntime, RecommenderTrainingConfig, train_recommender_from_datamart
from src.recommendation.learning.retriever import HybridRetriever, HybridRetrieverTrainerConfig


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def test_hybrid_retriever_blends_creator_memory_into_retrieval():
    rows = [
        {
            "row_id": "dance::1",
            "video_id": "dance-1",
            "caption": "growth tutorial ideas",
            "hashtags": ["#growth"],
            "keywords": ["growth", "tutorial"],
            "topic_key": "dance",
            "author_id": "creator-dance",
            "content_type": "challenge",
            "language": "en",
            "locale": "en-us",
            "as_of_time": "2026-01-01T00:00:00Z",
            "posted_at": "2026-01-01T00:00:00Z",
        },
        {
            "row_id": "tech::1",
            "video_id": "tech-1",
            "caption": "growth tutorial ideas",
            "hashtags": ["#growth"],
            "keywords": ["growth", "tutorial"],
            "topic_key": "tech",
            "author_id": "creator-tech",
            "content_type": "review",
            "language": "en",
            "locale": "en-us",
            "as_of_time": "2026-01-02T00:00:00Z",
            "posted_at": "2026-01-02T00:00:00Z",
        },
        {
            "row_id": "food::1",
            "video_id": "food-1",
            "caption": "banana recipe kitchen tips",
            "hashtags": ["#food"],
            "keywords": ["banana", "recipe"],
            "topic_key": "food",
            "author_id": "creator-food",
            "content_type": "recipe",
            "language": "en",
            "locale": "en-us",
            "as_of_time": "2026-01-03T00:00:00Z",
            "posted_at": "2026-01-03T00:00:00Z",
        },
    ]
    retriever = HybridRetriever.train(
        rows,
        config=HybridRetrieverTrainerConfig(),
        multimodal_vectors={
            "dance::1": [1.0, 0.0, 0.0, 0.0],
            "tech::1": [0.0, 1.0, 0.0, 0.0],
            "food::1": [0.0, 0.0, 1.0, 0.0],
        },
        objective_blend={
            "engagement": {
                "lexical": 0.45,
                "dense_text": 0.25,
                "multimodal": 0.30,
                "graph_dense": 0.0,
                "trajectory_dense": 0.0,
            }
        },
    )
    query = {
        "row_id": "query-1",
        "video_id": "query-1",
        "caption": "growth tutorial ideas",
        "hashtags": ["#growth"],
        "keywords": ["growth", "tutorial"],
        "topic_key": "general",
        "author_id": "seed-author",
        "content_type": "tutorial",
        "language": "en",
        "locale": "en-us",
        "as_of_time": "2026-01-05T00:00:00Z",
        "_fabric_output": {
            "text": {"clarity_score": 1.0, "token_count": 1.0, "hashtag_count": 0.0},
            "structure": {"hook_timing_seconds": 0.0, "payoff_timing_seconds": 0.0},
            "audio": {"speech_ratio": 0.0},
            "visual": {"visual_motion_score": 0.0},
        },
    }
    vanilla_items, vanilla_meta = retriever.retrieve(
        query_row=query,
        top_k=2,
        objective="engagement",
        return_metadata=True,
    )
    assert vanilla_meta["creator_retrieval"]["applied"] is False
    assert vanilla_items[0]["candidate_id"] == "dance-1"

    personalized_items, personalized_meta = retriever.retrieve(
        query_row=query,
        top_k=2,
        objective="engagement",
        user_context={
            "creator_id": "creator-1",
            "objective_effective": "engagement",
            "last_feedback_at": "2026-01-04T00:00:00Z",
            "support": {
                "explicit_positive_count": 6,
                "explicit_negative_count": 5,
                "explicit_request_count": 5,
                "objective_request_count": 4,
            },
            "objective_candidate_memory": {
                "positive_candidate_ids": ["tech-1"],
                "negative_candidate_ids": ["dance-1"],
            },
        },
        return_metadata=True,
    )
    assert personalized_meta["creator_retrieval"]["applied"] is True
    assert personalized_items[0]["candidate_id"] == "tech-1"
    assert personalized_items[0]["creator_retrieval_score"] > personalized_items[1]["creator_retrieval_score"]
    assert personalized_items[0]["creator_retrieval_trace"]["score_shift"] > 0.0


def _make_datamart() -> dict:
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


def test_runtime_surfaces_retrieval_personalization_metadata(tmp_path: Path):
    mart = _make_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="runtime-personalized-retrieval"),
    )
    runtime = RecommenderRuntime(bundle_dir=Path(result["bundle_dir"]))
    query = next(row for row in mart["rows"] if row["split"] == "test")
    positive_candidate = next(row for row in mart["rows"] if row["split"] == "train")
    negative_candidate = next(
        row for row in mart["rows"] if row["split"] == "validation" and row["row_id"] != positive_candidate["row_id"]
    )
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
        user_context={
            "creator_id": "creator-55",
            "objective_effective": "engagement",
            "last_feedback_at": "2026-02-20T00:00:00Z",
            "support": {
                "explicit_positive_count": 7,
                "explicit_negative_count": 6,
                "explicit_request_count": 5,
                "objective_request_count": 4,
            },
            "objective_candidate_memory": {
                "positive_candidate_ids": [positive_candidate["video_id"]],
                "negative_candidate_ids": [negative_candidate["video_id"]],
            },
        },
        top_k=5,
        retrieve_k=20,
        debug=True,
    )
    assert "retrieval_personalization_metadata" in response
    assert response["retrieval_personalization_metadata"]["enabled"] is True
    assert "creator_retrieval_score" in response["items"][0]
    assert "creator_retrieval_trace" in response["items"][0]
    assert "personalization" in response["debug"]["config"]["retrieval"]
