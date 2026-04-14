from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.recommendation import (
    BuildTrainingDataMartConfig,
    CanonicalDatasetBundle,
    build_contract_manifest,
    build_training_data_mart,
)
from src.recommendation.learning import (
    LearnedPairwiseReranker,
    PairwiseTrainingRow,
    RecommenderRuntime,
    RecommenderTrainingConfig,
    candidate_feature_payload_from_item,
    materialize_datamart_bootstrap_rows,
    materialize_labeling_session_rows,
    materialize_pairwise_rows,
    resolve_candidate_feedback_state,
    summarize_feedback_training_support,
    train_recommender_from_datamart,
)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def make_small_bundle() -> CanonicalDatasetBundle:
    generated_at = _dt("2026-03-01T00:00:00Z")
    authors = [{"author_id": "a1", "followers_count": 1500}]
    videos = []
    snapshots = []
    base_time = _dt("2026-01-01T00:00:00Z")
    for idx in range(1, 9):
        video_id = f"v{idx}"
        posted_at = base_time + timedelta(days=idx)
        topic = "beauty" if idx % 2 == 0 else "tech"
        videos.append(
            {
                "video_id": video_id,
                "author_id": "a1",
                "caption": f"{topic} tutorial {idx}",
                "hashtags": [f"#{topic}", f"#topic{idx%3}"],
                "keywords": [topic, "tutorial"],
                "search_query": topic,
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
                "views": 600 * idx,
                "likes": 70 * idx,
                "comments_count": 12 * idx,
                "shares": 5 * idx,
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
    return bundle


def make_small_datamart() -> dict:
    bundle = make_small_bundle()
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
            pair_candidates_per_query=5,
        ),
    )


def test_resolve_candidate_feedback_state_prioritizes_saved_and_conflict():
    saved = resolve_candidate_feedback_state(
        [{"event_name": "comparable_saved"}, {"event_name": "comparable_marked_relevant"}]
    )
    assert saved.state == "saved"
    assert saved.conflict is False

    conflict = resolve_candidate_feedback_state(
        [
            {"event_name": "comparable_marked_relevant"},
            {"event_name": "comparable_marked_not_relevant"},
        ]
    )
    assert conflict.state == "conflict"
    assert conflict.conflict is True

    weak_positive = resolve_candidate_feedback_state(
        [{"event_name": "comparable_opened"}]
    )
    assert weak_positive.state == "weak_positive"
    assert weak_positive.conflict is False


def test_materialize_pairwise_rows_builds_forward_and_reverse_pairs():
    served_item_a = {
        "candidate_id": "c1",
        "score": 0.8,
        "score_components": {
            "semantic_relevance": 0.9,
            "intent_alignment": 0.8,
            "reference_usefulness": 0.7,
            "support_confidence": 0.85,
        },
        "retrieval_branch_scores": {
            "semantic": 0.9,
            "hashtag_topic": 0.7,
            "structured_compatibility": 0.6,
            "fused_retrieval": 0.8,
        },
        "similarity": {"sparse": 0.9, "dense": 0.8, "fused": 0.8},
        "support_level": "full",
        "support_score": 0.9,
        "confidence": 0.8,
        "comment_trace": {
            "alignment_score": 0.7,
            "value_prop_coverage": 0.65,
            "on_topic_ratio": 0.6,
            "artifact_drift_ratio": 0.1,
            "alignment_confidence": 0.7,
        },
        "trajectory_trace": {"similarity": 0.2, "regime_confidence": 0.3},
        "ranking_reasons": ["strong_semantic_relevance", "fully_supported_reference"],
        "retrieval_branches": ["semantic", "hashtag_topic"],
    }
    served_item_b = {
        **served_item_a,
        "candidate_id": "c2",
        "score": 0.3,
        "score_components": {
            "semantic_relevance": 0.2,
            "intent_alignment": 0.5,
            "reference_usefulness": 0.4,
            "support_confidence": 0.6,
        },
        "support_level": "partial",
        "support_score": 0.55,
        "ranking_reasons": ["strong_intent_alignment"],
    }
    rows = materialize_pairwise_rows(
        requests=[{"request_id": "r1", "objective_effective": "engagement"}],
        served_outputs=[
            {"request_id": "r1", "candidate_id": "c1", "metadata": served_item_a},
            {"request_id": "r1", "candidate_id": "c2", "metadata": served_item_b},
        ],
        feedback_events=[
            {
                "request_id": "r1",
                "event_name": "comparable_marked_relevant",
                "entity_type": "comparable",
                "entity_id": "c1",
            },
            {
                "request_id": "r1",
                "event_name": "comparable_marked_not_relevant",
                "entity_type": "comparable",
                "entity_id": "c2",
            },
        ],
    )
    assert len(rows) == 2
    assert rows[0].candidate_a_id == "c1"
    assert rows[0].label == 1
    assert rows[1].candidate_a_id == "c2"
    assert rows[1].label == 0


def test_materialize_pairwise_rows_respects_served_rank_band_and_no_good_summary():
    served_item = {
        "candidate_id": "c1",
        "score": 0.8,
        "score_components": {
            "semantic_relevance": 0.9,
            "intent_alignment": 0.8,
            "reference_usefulness": 0.7,
            "support_confidence": 0.85,
        },
        "retrieval_branch_scores": {
            "semantic": 0.9,
            "hashtag_topic": 0.7,
            "structured_compatibility": 0.6,
            "fused_retrieval": 0.8,
        },
        "similarity": {"sparse": 0.9, "dense": 0.8, "fused": 0.8},
        "support_level": "partial",
        "support_score": 0.6,
        "confidence": 0.8,
        "comment_trace": {},
        "trajectory_trace": {},
        "ranking_reasons": [],
        "retrieval_branches": ["semantic"],
    }
    rows = materialize_pairwise_rows(
        requests=[{"request_id": "r1", "objective_effective": "engagement"}],
        served_outputs=[
            {"request_id": "r1", "candidate_id": "c1", "rank": 1, "metadata": served_item},
            {"request_id": "r1", "candidate_id": "c2", "rank": 7, "metadata": {**served_item, "candidate_id": "c2"}},
        ],
        feedback_events=[
            {
                "request_id": "r1",
                "event_name": "comparable_saved",
                "entity_type": "comparable",
                "entity_id": "c1",
            },
            {
                "request_id": "r1",
                "event_name": "comparable_marked_not_relevant",
                "entity_type": "comparable",
                "entity_id": "c2",
            },
            {
                "request_id": "r1",
                "event_name": "comparable_no_good_options",
                "entity_type": "report",
                "section": "comparables",
            },
        ],
        max_served_rank=5,
    )
    summary = summarize_feedback_training_support(
        requests=[{"request_id": "r1", "objective_effective": "engagement"}],
        served_outputs=[
            {"request_id": "r1", "candidate_id": "c1", "rank": 1, "metadata": served_item},
            {"request_id": "r1", "candidate_id": "c2", "rank": 7, "metadata": {**served_item, "candidate_id": "c2"}},
        ],
        feedback_events=[
            {
                "request_id": "r1",
                "event_name": "comparable_saved",
                "entity_type": "comparable",
                "entity_id": "c1",
            },
            {
                "request_id": "r1",
                "event_name": "comparable_marked_not_relevant",
                "entity_type": "comparable",
                "entity_id": "c2",
            },
            {
                "request_id": "r1",
                "event_name": "comparable_no_good_options",
                "entity_type": "report",
                "section": "comparables",
            },
        ],
        max_served_rank=5,
    )
    assert rows == []
    assert summary["engagement"]["no_good_option_request_count"] == 1
    assert summary["engagement"]["explicit_positive_request_count"] == 1
    assert summary["engagement"]["explicit_negative_request_count"] == 0
    assert summary["engagement"]["filtered_by_rank_feedback_event_count"] == 1


def test_materialize_datamart_bootstrap_rows_builds_runtime_aligned_pairs(tmp_path: Path):
    bundle = make_small_bundle()
    manifest = build_contract_manifest(bundle, tmp_path / "manifests", as_of_time=bundle.generated_at)
    mart = build_training_data_mart(
        bundle,
        config=BuildTrainingDataMartConfig(
            track="pre_publication",
            min_history_hours=24,
            label_window_hours=72,
            train_ratio=0.6,
            validation_ratio=0.2,
            include_pair_rows=True,
            pair_objective="reach",
            pair_candidates_per_query=5,
            source_manifest_path=str(manifest["manifest_dir"]),
        ),
    )
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path / "artifacts",
        config=RecommenderTrainingConfig(
            objectives=("reach", "engagement", "conversion"),
            retrieve_k=30,
            run_name="bootstrap-pair-test",
            dense_model_name="sentence-transformers/all-MiniLM-L6-v2",
        ),
    )
    rows = materialize_datamart_bootstrap_rows(
        datamart=mart,
        bundle_dir=Path(result["bundle_dir"]),
        source_bundle=bundle,
        objectives=["reach"],
    )

    assert len(rows) > 0
    assert all(row.objective_effective == "reach" for row in rows)
    assert any(row.pair_source.startswith("bootstrap_rel") for row in rows)
    assert all(row.candidate_a_id != row.candidate_b_id for row in rows)
    sample = rows[0]
    assert "baseline_score" in sample.features_a
    assert "retrieval_fused" in sample.features_a
    assert "score_component_semantic_relevance" in sample.features_a
    assert sample.pair_weight > 0.0


def test_materialize_labeling_session_rows_builds_runtime_aligned_pairs(tmp_path: Path):
    mart = make_small_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(
            objectives=("reach", "engagement", "conversion"),
            retrieve_k=30,
            run_name="labeling-session-pair-test",
            dense_model_name="sentence-transformers/all-MiniLM-L6-v2",
        ),
    )
    bundle_dir = Path(result["bundle_dir"])
    rows = mart["rows"]
    query = next(row for row in rows if row["split"] == "test")
    candidate_payloads = [
        {
            "candidate_id": row["row_id"],
            "video_id": row["video_id"],
            "text": f"{row['topic_key']} {row['row_id']}",
            "caption": f"{row['topic_key']} {row['row_id']}",
            "hashtags": [f"#{row['topic_key']}"] if row.get("topic_key") else [],
            "keywords": [str(row["topic_key"])] if row.get("topic_key") else [],
            "topic_key": row["topic_key"],
            "author_id": row["author_id"],
            "as_of_time": row["as_of_time"],
            "posted_at": row["as_of_time"],
            "language": "en",
            "content_type": "tutorial",
            "signal_hints": {
                "comment_intelligence": {
                    "available": True,
                    "alignment_score": 0.7,
                    "value_prop_coverage": 0.6,
                    "on_topic_ratio": 0.65,
                    "artifact_drift_ratio": 0.1,
                    "alignment_confidence": 0.7,
                }
            },
        }
        for row in rows
        if row["as_of_time"] < query["as_of_time"]
    ][:4]
    assert len(candidate_payloads) >= 4

    session_payload = {
        "version": "labeling.review_session.v1",
        "session_id": "session-1",
        "session_name": "test-session",
        "created_at": "2026-04-06T00:00:00Z",
        "updated_at": "2026-04-06T00:05:00Z",
        "source": {
            "source_id": "training-seed",
            "file_name": "training.json",
            "source_path": str(tmp_path / "training.json"),
            "generated_at": "2026-04-06T00:00:00Z",
            "case_count": 1,
            "objectives": ["engagement"],
        },
        "rubric": {
            "version": "labeling.review_session.v1",
            "labels": ["saved", "relevant", "not_relevant"],
            "instructions": [],
        },
        "cases": [
            {
                "case_id": "engagement::test-query",
                "objective": "engagement",
                "query": {
                    "query_id": query["row_id"],
                    "display": {"created_at": query["as_of_time"]},
                    "query_payload": {
                        "query_id": query["row_id"],
                        "text": f"{query['topic_key']} {query['row_id']}",
                        "description": f"{query['topic_key']} {query['row_id']}",
                        "hashtags": [f"#{query['topic_key']}"] if query.get("topic_key") else [],
                        "keywords": [str(query["topic_key"])] if query.get("topic_key") else [],
                        "author_id": query["author_id"],
                        "as_of_time": query["as_of_time"],
                    },
                },
                "retrieve_k": len(candidate_payloads),
                "label_pool_size": len(candidate_payloads),
                "source_candidate_pool_size": len(candidate_payloads),
                "notes": "",
                "candidates": [
                    {
                        "candidate_id": candidate_payloads[0]["candidate_id"],
                        "display": {},
                        "candidate_payload": candidate_payloads[0],
                        "baseline_rank": 1,
                        "baseline_score": 0.9,
                        "support_level": "partial",
                        "ranking_reasons": [],
                        "review": {"label": "saved", "note": "", "updated_at": "2026-04-06T00:01:00Z"},
                    },
                    {
                        "candidate_id": candidate_payloads[1]["candidate_id"],
                        "display": {},
                        "candidate_payload": candidate_payloads[1],
                        "baseline_rank": 2,
                        "baseline_score": 0.8,
                        "support_level": "partial",
                        "ranking_reasons": [],
                        "review": {"label": "relevant", "note": "", "updated_at": "2026-04-06T00:02:00Z"},
                    },
                    {
                        "candidate_id": candidate_payloads[2]["candidate_id"],
                        "display": {},
                        "candidate_payload": candidate_payloads[2],
                        "baseline_rank": 3,
                        "baseline_score": 0.3,
                        "support_level": "partial",
                        "ranking_reasons": [],
                        "review": {"label": "not_relevant", "note": "", "updated_at": "2026-04-06T00:03:00Z"},
                    },
                    {
                        "candidate_id": candidate_payloads[3]["candidate_id"],
                        "display": {},
                        "candidate_payload": candidate_payloads[3],
                        "baseline_rank": 4,
                        "baseline_score": 0.2,
                        "support_level": "partial",
                        "ranking_reasons": [],
                        "review": {"label": "not_relevant", "note": "", "updated_at": "2026-04-06T00:04:00Z"},
                    },
                ],
            }
        ],
    }
    session_path = tmp_path / "labeling_session.json"
    session_path.write_text(json.dumps(session_payload, default=str), encoding="utf-8")

    rows = materialize_labeling_session_rows(
        session_json_paths=[session_path],
        bundle_dir=bundle_dir,
        objectives=["engagement"],
    )

    assert len(rows) == 10
    assert all(row.objective_effective == "engagement" for row in rows)
    assert any(row.pair_source == "labeling_saved_vs_not_relevant" for row in rows)
    assert any(row.pair_source == "labeling_relevant_vs_not_relevant" for row in rows)
    assert any(row.pair_source == "labeling_saved_vs_relevant" for row in rows)
    sample = rows[0]
    assert "baseline_score" in sample.features_a
    assert "retrieval_fused" in sample.features_a
    assert "score_component_semantic_relevance" in sample.features_a
    assert sample.pair_weight > 0.0


def test_learned_reranker_training_and_runtime_integration(tmp_path: Path):
    mart = make_small_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(
            objectives=("reach", "engagement", "conversion"),
            retrieve_k=30,
            run_name="phase2-runtime-test",
            dense_model_name="sentence-transformers/all-MiniLM-L6-v2",
        ),
    )
    bundle_dir = Path(result["bundle_dir"])
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
    baseline_response = runtime.recommend(
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
        routing={"stage_budgets_ms": {"retrieval": 5000, "ranking": 5000}},
        debug=True,
    )
    baseline_items = baseline_response["items"]
    assert len(baseline_items) >= 2
    baseline_top = baseline_items[0]
    baseline_bottom = baseline_items[-1]

    reranker = LearnedPairwiseReranker.train(
        objective="engagement",
        rows=[
            PairwiseTrainingRow(
                request_id="train-1",
                objective_effective="engagement",
                candidate_a_id=str(baseline_bottom["candidate_id"]),
                candidate_b_id=str(baseline_top["candidate_id"]),
                label=1,
                pair_source="positive_vs_negative",
                pair_weight=1.0,
                features_a=candidate_feature_payload_from_item(baseline_bottom),
                features_b=candidate_feature_payload_from_item(baseline_top),
            ),
            PairwiseTrainingRow(
                request_id="train-1",
                objective_effective="engagement",
                candidate_a_id=str(baseline_top["candidate_id"]),
                candidate_b_id=str(baseline_bottom["candidate_id"]),
                label=0,
                pair_source="positive_vs_negative_reverse",
                pair_weight=1.0,
                features_a=candidate_feature_payload_from_item(baseline_top),
                features_b=candidate_feature_payload_from_item(baseline_bottom),
            ),
        ],
    )
    reranker.save(bundle_dir / "rankers" / "engagement" / "learned_reranker")

    runtime_with_learned = RecommenderRuntime(bundle_dir=bundle_dir)
    learned_response = runtime_with_learned.recommend(
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
        routing={"stage_budgets_ms": {"retrieval": 5000, "ranking": 5000}},
        debug=True,
    )
    assert learned_response["learned_reranker_metadata"]["enabled"] is True
    assert "learned_score" in learned_response["items"][0]
    assert "baseline_score" in learned_response["items"][0]
    assert "learned_trace" in learned_response["items"][0]


def test_runtime_ignores_incompatible_learned_reranker_artifact(tmp_path: Path):
    mart = make_small_datamart()
    result = train_recommender_from_datamart(
        datamart=mart,
        artifact_root=tmp_path,
        config=RecommenderTrainingConfig(run_name="phase2-incompatible-test"),
    )
    bundle_dir = Path(result["bundle_dir"])
    output_dir = bundle_dir / "rankers" / "engagement" / "learned_reranker"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model.pkl").write_bytes(b"not-a-real-model")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "ranker_id": "learned_pairwise_lgbm",
                "version": "recommender.ranker.learned_pairwise.v1",
                "objective": "engagement",
                "model_type": "lightgbm_classifier",
                "feature_names": ["bad"],
                "feature_schema_hash": "bad-hash",
                "train_summary": {},
            }
        ),
        encoding="utf-8",
    )
    runtime = RecommenderRuntime(bundle_dir=bundle_dir)
    assert "engagement" in runtime.learned_reranker_load_warnings
    assert "engagement" not in runtime.learned_rerankers


def test_learned_reranker_can_flip_handcrafted_pair_order():
    preferred = {
        "candidate_id": "preferred",
        "score": 0.25,
        "score_components": {
            "semantic_relevance": 0.15,
            "intent_alignment": 0.40,
            "reference_usefulness": 0.35,
            "support_confidence": 0.55,
        },
        "retrieval_branch_scores": {
            "semantic": 0.10,
            "hashtag_topic": 0.15,
            "structured_compatibility": 0.35,
            "fused_retrieval": 0.20,
        },
        "similarity": {"sparse": 0.15, "dense": 0.40, "fused": 0.25},
        "support_level": "partial",
        "support_score": 0.55,
        "confidence": 0.55,
        "comment_trace": {
            "alignment_score": 0.35,
            "value_prop_coverage": 0.4,
            "on_topic_ratio": 0.35,
            "artifact_drift_ratio": 0.25,
            "alignment_confidence": 0.55,
        },
        "trajectory_trace": {"similarity": 0.25, "regime_confidence": 0.3},
        "trajectory_similarity": 0.25,
        "trajectory_regime_confidence": 0.3,
        "ranking_reasons": ["strong_intent_alignment"],
        "retrieval_branches": ["structured_compatibility"],
        "policy_penalty": 0.0,
        "policy_bonus": 0.0,
        "policy_adjusted_score": 0.25,
        "calibration_trace": {},
        "policy_trace": {},
        "portfolio_trace": None,
        "score_raw": 0.25,
        "score_calibrated": 0.25,
        "score_mean": 0.25,
        "score_std": 0.0,
        "selected_ranker_id": "baseline_weighted",
        "global_score_mean": 0.25,
        "segment_blend_weight": 0.0,
    }
    baseline_leader = {
        **preferred,
        "candidate_id": "baseline_leader",
        "score": 0.75,
        "score_components": {
            "semantic_relevance": 0.85,
            "intent_alignment": 0.75,
            "reference_usefulness": 0.70,
            "support_confidence": 0.82,
        },
        "retrieval_branch_scores": {
            "semantic": 0.90,
            "hashtag_topic": 0.75,
            "structured_compatibility": 0.55,
            "fused_retrieval": 0.78,
        },
        "similarity": {"sparse": 0.85, "dense": 0.75, "fused": 0.75},
        "support_level": "full",
        "support_score": 0.82,
        "confidence": 0.8,
        "comment_trace": {
            "alignment_score": 0.75,
            "value_prop_coverage": 0.7,
            "on_topic_ratio": 0.72,
            "artifact_drift_ratio": 0.1,
            "alignment_confidence": 0.75,
        },
        "trajectory_trace": {"similarity": 0.1, "regime_confidence": 0.2},
        "trajectory_similarity": 0.1,
        "trajectory_regime_confidence": 0.2,
        "ranking_reasons": ["strong_semantic_relevance", "fully_supported_reference"],
        "retrieval_branches": ["semantic", "hashtag_topic"],
        "policy_adjusted_score": 0.75,
        "score_raw": 0.75,
        "score_calibrated": 0.75,
        "score_mean": 0.75,
        "global_score_mean": 0.75,
    }
    rows = []
    for idx in range(8):
        rows.append(
            PairwiseTrainingRow(
                request_id=f"r{idx}",
                objective_effective="engagement",
                candidate_a_id="preferred",
                candidate_b_id="baseline_leader",
                label=1,
                pair_source="positive_vs_negative",
                pair_weight=1.0,
                features_a=candidate_feature_payload_from_item(preferred),
                features_b=candidate_feature_payload_from_item(baseline_leader),
            )
        )
        rows.append(
            PairwiseTrainingRow(
                request_id=f"r{idx}",
                objective_effective="engagement",
                candidate_a_id="baseline_leader",
                candidate_b_id="preferred",
                label=0,
                pair_source="positive_vs_negative_reverse",
                pair_weight=1.0,
                features_a=candidate_feature_payload_from_item(baseline_leader),
                features_b=candidate_feature_payload_from_item(preferred),
            )
        )
    reranker = LearnedPairwiseReranker.train(objective="engagement", rows=rows)
    reranked, meta = reranker.rerank_items(items=[baseline_leader, preferred])
    assert meta["applied"] is True
    reranked_by_id = {item["candidate_id"]: item for item in reranked}
    assert reranked_by_id["preferred"]["learned_score"] > reranked_by_id["baseline_leader"]["learned_score"]
    assert reranked_by_id["preferred"]["score"] > preferred["score"]
    assert reranked_by_id["baseline_leader"]["score"] < baseline_leader["score"]
