from __future__ import annotations

from pathlib import Path

from src.recommendation.learning import (
    LABEL_BAD,
    LABEL_GOOD,
    LABEL_UNCLEAR,
    BenchmarkCandidate,
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkQuery,
    aggregate_case_metrics,
    evaluate_case_ranked_ids,
    label_to_relevance,
    load_benchmark_dataset,
    save_benchmark_dataset,
    summarize_benchmark_dataset,
)


def _candidate(candidate_id: str, *, rank: int, label: str | None) -> BenchmarkCandidate:
    return BenchmarkCandidate(
        candidate_id=candidate_id,
        display={"video_id": candidate_id, "caption": candidate_id},
        candidate_payload={
            "candidate_id": candidate_id,
            "caption": candidate_id,
            "hashtags": [],
            "keywords": [],
            "author_id": f"author-{candidate_id}",
            "posted_at": "2026-04-01T00:00:00Z",
        },
        baseline_rank=rank,
        baseline_score=1.0 / float(rank),
        support_level="full",
        ranking_reasons=["strong_semantic_relevance"],
        label=label,
    )


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="engagement::q1",
        objective="engagement",
        query=BenchmarkQuery(
            query_id="q1",
            display={"video_id": "q1", "caption": "query"},
            query_payload={
                "query_id": "q1",
                "text": "query",
                "description": "query",
                "hashtags": ["#topic"],
                "keywords": ["topic"],
                "author_id": "author-q1",
                "as_of_time": "2026-04-02T00:00:00Z",
            },
        ),
        candidates=[
            _candidate("c1", rank=1, label=LABEL_GOOD),
            _candidate("c2", rank=2, label=LABEL_UNCLEAR),
            _candidate("c3", rank=3, label=LABEL_BAD),
            _candidate("c4", rank=4, label=None),
        ],
        retrieve_k=12,
        label_pool_size=4,
        source_candidate_pool_size=60,
    )


def test_label_to_relevance_uses_graded_scale():
    assert label_to_relevance(LABEL_GOOD) == 2.0
    assert label_to_relevance(LABEL_UNCLEAR) == 1.0
    assert label_to_relevance(LABEL_BAD) == 0.0
    assert label_to_relevance(None) == 0.0


def test_case_pairwise_preferences_ignore_unclear_and_unlabeled():
    case = _case()
    assert case.pairwise_preferences() == [("c1", "c3")]
    assert case.relevance_map() == {"c1": 2.0, "c2": 1.0, "c3": 0.0}
    assert case.relevant_ids() == {"c1"}


def test_evaluate_case_ranked_ids_scores_expected_ordering():
    case = _case()
    good_first = evaluate_case_ranked_ids(case, ["c1", "c2", "c3", "c4"], k_values=(1, 3))
    bad_first = evaluate_case_ranked_ids(case, ["c3", "c2", "c1", "c4"], k_values=(1, 3))

    assert good_first["ndcg@3"] > bad_first["ndcg@3"]
    assert good_first["mrr@3"] > bad_first["mrr@3"]
    assert good_first["recall@3"] >= bad_first["recall@3"]
    assert good_first["good_rate@1"] > bad_first["good_rate@1"]
    assert good_first["bad_rate@1"] < bad_first["bad_rate@1"]


def test_evaluate_case_ranked_ids_keeps_no_good_cases_in_scope():
    case = BenchmarkCase(
        case_id="engagement::q2",
        objective="engagement",
        query=_case().query,
        candidates=[
            _candidate("c1", rank=1, label=LABEL_UNCLEAR),
            _candidate("c2", rank=2, label=LABEL_BAD),
            _candidate("c3", rank=3, label=LABEL_UNCLEAR),
            _candidate("c4", rank=4, label=LABEL_BAD),
        ],
        retrieve_k=12,
        label_pool_size=4,
        source_candidate_pool_size=60,
    )

    metrics = evaluate_case_ranked_ids(case, ["c1", "c2", "c3", "c4"], k_values=(3,))
    assert metrics["has_good_labels"] == 0.0
    assert metrics["all_bad_case"] == 0.0
    assert metrics["ndcg@3"] > 0.0
    assert metrics["good_rate@3"] == 0.0
    assert metrics["unclear_rate@3"] == 2.0 / 3.0
    assert metrics["bad_rate@3"] == 1.0 / 3.0
    assert metrics["mrr@3"] is None
    assert metrics["recall@3"] is None


def test_evaluate_case_ranked_ids_marks_all_bad_cases():
    case = BenchmarkCase(
        case_id="engagement::q3",
        objective="engagement",
        query=_case().query,
        candidates=[
            _candidate("c1", rank=1, label=LABEL_BAD),
            _candidate("c2", rank=2, label=LABEL_BAD),
            _candidate("c3", rank=3, label=LABEL_BAD),
        ],
        retrieve_k=12,
        label_pool_size=3,
        source_candidate_pool_size=40,
    )

    metrics = evaluate_case_ranked_ids(case, ["c1", "c2", "c3"], k_values=(3,))
    assert metrics["has_good_labels"] == 0.0
    assert metrics["all_bad_case"] == 1.0
    assert metrics["ndcg@3"] == 0.0
    assert metrics["good_rate@3"] == 0.0
    assert metrics["unclear_rate@3"] == 0.0
    assert metrics["bad_rate@3"] == 1.0
    assert metrics["mrr@3"] is None
    assert metrics["recall@3"] is None


def test_dataset_roundtrip_and_summary(tmp_path: Path):
    dataset = BenchmarkDataset(
        version="recommender.human_comparable_benchmark.v1",
        generated_at="2026-04-05T12:00:00Z",
        bundle_dir=str(tmp_path / "bundle"),
        sample_metadata={"query_count_built": 1},
        rubric={"version": "rubric.v1"},
        cases=[_case()],
    )
    output = tmp_path / "benchmark.json"
    save_benchmark_dataset(dataset, output)
    loaded = load_benchmark_dataset(output)

    summary = summarize_benchmark_dataset(loaded)
    assert summary["case_count"] == 1
    assert summary["candidate_count"] == 4
    assert summary["label_counts"][LABEL_GOOD] == 1
    assert summary["label_counts"][LABEL_BAD] == 1
    assert summary["label_counts"]["unlabeled"] == 1
    assert summary["pairwise_preference_count"] == 1

    aggregate = aggregate_case_metrics(
        [
            {"ndcg@5": 1.0, "mrr@5": 1.0, "bad_rate@5": 0.0},
            {"ndcg@5": 0.5, "mrr@5": None, "bad_rate@5": 0.4},
        ]
    )
    assert aggregate == {"bad_rate@5": 0.2, "mrr@5": 1.0, "ndcg@5": 0.75}
