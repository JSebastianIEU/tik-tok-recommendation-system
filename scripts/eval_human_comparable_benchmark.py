#!/usr/bin/env python3
"""Evaluate a recommender bundle against a human-labeled comparable benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.recommendation.learning import (
    aggregate_case_metrics,
    evaluate_case_ranked_ids,
    load_benchmark_dataset,
    RecommenderRuntime,
)


def _parse_k_values(value: str) -> List[int]:
    out: List[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(max(1, int(item)))
    if not out:
        raise ValueError("At least one k value is required.")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a recommender bundle against a human-labeled comparable benchmark."
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument(
        "--k-values",
        type=str,
        default="5,10",
        help="Comma-separated cutoff values for ndcg/mrr/recall.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default=None,
        help="Optional objective filter for benchmark cases.",
    )
    args = parser.parse_args(argv)

    dataset = load_benchmark_dataset(args.benchmark_json)
    runtime = RecommenderRuntime(bundle_dir=args.bundle_dir)
    k_values = _parse_k_values(args.k_values)

    case_reports: List[Dict[str, Any]] = []
    bundle_metric_rows: List[Dict[str, float]] = []
    baseline_metric_rows: List[Dict[str, float]] = []
    objective_bundle_rows: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    objective_baseline_rows: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    skipped = Counter()
    case_type_counts = Counter()

    for case in dataset.cases:
        if args.objective and case.objective != args.objective:
            skipped["objective_filter"] += 1
            continue
        if not case.relevance_map():
            skipped["no_labels"] += 1
            continue
        if case.relevant_ids():
            case_type_counts["with_good_labels"] += 1
        else:
            case_type_counts["without_good_labels"] += 1
        label_summary = case.label_summary()
        if label_summary.get("bad", 0) > 0 and label_summary.get("good", 0) == 0 and label_summary.get("unclear", 0) == 0:
            case_type_counts["all_bad_cases"] += 1

        candidate_pool = [candidate.candidate_payload for candidate in case.candidates]
        query_payload = dict(case.query.query_payload)
        as_of_time = query_payload.get("as_of_time") or case.query.display.get("created_at")
        if not as_of_time:
            skipped["missing_as_of_time"] += 1
            continue

        response = runtime.recommend(
            objective=case.objective,
            as_of_time=as_of_time,
            query=query_payload,
            candidates=candidate_pool,
            top_k=len(candidate_pool),
            retrieve_k=len(candidate_pool),
            explainability={"enabled": False},
            debug=False,
        )
        ranked_ids = [str(item.get("candidate_id") or "") for item in list(response.get("items") or [])]
        baseline_ranked_ids = case.baseline_ranked_ids()

        bundle_metrics = evaluate_case_ranked_ids(case, ranked_ids, k_values=k_values)
        baseline_metrics = evaluate_case_ranked_ids(case, baseline_ranked_ids, k_values=k_values)

        bundle_metric_rows.append(bundle_metrics)
        baseline_metric_rows.append(baseline_metrics)
        objective_bundle_rows[case.objective].append(bundle_metrics)
        objective_baseline_rows[case.objective].append(baseline_metrics)

        case_reports.append(
            {
                "case_id": case.case_id,
                "objective": case.objective,
                "label_summary": label_summary,
                "bundle_metrics": bundle_metrics,
                "baseline_snapshot_metrics": baseline_metrics,
                "bundle_ranked_ids": ranked_ids,
                "baseline_ranked_ids": baseline_ranked_ids,
                "learned_reranker_metadata": dict(
                    response.get("learned_reranker_metadata") or {}
                ),
            }
        )

    objective_reports: Dict[str, Any] = {}
    for objective, rows in objective_bundle_rows.items():
        objective_reports[objective] = {
            "bundle_metrics": aggregate_case_metrics(rows),
            "baseline_snapshot_metrics": aggregate_case_metrics(
                objective_baseline_rows.get(objective, [])
            ),
            "case_count": len(rows),
        }

    report = {
        "benchmark_json": str(args.benchmark_json.resolve()),
        "bundle_dir": str(args.bundle_dir.resolve()),
        "evaluated_case_count": len(case_reports),
        "skipped_case_counts": dict(skipped),
        "case_type_counts": dict(case_type_counts),
        "bundle_metrics": aggregate_case_metrics(bundle_metric_rows),
        "baseline_snapshot_metrics": aggregate_case_metrics(baseline_metric_rows),
        "objective_metrics": objective_reports,
        "cases": case_reports,
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
