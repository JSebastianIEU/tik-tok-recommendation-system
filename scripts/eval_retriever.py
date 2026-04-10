#!/usr/bin/env python3
"""Evaluate retriever.v2 only (Recall@K) per objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.recommendation.learning.objectives import map_objective
from src.recommendation.learning.pipeline import (
    _evaluate_retriever_objective,
    _group_relevance_by_query,
    _pair_rows_for_objective,
    _rows_by_id,
)
from src.recommendation.learning.retriever import HybridRetriever
from src.recommendation.learning.temporal import split_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retriever.v2 on datamart rows/pairs.")
    parser.add_argument("datamart_json", type=Path, help="Path to datamart JSON.")
    parser.add_argument(
        "--retriever-dir",
        type=Path,
        default=Path("artifacts/retriever/latest"),
        help="Retriever artifact directory.",
    )
    parser.add_argument(
        "--objectives",
        type=str,
        default="reach,engagement,conversion",
        help="Comma-separated objective list.",
    )
    parser.add_argument("--retrieve-k", type=int, default=200)
    parser.add_argument("--max-age-days", type=int, default=180)
    parser.add_argument(
        "--pair-target-source",
        type=str,
        default="scalar_v1",
        choices=["scalar_v1", "trajectory_v2_composite"],
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON file path.",
    )
    args = parser.parse_args()

    datamart = json.loads(args.datamart_json.read_text(encoding="utf-8"))
    rows = list(datamart.get("rows") or [])
    pair_rows = list(datamart.get("pair_rows") or [])
    split = split_rows(rows)
    retriever = HybridRetriever.load(args.retriever_dir)
    rows_by_id = _rows_by_id(rows)

    output: Dict[str, Any] = {"objectives": {}}
    objectives = [item.strip() for item in args.objectives.split(",") if item.strip()]
    for objective in objectives:
        _, effective = map_objective(objective)
        objective_pairs = _pair_rows_for_objective(
            rows_by_id=rows_by_id,
            pair_rows=pair_rows,
            objective=effective,
            pair_target_source=args.pair_target_source,
        )
        relevance = _group_relevance_by_query(
            pair_rows=objective_pairs,
            objective=effective,
        )
        output["objectives"][effective] = _evaluate_retriever_objective(
            retriever=retriever,
            objective=effective,
            rows_split=split,
            relevance_by_query=relevance,
            retrieve_k=max(1, int(args.retrieve_k)),
            max_age_days=max(1, int(args.max_age_days)),
        )

    serialized = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
