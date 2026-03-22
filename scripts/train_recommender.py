#!/usr/bin/env python3
"""Train recommender-learning-v1 artifacts from datamart.v1 JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.recommendation.learning import (
    RecommenderTrainingConfig,
    train_recommender_from_datamart,
)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(inner) for inner in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Train recommender learning artifacts.")
    parser.add_argument(
        "datamart_json",
        type=Path,
        help="Path to training datamart JSON.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/recommender"),
        help="Root directory where recommender bundles are written.",
    )
    parser.add_argument(
        "--objectives",
        type=str,
        default="reach,engagement,conversion",
        help="Comma-separated objectives to train.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="recommender-v1",
        help="Run name suffix for artifact bundle folder.",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=180,
    )
    parser.add_argument(
        "--dense-model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    args = parser.parse_args()

    datamart = json.loads(args.datamart_json.read_text(encoding="utf-8"))
    objectives = [item.strip() for item in args.objectives.split(",") if item.strip()]
    result = train_recommender_from_datamart(
        datamart=datamart,
        artifact_root=args.artifact_root,
        config=RecommenderTrainingConfig(
            objectives=objectives,
            retrieve_k=max(1, args.retrieve_k),
            max_age_days=max(1, args.max_age_days),
            dense_model_name=args.dense_model_name,
            run_name=args.run_name,
            contract_version=str(datamart.get("source_contract_version", "contract.v1")),
            datamart_version=str(datamart.get("version", "datamart.v1")),
        ),
    )
    bundle_dir = Path(result["bundle_dir"])
    latest_link = args.artifact_root / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        if latest_link.is_symlink() or latest_link.is_file():
            latest_link.unlink()
        else:
            # Keep non-symlink directory untouched to avoid destructive behavior.
            pass
    if not latest_link.exists():
        try:
            latest_link.symlink_to(bundle_dir.resolve())
        except OSError:
            latest_link.write_text(str(bundle_dir.resolve()), encoding="utf-8")

    print(json.dumps(_to_jsonable(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
