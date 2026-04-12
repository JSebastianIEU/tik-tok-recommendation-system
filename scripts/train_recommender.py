#!/usr/bin/env python3
"""Train recommender-learning-v1 artifacts from datamart.v1 JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._utils import to_jsonable
from src.recommendation.learning import RecommenderTrainingConfig, train_recommender_from_datamart


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
    parser.add_argument(
        "--pair-target-source",
        type=str,
        default="scalar_v1",
        choices=["scalar_v1", "trajectory_v2_composite"],
        help="Which pair target source to use during ranker training.",
    )
    parser.add_argument(
        "--feature-snapshot-manifest-path",
        type=str,
        default=None,
        help="Optional feature fabric snapshot manifest path/dir for multimodal retriever vectors.",
    )
    parser.add_argument(
        "--graph-enabled",
        dest="graph_enabled",
        action="store_true",
        help="Enable Creator DNA x Video DNA graph branch and graph-derived features.",
    )
    parser.add_argument(
        "--no-graph",
        dest="graph_enabled",
        action="store_false",
        help="Disable graph branch and graph-derived features.",
    )
    parser.set_defaults(graph_enabled=True)
    parser.add_argument(
        "--graph-embedding-dim",
        type=int,
        default=32,
        help="Embedding dimension for graph node embeddings.",
    )
    parser.add_argument(
        "--graph-walk-length",
        type=int,
        default=12,
        help="Graph random walk length.",
    )
    parser.add_argument(
        "--graph-num-walks",
        type=int,
        default=20,
        help="Graph random walks per node.",
    )
    parser.add_argument(
        "--graph-context-size",
        type=int,
        default=4,
        help="Context window for node2vec-like co-occurrence.",
    )
    parser.add_argument(
        "--graph-branch-weight",
        type=float,
        default=0.10,
        help="Default graph branch weight before objective-specific blend fitting.",
    )
    parser.add_argument(
        "--trajectory-enabled",
        dest="trajectory_enabled",
        action="store_true",
        help="Enable trajectory branch and trajectory-derived features.",
    )
    parser.add_argument(
        "--no-trajectory",
        dest="trajectory_enabled",
        action="store_false",
        help="Disable trajectory branch and trajectory-derived features.",
    )
    parser.set_defaults(trajectory_enabled=True)
    parser.add_argument(
        "--trajectory-embedding-dim",
        type=int,
        default=16,
        help="Embedding dimension for trajectory vectors.",
    )
    parser.add_argument(
        "--trajectory-feature-version",
        type=str,
        default="trajectory_features.v2",
        help="Trajectory feature schema/version identifier.",
    )
    parser.add_argument(
        "--trajectory-branch-weight",
        type=float,
        default=0.08,
        help="Default trajectory branch weight before objective-specific blend fitting.",
    )
    parser.add_argument(
        "--trajectory-encoder-mode",
        type=str,
        default="feature_only",
        choices=["feature_only", "sequence_encoder_shadow"],
        help="Trajectory encoder mode.",
    )
    parser.add_argument(
        "--trajectory-manifest-path",
        type=str,
        default=None,
        help="Optional trajectory artifact manifest path/dir.",
    )
    parser.add_argument(
        "--blend-grid-levels",
        type=int,
        default=3,
        help=(
            "Points per axis in the retriever branch-weight grid search (2–11). "
            "Lower is faster (default 3 => [0, 0.5, 1.0], 15 weight tuples). "
            "Legacy full search used 5 levels (70 tuples) and could run for hours on large datamarts."
        ),
    )
    parser.add_argument(
        "--blend-max-eval-queries",
        type=int,
        default=128,
        help=(
            "Max labeled validation queries used when fitting per-objective retriever blend weights. "
            "Caps retrieve() calls during grid search."
        ),
    )
    args = parser.parse_args()

    with open(args.datamart_json, "r", encoding="utf-8") as f:
        datamart = json.load(f)
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
            pair_target_source=args.pair_target_source,
            feature_snapshot_manifest_path=args.feature_snapshot_manifest_path,
            graph_enabled=bool(args.graph_enabled),
            graph_embedding_dim=max(4, int(args.graph_embedding_dim)),
            graph_walk_params={
                "walk_length": max(4, int(args.graph_walk_length)),
                "num_walks": max(2, int(args.graph_num_walks)),
                "context_size": max(1, int(args.graph_context_size)),
                "seed": 13,
            },
            graph_weighting_params={
                "recency_half_life_days": 45.0,
                "include_creator_similarity": True,
                "creator_similarity_top_k": 5,
                "creator_similarity_min_jaccard": 0.15,
                "branch_weight": max(0.0, float(args.graph_branch_weight)),
            },
            trajectory_enabled=bool(args.trajectory_enabled),
            trajectory_embedding_dim=max(4, int(args.trajectory_embedding_dim)),
            trajectory_feature_version=str(args.trajectory_feature_version),
            trajectory_branch_weight=max(0.0, float(args.trajectory_branch_weight)),
            trajectory_encoder_mode=str(args.trajectory_encoder_mode),
            trajectory_manifest_path=args.trajectory_manifest_path,
            contract_version=str(datamart.get("source_contract_version", "contract.v2")),
            datamart_version=str(datamart.get("version", "datamart.v1")),
            blend_grid_levels=max(2, min(11, int(args.blend_grid_levels))),
            blend_search_max_eval_queries=max(8, int(args.blend_max_eval_queries)),
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

    print(json.dumps(to_jsonable(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
