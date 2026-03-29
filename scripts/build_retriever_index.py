#!/usr/bin/env python3
"""Build retriever.v2 index artifacts from a datamart JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from src.recommendation.learning.pipeline import (
    HybridRetrieverTrainer,
    HybridRetrieverTrainerConfig,
    _load_feature_snapshot_vectors,
)
from src.recommendation.learning.graph import GraphBuildConfig, build_creator_video_dna_graph
from src.recommendation.learning.trajectory import (
    TrajectoryBuildConfig,
    TrajectoryBundle,
    build_trajectory_bundle,
)
from src.recommendation.learning.temporal import split_rows


def _build_multimodal_row_vectors(
    rows: Sequence[Dict[str, Any]],
    by_video: Dict[str, Sequence[float]],
) -> Dict[str, Sequence[float]]:
    out: Dict[str, Sequence[float]] = {}
    for row in rows:
        row_id = str(row.get("row_id") or "")
        video_id = str(row.get("video_id") or row_id.split("::", 1)[0])
        if not row_id or not video_id:
            continue
        vector = by_video.get(video_id)
        if vector is not None:
            out[row_id] = vector
    return out


def _resolve_manifest_path(manifest_ref: str) -> Path:
    path = Path(manifest_ref)
    if path.is_dir():
        return path / "manifest.json"
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build retriever.v2 index artifacts.")
    parser.add_argument("datamart_json", type=Path, help="Path to datamart JSON.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/retriever/latest"),
        help="Retriever artifact output directory.",
    )
    parser.add_argument(
        "--dense-model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--feature-snapshot-manifest-path",
        type=str,
        default=None,
        help="Optional feature snapshot manifest path/dir for multimodal vectors.",
    )
    parser.add_argument(
        "--graph-enabled",
        dest="graph_enabled",
        action="store_true",
        help="Enable Creator DNA graph branch vectors.",
    )
    parser.add_argument(
        "--no-graph",
        dest="graph_enabled",
        action="store_false",
        help="Disable Creator DNA graph branch vectors.",
    )
    parser.set_defaults(graph_enabled=True)
    parser.add_argument(
        "--trajectory-enabled",
        dest="trajectory_enabled",
        action="store_true",
        help="Enable trajectory retrieval branch vectors.",
    )
    parser.add_argument(
        "--no-trajectory",
        dest="trajectory_enabled",
        action="store_false",
        help="Disable trajectory retrieval branch vectors.",
    )
    parser.set_defaults(trajectory_enabled=True)
    parser.add_argument(
        "--trajectory-manifest-path",
        type=str,
        default=None,
        help="Optional trajectory artifact manifest path/dir.",
    )
    parser.add_argument(
        "--trajectory-embedding-dim",
        type=int,
        default=16,
        help="Embedding dimension for trajectory vectors.",
    )
    parser.add_argument(
        "--trajectory-branch-weight",
        type=float,
        default=0.08,
        help="Default trajectory branch weight before learned objective fusion.",
    )
    args = parser.parse_args()

    datamart = json.loads(args.datamart_json.read_text(encoding="utf-8"))
    rows = list(datamart.get("rows") or [])
    split = split_rows(rows)
    if not split["train"]:
        raise ValueError("Datamart must include train rows for retriever build.")

    _, feature_vectors = _load_feature_snapshot_vectors(args.feature_snapshot_manifest_path)
    multimodal_vectors = _build_multimodal_row_vectors(split["train"], feature_vectors)
    graph_vectors = {}
    graph_lookup = None
    graph_metadata = None
    if args.graph_enabled:
        graph = build_creator_video_dna_graph(
            rows=split["train"],
            as_of_time=datamart.get("generated_at"),
            run_cutoff_time=datamart.get("generated_at"),
            config=GraphBuildConfig(embedding_dim=32, seed=13),
        )
        graph_vectors = _build_multimodal_row_vectors(split["train"], graph.video_embeddings)
        graph_lookup = {
            "video": graph.video_embeddings,
            "creator": graph.creator_embeddings,
            "hashtag": graph.hashtag_embeddings,
            "audio_motif": graph.audio_embeddings,
            "style_signature": graph.style_embeddings,
        }
        graph_metadata = {
            "graph_bundle_id": graph.graph_bundle_id,
            "graph_version": graph.version,
            "graph_schema_hash": graph.graph_schema_hash,
        }
    trajectory_vectors = {}
    trajectory_lookup = None
    trajectory_metadata = None
    if args.trajectory_enabled:
        trajectory_bundle: TrajectoryBundle
        if args.trajectory_manifest_path:
            trajectory_manifest = _resolve_manifest_path(args.trajectory_manifest_path)
            trajectory_bundle = TrajectoryBundle.load(trajectory_manifest.parent)
        else:
            trajectory_bundle = build_trajectory_bundle(
                rows=split["train"],
                as_of_time=datamart.get("generated_at"),
                run_cutoff_time=datamart.get("generated_at"),
                config=TrajectoryBuildConfig(
                    embedding_dim=max(4, int(args.trajectory_embedding_dim))
                ),
            )
        trajectory_vectors = _build_multimodal_row_vectors(
            split["train"], trajectory_bundle.embeddings_by_video
        )
        trajectory_lookup = {"video": trajectory_bundle.embeddings_by_video}
        trajectory_metadata = {
            "trajectory_manifest_id": trajectory_bundle.trajectory_manifest_id,
            "trajectory_version": trajectory_bundle.version,
            "trajectory_schema_hash": trajectory_bundle.trajectory_schema_hash,
        }

    trainer = HybridRetrieverTrainer(
        HybridRetrieverTrainerConfig(
            dense_model_name=args.dense_model_name,
            trajectory_weight=max(0.0, float(args.trajectory_branch_weight)),
        )
    )
    retriever = trainer.train(
        split["train"],
        multimodal_vectors=multimodal_vectors or None,
        graph_vectors=graph_vectors or None,
        graph_lookup=graph_lookup,
        graph_metadata=graph_metadata,
        trajectory_vectors=trajectory_vectors or None,
        trajectory_lookup=trajectory_lookup,
        trajectory_metadata=trajectory_metadata,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "artifact_version": retriever.artifact_version,
                "train_rows": len(split["train"]),
                "objective_blend": retriever.objective_blend,
                "graph_bundle_id": retriever.graph_bundle_id,
                "graph_version": retriever.graph_version,
                "trajectory_manifest_id": retriever.trajectory_manifest_id,
                "trajectory_version": retriever.trajectory_version,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
