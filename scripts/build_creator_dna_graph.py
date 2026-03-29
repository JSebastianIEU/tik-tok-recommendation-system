#!/usr/bin/env python3
"""Build Creator DNA x Video DNA graph artifacts from datamart rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.recommendation.learning.graph import (
    GraphBuildConfig,
    build_creator_video_dna_graph,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build creator-video DNA graph artifacts.")
    parser.add_argument("datamart_json", type=Path, help="Path to datamart JSON.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/recommender/graph"),
        help="Output directory for graph artifact bundle.",
    )
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--walk-length", type=int, default=12)
    parser.add_argument("--num-walks", type=int, default=20)
    parser.add_argument("--context-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    datamart = json.loads(args.datamart_json.read_text(encoding="utf-8"))
    rows = list(datamart.get("rows") or [])
    graph = build_creator_video_dna_graph(
        rows=rows,
        as_of_time=datamart.get("generated_at"),
        run_cutoff_time=datamart.get("generated_at"),
        config=GraphBuildConfig(
            embedding_dim=max(4, int(args.embedding_dim)),
            walk_length=max(4, int(args.walk_length)),
            num_walks=max(2, int(args.num_walks)),
            context_size=max(1, int(args.context_size)),
            seed=int(args.seed),
        ),
    )
    manifest_path = graph.save(args.output_dir)
    print(
        json.dumps(
            {
                "graph_manifest_path": str(manifest_path),
                "graph_bundle_id": graph.graph_bundle_id,
                "graph_schema_hash": graph.graph_schema_hash,
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

