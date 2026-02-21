#!/usr/bin/env python3
"""
Build retrieval index from mock TikTok data.

Usage:
    python scripts/build_index.py

This script:
1. Loads posts from data/mock/tiktok_posts_mock.jsonl
2. Builds a TF-IDF index on combined text fields
3. Saves the index to data/mock/retrieval_index.pkl

TODO: Add CLI arguments for:
- Custom input/output paths
- Index configuration (ngram range, max features)
- Progress tracking for large datasets
TODO: Support incremental index updates (add new posts without full rebuild)
TODO: Add index quality metrics (vocabulary size, sparsity, etc.)
"""

from pathlib import Path

from src.retrieval.index import RetrievalIndex
from src.common.schemas import TikTokPost
from src.common.constants import ROOT

def main() -> int:
    # Paths
    data_path = ROOT / "data" / "mock" / "tiktok_posts_mock.jsonl"
    index_path = ROOT / "data" / "mock" / "retrieval_index.pkl"

    print(f"Loading posts from {data_path}...")

    # Load posts from JSONL file
    posts = []
    for line in data_path.read_text().splitlines():
        if line.strip():
            posts.append(TikTokPost.model_validate_json(line))

    print(f"Loaded {len(posts)} posts")

    # Build index
    print("Building TF-IDF index...")
    index = RetrievalIndex()
    index.build(posts)

    # Save index
    print(f"Saving index to {index_path}...")
    index.save(index_path)

    print("\n✓ Index built successfully!")
    print(f"  Posts indexed: {len(posts)}")
    print(f"  Index location: {index_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
