#!/usr/bin/env python3
"""Train the hashtag recommender from exported Supabase JSONL data.

Usage:
    python scripts/train_hashtag_recommender.py [--input PATH] [--output PATH] [--k K]

Defaults:
    --input  data/real/tiktok_posts_real.jsonl
    --output artifacts/hashtag_recommender
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommendation.hashtag_recommender import HashtagRecommender


def main() -> int:
    parser = argparse.ArgumentParser(description="Train hashtag recommender")
    parser.add_argument("--input", default="data/real/tiktok_posts_real.jsonl", help="Input JSONL")
    parser.add_argument("--output", default="artifacts/hashtag_recommender", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run export_supabase_to_jsonl.py first.")
        return 1

    print(f"Loading records from {input_path}...")
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records")

    recommender = HashtagRecommender.train(records, batch_size=args.batch_size)

    output_path = Path(args.output)
    recommender.save(output_path)

    # Quick test
    print("\n--- Quick test ---")
    test_captions = [
        "How to grow your audience on social media with these simple tips",
        "Best recipe for homemade pasta from scratch",
        "Morning workout routine for beginners at home",
    ]
    for caption in test_captions:
        results = recommender.recommend(caption, k=10, top_n=5)
        print(f"\nCaption: {caption[:60]}...")
        for r in results:
            print(f"  {r['hashtag']:20s}  freq={r['frequency']}  eng={r['avg_engagement']:.2f}  score={r['score']:.4f}")

    print(f"\nArtifacts saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
