#!/usr/bin/env python3
"""
Query the retrieval index with a text search.

Usage:
    python scripts/query_index.py "your search query"
    python scripts/query_index.py "core workout" --topk 5
    python scripts/query_index.py "ramen hack" --language en

Examples:
    python scripts/query_index.py "core workout"
    python scripts/query_index.py "easy recipes"
    python scripts/query_index.py "travel tips japan"

TODO: Add CLI flags for:
- --topk: number of results (default: 10)
- --language: filter by language
- --min-likes: minimum likes threshold
- --format: output format (table, json, csv)
- --explain: show ranking explanations
TODO: Support query from file or stdin
TODO: Add interactive query mode
"""

import sys
import argparse
import json

from src.common.constants import ROOT
from src.retrieval.index import RetrievalIndex
from src.retrieval.search import search

def main() -> int:
    parser = argparse.ArgumentParser(description="Query the TikTok retrieval index")
    parser.add_argument("query", nargs="+", help="Search query text")
    parser.add_argument("--topk", type=int, default=10, help="Number of results to return")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()
    query_text = " ".join(args.query)

    # Load the pre-built index
    index_path = ROOT / "data" / "mock" / "retrieval_index.pkl"

    if not index_path.exists():
        print(f"Error: Index not found at {index_path}")
        print("Please run 'python scripts/build_index.py' first")
        return 1

    print(f"Loading index from {index_path}...")
    index = RetrievalIndex.load(index_path)

    # Perform search
    print(f"\nQuery: '{query_text}'")
    print(f"Top {args.topk} results:\n")

    results = search(index, query=query_text, topk=args.topk)

    if args.json:
        # Output as JSON
        print(json.dumps(results, indent=2))
    else:
        # Output as formatted text
        if not results:
            print("No results found.")
        else:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['video_id']} (score: {result['score']:.4f})")
                print(f"   Caption: {result['caption']}")
                print(f"   Hashtags: {', '.join(result['hashtags'])}")
                print(f"   URL: {result['video_url']}")
                print()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
