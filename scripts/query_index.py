import sys
from pathlib import Path

from src.common.schemas import TikTokPost
from src.common.constants import ROOT
from src.retrieval.index import RetrievalIndex
from src.retrieval.search import simple_search

def main() -> int:
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "example query"
    data_path = ROOT / "data" / "mock" / "tiktok_posts_mock.jsonl"
    posts = [TikTokPost.model_validate_json(line) for line in data_path.read_text().splitlines() if line.strip()]
    index = RetrievalIndex()
    index.build(posts)
    results = simple_search(index, query=query, topk=3)
    for post, score in results:
        print(f"{post.video_id} | score={score:.4f} | caption={post.caption}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

# TODO: add CLI flags for top-k, filters, and backend selection.
