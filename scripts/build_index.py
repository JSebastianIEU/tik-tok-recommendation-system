from pathlib import Path
import json

from src.retrieval.index import RetrievalIndex
from src.common.schemas import TikTokPost
from src.common.constants import ROOT

def main() -> int:
    data_path = ROOT / "data" / "mock" / "tiktok_posts_mock.jsonl"
    idx_path = ROOT / "data" / "mock" / "index_artifact.txt"
    posts = [TikTokPost.model_validate_json(line) for line in data_path.read_text().splitlines() if line.strip()]
    index = RetrievalIndex()
    index.build(posts)
    index.save(idx_path)
    print(f"Saved placeholder index to {idx_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

# TODO: persist real vector store artifacts and support incremental updates.
