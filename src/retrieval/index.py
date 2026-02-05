from pathlib import Path
from typing import List

from ..common.schemas import TikTokPost

class RetrievalIndex:
    def __init__(self) -> None:
        self._posts: List[TikTokPost] = []
        # TODO: hold TF-IDF vectors or embedding placeholders.

    def build(self, posts: List[TikTokPost]) -> None:
        """Store posts and prepare lightweight index."""
        self._posts = posts
        # TODO: compute feature matrix (e.g., captions+hashtags TF-IDF).

    def save(self, path: Path) -> None:
        """Persist index artifacts."""
        # TODO: serialize matrix/vocabulary; keep simple JSON for now.
        path.write_text("# TODO: persist index artifacts\n")

    # TODO: add load() to restore saved index.
