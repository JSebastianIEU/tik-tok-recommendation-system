from pathlib import Path
from typing import Dict, List, Optional
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from ..common.schemas import TikTokPost

class RetrievalIndex:
    """
    Baseline retrieval index using TF-IDF on textual fields.

    TODO: Make this modular to support alternative ranking methods:
    - BM25 (via rank-bm25 library)
    - Semantic embeddings (SBERT/sentence-transformers)
    - Dense retrieval (FAISS with learned embeddings)
    """

    def __init__(self) -> None:
        self._posts: List[TikTokPost] = []
        self._posts_by_id: Dict[str, TikTokPost] = {}
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix: Optional[csr_matrix] = None

    def build(self, posts: List[TikTokPost]) -> None:
        """
        Store posts and build TF-IDF index on combined textual fields.

        Combines: caption + hashtags + keywords + search_query

        TODO: Add support for:
        - Custom tokenization/preprocessing
        - Language-specific stemmers
        - Stop word filtering per language
        """
        self._posts = posts
        self._posts_by_id = {post.video_id: post for post in posts}

        # Combine all textual fields into a single document per post
        corpus = []
        for post in posts:
            text_parts = [
                post.caption,
                " ".join(post.hashtags),
                " ".join(post.keywords),
                post.search_query
            ]
            combined_text = " ".join(text_parts)
            corpus.append(combined_text)

        # Build TF-IDF vectorizer and matrix
        # TODO: Experiment with ngram_range, max_features, min_df for optimization
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            token_pattern=r'\b\w+\b'
        )
        self._matrix = self._vectorizer.fit_transform(corpus)

        print(f"Built index with {len(posts)} posts and {self._matrix.shape[1]} features")

    def save(self, path: Path) -> None:
        """
        Persist index artifacts to disk.

        Saves:
        - Posts list
        - TF-IDF vectorizer (vocabulary + IDF weights)
        - Sparse feature matrix

        TODO: Add versioning for backward compatibility
        TODO: Support incremental updates without full rebuild
        """
        if not self._posts or self._vectorizer is None or self._matrix is None:
            raise ValueError("Index must be built before saving")

        index_data = {
            "posts": [post.model_dump() for post in self._posts],
            "vectorizer": self._vectorizer,
            "matrix": self._matrix
        }

        with open(path, 'wb') as f:
            pickle.dump(index_data, f)

        print(f"Saved index to {path}")

    @classmethod
    def load(cls, path: Path) -> "RetrievalIndex":
        """
        Load a previously saved index from disk.

        TODO: Add validation for corrupted files
        TODO: Support loading from different storage backends (S3, GCS)
        """
        with open(path, 'rb') as f:
            index_data = pickle.load(f)

        index = cls()
        index._posts = [TikTokPost.model_validate(post_dict) for post_dict in index_data["posts"]]
        index._posts_by_id = {post.video_id: post for post in index._posts}
        index._vectorizer = index_data["vectorizer"]
        index._matrix = index_data["matrix"]

        print(f"Loaded index with {len(index._posts)} posts")
        return index

    def get_posts(self) -> List[TikTokPost]:
        """Return all posts in the index."""
        return self._posts

    def get_post(self, video_id: str) -> Optional[TikTokPost]:
        """Return one post by video_id if present."""
        return self._posts_by_id.get(video_id)

    def get_matrix(self) -> Optional[csr_matrix]:
        """Return the TF-IDF feature matrix."""
        return self._matrix

    def get_vectorizer(self) -> Optional[TfidfVectorizer]:
        """Return the TF-IDF vectorizer."""
        return self._vectorizer
