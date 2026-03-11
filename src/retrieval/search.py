from typing import Any, Dict, List, Optional
import numpy as np

from .index import RetrievalIndex


def _normalize_topk(topk: int) -> int:
    return max(0, int(topk))


def _query_scores(index: RetrievalIndex, query: str) -> np.ndarray:
    posts = index.get_posts()
    vectorizer = index.get_vectorizer()
    matrix = index.get_matrix()
    if not posts or vectorizer is None or matrix is None:
        return np.array([], dtype=float)

    query_text = (query or "").strip()
    if not query_text:
        return np.array([], dtype=float)

    query_vec = vectorizer.transform([query_text])
    return (matrix @ query_vec.T).toarray().ravel()


def _format_result(post, score: float) -> Dict[str, Any]:
    return {
        "video_id": post.video_id,
        "video_url": str(post.video_url),  # Convert HttpUrl to string for JSON serialization
        "score": float(score),
        "caption": post.caption,
        "hashtags": post.hashtags,
        "likes": int(post.likes),
        "language": post.video_meta.language,
    }


def search(
    index: RetrievalIndex,
    query: str,
    topk: int = 10,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Baseline TF-IDF cosine similarity search.

    Args:
        index: Pre-built retrieval index
        query: Search query text
        topk: Number of results to return

    Returns:
        List of dicts with keys: video_id, video_url, score, caption, hashtags

    TODO: Replace with configurable ranking backend:
    - BM25 (rank-bm25 library for better term frequency handling)
    - SBERT (sentence-transformers for semantic similarity)
    - FAISS (facebook/faiss for efficient dense vector search)
    - Hybrid (combine lexical + semantic signals)
    """
    posts = index.get_posts()
    k = _normalize_topk(topk)
    if k == 0:
        return []

    scores = _query_scores(index, query)
    if scores.size == 0:
        return []

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1]

    # Format results
    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= float(min_score):
            continue
        results.append(_format_result(posts[idx], score))
        if len(results) >= k:
            break

    return results


def filtered_search(
    index: RetrievalIndex,
    query: str,
    topk: int = 10,
    language: Optional[str] = None,
    min_likes: Optional[int] = None,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Search with additional filtering options.

    TODO: Implement pre-filtering or post-filtering strategies
    TODO: Add date range filtering
    TODO: Add author filtering
    TODO: Support multi-lingual ranking adjustments
    """
    k = _normalize_topk(topk)
    if k == 0:
        return []

    # Over-fetch to compensate for filters without scanning all scores repeatedly.
    pool_size = max(k * 5, 50)
    results = search(index, query, topk=pool_size, min_score=min_score)
    if not results:
        return []

    # Apply filters
    filtered = []
    for result in results:
        # O(1) post lookup avoids O(n^2) behavior for larger candidate pools.
        post = index.get_post(result["video_id"])
        if post is None:
            continue

        # Language filter
        if language and post.video_meta.language.lower() != language.lower():
            continue

        # Likes filter
        if min_likes is not None and post.likes < min_likes:
            continue

        filtered.append(result)

        # Stop when we have enough results
        if len(filtered) >= k:
            break

    return filtered


# TODO: Add ranking explanation feature (show which terms matched)
# TODO: Add query expansion (synonyms, related terms)
# TODO: Support personalized ranking (user history, preferences)
