from typing import List, Dict, Any
import numpy as np

from .index import RetrievalIndex

def search(
    index: RetrievalIndex,
    query: str,
    topk: int = 10
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
    vectorizer = index.get_vectorizer()
    matrix = index.get_matrix()

    if not posts or vectorizer is None or matrix is None:
        return []

    # Transform query using the pre-built vectorizer
    query_vec = vectorizer.transform([query])

    # Compute cosine similarity scores (TF-IDF dot product)
    scores = (matrix @ query_vec.T).toarray().ravel()

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:topk]

    # Format results
    results = []
    for idx in top_indices:
        post = posts[idx]
        results.append({
            "video_id": post.video_id,
            "video_url": str(post.video_url),  # Convert HttpUrl to string for JSON serialization
            "score": float(scores[idx]),
            "caption": post.caption,
            "hashtags": post.hashtags
        })

    return results


def filtered_search(
    index: RetrievalIndex,
    query: str,
    topk: int = 10,
    language: str = None,
    min_likes: int = None
) -> List[Dict[str, Any]]:
    """
    Search with additional filtering options.

    TODO: Implement pre-filtering or post-filtering strategies
    TODO: Add date range filtering
    TODO: Add author filtering
    TODO: Support multi-lingual ranking adjustments
    """
    # Get initial results
    results = search(index, query, topk=topk * 2)  # Over-fetch for filtering

    # Apply filters
    filtered = []
    for result in results:
        # Find the original post for additional filtering
        post = next((p for p in index.get_posts() if p.video_id == result["video_id"]), None)
        if post is None:
            continue

        # Language filter
        if language and post.video_meta.language != language:
            continue

        # Likes filter
        if min_likes and post.likes < min_likes:
            continue

        filtered.append(result)

        # Stop when we have enough results
        if len(filtered) >= topk:
            break

    return filtered


# TODO: Add ranking explanation feature (show which terms matched)
# TODO: Add query expansion (synonyms, related terms)
# TODO: Support personalized ranking (user history, preferences)
