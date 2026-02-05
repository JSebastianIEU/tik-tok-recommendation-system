from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer  # placeholder import
import numpy as np

from .index import RetrievalIndex
from ..common.schemas import TikTokPost

def simple_search(index: RetrievalIndex, query: str, topk: int = 3) -> List[Tuple[TikTokPost, float]]:
    """
    Placeholder search using TF-IDF on captions + hashtags + keywords.
    TODO: replace with configurable backend (TF-IDF, BM25, SBERT, FAISS).
    """
    if not index._posts:
        return []
    corpus = [" ".join([p.caption, " ".join(p.hashtags), " ".join(p.keywords)]) for p in index._posts]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    q_vec = vectorizer.transform([query])
    scores = (matrix @ q_vec.T).toarray().ravel()
    top_indices = np.argsort(scores)[::-1][:topk]
    return [(index._posts[i], float(scores[i])) for i in top_indices]

# TODO: expose ranking explanations and support filtering by language or date.
