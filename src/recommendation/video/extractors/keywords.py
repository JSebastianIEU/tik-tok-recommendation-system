"""YAKE keyword extraction — fast, statistical, no model needed.

Designed for short social media text. Outperforms TF-IDF on captions.
"""

from __future__ import annotations

from typing import List

from ..models import KeywordResult


def extract_keywords(
    text: str,
    max_keywords: int = 20,
    language: str = "en",
    n_gram_size: int = 2,
    dedup_threshold: float = 0.7,
) -> List[KeywordResult]:
    """Extract keywords from text using YAKE."""
    if not text or len(text.strip()) < 5:
        return []

    import yake

    extractor = yake.KeywordExtractor(
        lan=language,
        n=n_gram_size,
        dedupLim=dedup_threshold,
        top=max_keywords,
        features=None,
    )

    raw = extractor.extract_keywords(text)
    return [
        KeywordResult(keyword=kw.lower().strip(), score=round(score, 6))
        for kw, score in raw
        if kw.strip()
    ]
