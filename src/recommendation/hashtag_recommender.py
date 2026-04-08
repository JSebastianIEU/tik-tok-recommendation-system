"""Hashtag recommendation via semantic neighborhood aggregation.

Given a caption, finds the K most semantically similar videos in the corpus
(via SBERT embeddings + FAISS), aggregates their hashtags by frequency and
average engagement, and returns the top-N suggestions.

Based on the NLP project findings: semantically similar captions exhibit
10-13% lower within-group variance in engagement, making neighborhood-based
hashtag recommendations empirically grounded.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def _remove_mentions(text: str) -> str:
    return re.sub(r"@\w+", "", text)


def _remove_hashtags(text: str) -> str:
    return re.sub(r"#\w+", "", text)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_caption(text: str) -> str:
    """Remove URLs, @mentions, #hashtags and normalise whitespace."""
    text = _remove_urls(str(text))
    text = _remove_mentions(text)
    text = _remove_hashtags(text)
    return _normalize_whitespace(text)


# ---------------------------------------------------------------------------
# Hashtag extraction
# ---------------------------------------------------------------------------

def extract_hashtags_from_text(text: str) -> List[str]:
    """Extract unique lowercase #hashtags from a string."""
    if not text or not text.strip():
        return []
    tags = re.findall(r"#\w+", text.lower())
    return list(dict.fromkeys(tags))


def extract_combined_hashtags(caption: str, hashtag_list: Sequence[str]) -> List[str]:
    """Merge hashtags found in the caption with an explicit hashtag list."""
    from_caption = extract_hashtags_from_text(caption)
    from_field = [
        t.lower() if t.startswith("#") else f"#{t.lower()}"
        for t in hashtag_list
        if t and t.strip()
    ]
    seen: set[str] = set()
    combined: List[str] = []
    for tag in from_caption + from_field:
        if tag not in seen:
            seen.add(tag)
            combined.append(tag)
    return combined


# ---------------------------------------------------------------------------
# Hashtag Recommender
# ---------------------------------------------------------------------------

class HashtagRecommender:
    """Semantic-neighborhood hashtag recommender.

    Training:
        1. Encode all corpus captions with SBERT.
        2. Build a FAISS inner-product index over L2-normalised embeddings.
        3. Store corpus metadata (hashtags, engagement) alongside the index.

    Inference:
        1. Encode the query caption with the same SBERT model.
        2. Retrieve K nearest neighbours from the FAISS index.
        3. Aggregate neighbour hashtags by frequency and avg engagement.
        4. Return top-N recommendations.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        corpus_hashtags: List[List[str]],
        corpus_engagement: np.ndarray,
        corpus_captions: List[str],
        corpus_video_ids: List[str],
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        import faiss

        self.model_name = model_name
        self.embeddings = embeddings.astype(np.float32)
        self.corpus_hashtags = corpus_hashtags
        self.corpus_engagement = corpus_engagement
        self.corpus_captions = corpus_captions
        self.corpus_video_ids = corpus_video_ids
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ----- training --------------------------------------------------------

    @classmethod
    def train(
        cls,
        records: Sequence[Dict[str, Any]],
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
    ) -> "HashtagRecommender":
        """Build a recommender from raw JSONL-style records."""
        from sentence_transformers import SentenceTransformer

        # Prepare corpus
        captions_clean: List[str] = []
        hashtags_list: List[List[str]] = []
        engagement_vals: List[float] = []
        video_ids: List[str] = []
        raw_captions: List[str] = []

        for rec in records:
            caption = rec.get("caption", "")
            if not caption or len(caption.strip()) < 4:
                continue

            clean = clean_caption(caption)
            if len(clean) < 4:
                continue

            # Extract hashtags: combine caption + explicit field
            explicit_tags = rec.get("hashtags", [])
            if isinstance(explicit_tags, str):
                explicit_tags = [t.strip() for t in explicit_tags.split(",") if t.strip()]
            tags = extract_combined_hashtags(caption, explicit_tags)

            views = max(int(rec.get("views", 0) or 0), 0)

            captions_clean.append(clean)
            hashtags_list.append(tags)
            engagement_vals.append(np.log1p(views))
            video_ids.append(rec.get("video_id", ""))
            raw_captions.append(caption)

        print(f"HashtagRecommender: encoding {len(captions_clean)} captions...")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            captions_clean,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        instance = cls(
            embeddings=embeddings,
            corpus_hashtags=hashtags_list,
            corpus_engagement=np.array(engagement_vals, dtype=np.float32),
            corpus_captions=raw_captions,
            corpus_video_ids=video_ids,
            model_name=model_name,
        )
        instance._model = model
        print(f"HashtagRecommender: index built with {len(captions_clean)} entries")
        return instance

    # ----- inference -------------------------------------------------------

    def recommend(
        self,
        caption: str,
        k: int = 10,
        top_n: int = 10,
        exclude_tags: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Recommend hashtags for a new caption.

        Returns list of dicts with: hashtag, frequency, avg_engagement, score
        """
        clean = clean_caption(caption)
        if len(clean) < 2:
            return []

        query_vec = self.model.encode(
            [clean], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_vec, min(k, self.embeddings.shape[0]))
        neighbour_indices = indices[0]
        neighbour_scores = scores[0]

        exclude = {t.lower() for t in (exclude_tags or [])}
        tag_stats: Dict[str, Dict[str, float]] = {}

        for i, nidx in enumerate(neighbour_indices):
            if nidx < 0:
                continue
            sim = float(neighbour_scores[i])
            eng = float(self.corpus_engagement[nidx])
            for tag in self.corpus_hashtags[nidx]:
                if tag in exclude:
                    continue
                if tag not in tag_stats:
                    tag_stats[tag] = {"count": 0, "total_eng": 0.0, "total_sim": 0.0}
                tag_stats[tag]["count"] += 1
                tag_stats[tag]["total_eng"] += eng
                tag_stats[tag]["total_sim"] += sim

        if not tag_stats:
            return []

        recs = []
        for tag, stats in tag_stats.items():
            count = stats["count"]
            avg_eng = stats["total_eng"] / count
            avg_sim = stats["total_sim"] / count
            # Score: weighted combination of frequency, engagement, and similarity
            score = (count / k) * 0.4 + (avg_eng / 20.0) * 0.3 + avg_sim * 0.3
            recs.append({
                "hashtag": tag,
                "frequency": int(count),
                "avg_engagement": round(avg_eng, 4),
                "avg_similarity": round(avg_sim, 4),
                "score": round(score, 4),
            })

        recs.sort(key=lambda x: (-x["score"], -x["frequency"]))
        return recs[:top_n]

    def recommend_with_neighbours(
        self,
        caption: str,
        k: int = 10,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """Recommend hashtags and also return the nearest neighbour videos."""
        clean = clean_caption(caption)
        if len(clean) < 2:
            return {"hashtags": [], "neighbours": []}

        query_vec = self.model.encode(
            [clean], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_vec, min(k, self.embeddings.shape[0]))
        neighbour_indices = indices[0]
        neighbour_scores = scores[0]

        neighbours = []
        tag_stats: Dict[str, Dict[str, float]] = {}

        for i, nidx in enumerate(neighbour_indices):
            if nidx < 0:
                continue
            sim = float(neighbour_scores[i])
            eng = float(self.corpus_engagement[nidx])
            neighbours.append({
                "video_id": self.corpus_video_ids[nidx],
                "caption": self.corpus_captions[nidx],
                "similarity": round(sim, 4),
                "engagement_log": round(eng, 4),
                "hashtags": self.corpus_hashtags[nidx],
            })
            for tag in self.corpus_hashtags[nidx]:
                if tag not in tag_stats:
                    tag_stats[tag] = {"count": 0, "total_eng": 0.0, "total_sim": 0.0}
                tag_stats[tag]["count"] += 1
                tag_stats[tag]["total_eng"] += eng
                tag_stats[tag]["total_sim"] += sim

        recs = []
        for tag, stats in tag_stats.items():
            count = stats["count"]
            avg_eng = stats["total_eng"] / count
            avg_sim = stats["total_sim"] / count
            score = (count / k) * 0.4 + (avg_eng / 20.0) * 0.3 + avg_sim * 0.3
            recs.append({
                "hashtag": tag,
                "frequency": int(count),
                "avg_engagement": round(avg_eng, 4),
                "avg_similarity": round(avg_sim, 4),
                "score": round(score, 4),
            })

        recs.sort(key=lambda x: (-x["score"], -x["frequency"]))
        return {"hashtags": recs[:top_n], "neighbours": neighbours}

    # ----- persistence -----------------------------------------------------

    def save(self, path: Path) -> None:
        """Save the trained recommender to disk."""
        import faiss

        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss.index"))
        np.save(path / "embeddings.npy", self.embeddings)
        np.save(path / "engagement.npy", self.corpus_engagement)
        meta = {
            "model_name": self.model_name,
            "corpus_size": len(self.corpus_captions),
            "embedding_dim": int(self.embeddings.shape[1]),
            "corpus_hashtags": self.corpus_hashtags,
            "corpus_captions": self.corpus_captions,
            "corpus_video_ids": self.corpus_video_ids,
        }
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        print(f"HashtagRecommender saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "HashtagRecommender":
        """Load a trained recommender from disk."""
        import faiss

        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        embeddings = np.load(path / "embeddings.npy")
        engagement = np.load(path / "engagement.npy")
        instance = cls(
            embeddings=embeddings,
            corpus_hashtags=meta["corpus_hashtags"],
            corpus_engagement=engagement,
            corpus_captions=meta["corpus_captions"],
            corpus_video_ids=meta["corpus_video_ids"],
            model_name=meta["model_name"],
        )
        instance.index = faiss.read_index(str(path / "faiss.index"))
        print(f"HashtagRecommender loaded: {meta['corpus_size']} entries, {meta['embedding_dim']}d")
        return instance
