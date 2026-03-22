from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .temporal import row_as_of_time, row_text

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BM25Okapi = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


@dataclass
class HybridRetrieverTrainerConfig:
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_weight: float = 0.5
    dense_weight: float = 0.5


def _to_tokens(text: str) -> List[str]:
    return [
        token
        for token in text.lower().replace("#", " ").split()
        if len(token.strip()) >= 2
    ]


def _normalize_scores(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return np.array([], dtype=np.float32)
    min_v = float(np.min(array))
    max_v = float(np.max(array))
    if math.isclose(min_v, max_v):
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_v) / (max_v - min_v)


class HybridRetriever:
    def __init__(
        self,
        row_ids: List[str],
        row_texts: List[str],
        row_times: Dict[str, str],
        sparse_backend: str,
        sparse_payload: Dict[str, Any],
        dense_backend: str,
        dense_payload: Dict[str, Any],
        config: HybridRetrieverTrainerConfig,
    ) -> None:
        self.row_ids = row_ids
        self.row_texts = row_texts
        self.row_times = row_times
        self.sparse_backend = sparse_backend
        self.sparse_payload = sparse_payload
        self.dense_backend = dense_backend
        self.dense_payload = dense_payload
        self.config = config
        self._dense_encoder: Any = None

    @classmethod
    def train(
        cls,
        rows: Iterable[Dict[str, Any]],
        config: Optional[HybridRetrieverTrainerConfig] = None,
    ) -> "HybridRetriever":
        cfg = config or HybridRetrieverTrainerConfig()
        ordered_rows = list(rows)
        row_ids = [str(row.get("row_id")) for row in ordered_rows]
        row_texts = [row_text(row) for row in ordered_rows]
        row_times = {
            str(row.get("row_id")): (
                row_as_of_time(row).isoformat().replace("+00:00", "Z")
                if row_as_of_time(row)
                else ""
            )
            for row in ordered_rows
        }

        sparse_backend = "tfidf"
        sparse_payload: Dict[str, Any]
        if BM25Okapi is not None:
            tokenized = [_to_tokens(text) for text in row_texts]
            sparse_backend = "bm25"
            sparse_payload = {"model": BM25Okapi(tokenized), "tokenized": tokenized}
        else:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                token_pattern=r"\b\w+\b",
            )
            matrix = vectorizer.fit_transform(row_texts)
            sparse_payload = {"vectorizer": vectorizer, "matrix": matrix}

        dense_backend = "tfidf-char"
        dense_payload: Dict[str, Any]
        if SentenceTransformer is not None:
            try:
                encoder = SentenceTransformer(cfg.dense_model_name)
                embeddings = encoder.encode(
                    row_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                dense_backend = "sentence-transformers"
                dense_payload = {
                    "model_name": cfg.dense_model_name,
                    "embeddings": embeddings,
                }
            except Exception:  # pragma: no cover - depends on local model availability
                dense_vectorizer = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    lowercase=True,
                )
                dense_matrix = dense_vectorizer.fit_transform(row_texts)
                dense_payload = {"vectorizer": dense_vectorizer, "matrix": dense_matrix}
        else:
            dense_vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                lowercase=True,
            )
            dense_matrix = dense_vectorizer.fit_transform(row_texts)
            dense_payload = {"vectorizer": dense_vectorizer, "matrix": dense_matrix}

        return cls(
            row_ids=row_ids,
            row_texts=row_texts,
            row_times=row_times,
            sparse_backend=sparse_backend,
            sparse_payload=sparse_payload,
            dense_backend=dense_backend,
            dense_payload=dense_payload,
            config=cfg,
        )

    def _sparse_scores(self, query_text: str) -> np.ndarray:
        if self.sparse_backend == "bm25":
            model = self.sparse_payload["model"]
            query_tokens = _to_tokens(query_text)
            return np.asarray(model.get_scores(query_tokens), dtype=np.float32)
        vectorizer = self.sparse_payload["vectorizer"]
        matrix = self.sparse_payload["matrix"]
        q = vectorizer.transform([query_text])
        return cosine_similarity(q, matrix)[0].astype(np.float32)

    def _dense_scores(self, query_text: str) -> np.ndarray:
        if self.dense_backend == "sentence-transformers":
            if SentenceTransformer is None:  # pragma: no cover
                return np.zeros(len(self.row_ids), dtype=np.float32)
            model_name = self.dense_payload["model_name"]
            try:
                if self._dense_encoder is None:
                    self._dense_encoder = SentenceTransformer(model_name)
                q = self._dense_encoder.encode(
                    [query_text],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )[0]
            except Exception:  # pragma: no cover - depends on local model availability
                return np.zeros(len(self.row_ids), dtype=np.float32)
            embeddings = self.dense_payload["embeddings"]
            return np.dot(embeddings, q).astype(np.float32)
        vectorizer = self.dense_payload["vectorizer"]
        matrix = self.dense_payload["matrix"]
        q = vectorizer.transform([query_text])
        return cosine_similarity(q, matrix)[0].astype(np.float32)

    def retrieve(
        self,
        query_row: Dict[str, Any],
        candidate_rows: Sequence[Dict[str, Any]],
        top_k: int = 200,
        index_cutoff_time: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        query_text = row_text(query_row)
        if not query_text.strip():
            query_text = str(query_row.get("topic_key") or "general")

        query_as_of = row_as_of_time(query_row)
        index_cutoff = row_as_of_time({"as_of_time": index_cutoff_time}) if index_cutoff_time else query_as_of

        sparse_scores = self._sparse_scores(query_text)
        dense_scores = self._dense_scores(query_text)

        sparse_norm = _normalize_scores(sparse_scores)
        dense_norm = _normalize_scores(dense_scores)
        fused = self.config.sparse_weight * sparse_norm + self.config.dense_weight * dense_norm

        candidate_by_id = {str(row.get("row_id")): row for row in candidate_rows}
        items: List[Dict[str, Any]] = []
        for idx, row_id in enumerate(self.row_ids):
            candidate = candidate_by_id.get(row_id)
            if candidate is None:
                continue
            candidate_as_of = row_as_of_time(candidate)
            if query_as_of and candidate_as_of and candidate_as_of >= query_as_of:
                continue
            if index_cutoff and candidate_as_of and candidate_as_of > index_cutoff:
                continue
            items.append(
                {
                    "candidate_row_id": row_id,
                    "sparse_score": float(sparse_norm[idx]) if idx < len(sparse_norm) else 0.0,
                    "dense_score": float(dense_norm[idx]) if idx < len(dense_norm) else 0.0,
                    "fused_score": float(fused[idx]) if idx < len(fused) else 0.0,
                }
            )
        items.sort(key=lambda item: item["fused_score"], reverse=True)
        return items[: max(1, top_k)]

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "row_ids": self.row_ids,
            "row_texts": self.row_texts,
            "row_times": self.row_times,
            "sparse_backend": self.sparse_backend,
            "dense_backend": self.dense_backend,
            "config": {
                "dense_model_name": self.config.dense_model_name,
                "sparse_weight": self.config.sparse_weight,
                "dense_weight": self.config.dense_weight,
            },
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with (output_dir / "sparse.pkl").open("wb") as fh:
            pickle.dump(self.sparse_payload, fh)
        with (output_dir / "dense.pkl").open("wb") as fh:
            pickle.dump(self.dense_payload, fh)

    @classmethod
    def load(cls, output_dir: Path) -> "HybridRetriever":
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        with (output_dir / "sparse.pkl").open("rb") as fh:
            sparse_payload = pickle.load(fh)
        with (output_dir / "dense.pkl").open("rb") as fh:
            dense_payload = pickle.load(fh)

        cfg = HybridRetrieverTrainerConfig(
            dense_model_name=manifest["config"]["dense_model_name"],
            sparse_weight=float(manifest["config"]["sparse_weight"]),
            dense_weight=float(manifest["config"]["dense_weight"]),
        )
        return cls(
            row_ids=list(manifest["row_ids"]),
            row_texts=list(manifest["row_texts"]),
            row_times=dict(manifest["row_times"]),
            sparse_backend=str(manifest["sparse_backend"]),
            sparse_payload=sparse_payload,
            dense_backend=str(manifest["dense_backend"]),
            dense_payload=dense_payload,
            config=cfg,
        )
