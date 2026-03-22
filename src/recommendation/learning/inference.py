from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .artifacts import ArtifactRegistry
from .objectives import map_objective
from .ranker import ObjectiveRankerModel
from .retriever import HybridRetriever
from .temporal import parse_dt


def _to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _token_count(text: str) -> int:
    return len([token for token in text.split() if token.strip()])


def _extract_hashtag_count(text: str) -> int:
    return len([token for token in text.split() if token.strip().startswith("#")])


def _pair_vector(query_row: Dict[str, Any], candidate_row: Dict[str, Any], similarity: float) -> np.ndarray:
    qf = query_row.get("features", {})
    cf = candidate_row.get("features", {})
    q_caption = float(qf.get("caption_word_count") or 0.0)
    q_hashtag = float(qf.get("hashtag_count") or 0.0)
    q_keyword = float(qf.get("keyword_count") or 0.0)
    c_caption = float(cf.get("caption_word_count") or 0.0)
    c_hashtag = float(cf.get("hashtag_count") or 0.0)
    c_keyword = float(cf.get("keyword_count") or 0.0)

    return np.asarray(
        [
            [
                float(similarity),
                q_caption,
                q_hashtag,
                q_keyword,
                c_caption,
                c_hashtag,
                c_keyword,
                abs(q_caption - c_caption),
                abs(q_hashtag - c_hashtag),
                abs(q_keyword - c_keyword),
                1.0 if query_row.get("author_id") == candidate_row.get("author_id") else 0.0,
                1.0 if query_row.get("topic_key") == candidate_row.get("topic_key") else 0.0,
            ]
        ],
        dtype=np.float32,
    )


def _to_runtime_row(row_id: str, payload: Dict[str, Any], fallback_as_of: datetime) -> Dict[str, Any]:
    text = _safe_text(payload.get("text"))
    topic_key = _safe_text(payload.get("topic_key")) or "general"
    author_id = _safe_text(payload.get("author_id")) or "unknown"
    as_of = parse_dt(payload.get("as_of_time")) or fallback_as_of
    return {
        "row_id": row_id,
        "video_id": row_id,
        "author_id": author_id,
        "topic_key": topic_key,
        "as_of_time": as_of,
        "features": {
            "caption_word_count": _token_count(text),
            "hashtag_count": _extract_hashtag_count(text),
            "keyword_count": len(
                [token for token in text.lower().split() if len(token) >= 4]
            ),
            "missingness_flags": [],
        },
        "_runtime_text": text,
        "_source_payload": payload,
    }


@dataclass
class RecommenderRuntimeConfig:
    feature_schema_hash: Optional[str] = None
    contract_version: Optional[str] = None
    datamart_version: Optional[str] = None
    component: Optional[str] = "recommender-learning-v1"


class RecommenderRuntime:
    def __init__(
        self,
        bundle_dir: Path,
        config: Optional[RecommenderRuntimeConfig] = None,
    ) -> None:
        self.bundle_dir = bundle_dir
        self.config = config or RecommenderRuntimeConfig()
        registry = ArtifactRegistry(bundle_dir.parent)
        manifest = registry.load_manifest(bundle_dir)
        required_fields = (
            "component",
            "contract_version",
            "datamart_version",
            "feature_schema_hash",
            "objectives",
        )
        missing = [field for field in required_fields if field not in manifest]
        if missing:
            raise ValueError(
                f"Invalid recommender artifact manifest. Missing fields: {', '.join(missing)}"
            )

        expected: Dict[str, Any] = {}
        if self.config.component:
            expected["component"] = self.config.component
        if self.config.feature_schema_hash:
            expected["feature_schema_hash"] = self.config.feature_schema_hash
        if self.config.contract_version:
            expected["contract_version"] = self.config.contract_version
        if self.config.datamart_version:
            expected["datamart_version"] = self.config.datamart_version
        if expected:
            registry.assert_compatible(bundle_dir, expected)
        self.manifest = manifest
        self.retriever = HybridRetriever.load(bundle_dir / "retriever")
        self.rankers: Dict[str, ObjectiveRankerModel] = {}
        for objective in manifest.get("objectives", []):
            ranker_dir = bundle_dir / "rankers" / str(objective)
            if ranker_dir.exists():
                self.rankers[str(objective)] = ObjectiveRankerModel.load(ranker_dir)
        missing_rankers = [
            str(objective)
            for objective in manifest.get("objectives", [])
            if str(objective) not in self.rankers
        ]
        if missing_rankers:
            raise ValueError(
                "Invalid recommender artifact bundle. Missing ranker models for: "
                + ", ".join(sorted(missing_rankers))
            )

    def recommend(
        self,
        objective: str,
        as_of_time: Any,
        query: Dict[str, Any],
        candidates: Sequence[Dict[str, Any]],
        top_k: int = 20,
        retrieve_k: int = 200,
        debug: bool = False,
    ) -> Dict[str, Any]:
        requested_objective, effective_objective = map_objective(objective)
        configured_objectives = {
            str(item) for item in self.manifest.get("objectives", [])
        }
        if effective_objective not in configured_objectives:
            raise ValueError(
                f"Objective '{effective_objective}' is not available in loaded artifact bundle."
            )
        as_of = parse_dt(as_of_time)
        if as_of is None:
            raise ValueError("as_of_time must be a valid ISO-8601 timestamp.")
        query_row = _to_runtime_row(
            row_id=str(query.get("query_id") or "query"),
            payload={
                "text": query.get("text")
                or " ".join(
                    [
                        _safe_text(query.get("description")),
                        " ".join(query.get("hashtags") or []),
                        " ".join(query.get("mentions") or []),
                    ]
                ),
                "topic_key": query.get("topic_key") or query.get("objective") or "general",
                "author_id": query.get("author_id"),
                "as_of_time": query.get("as_of_time") or as_of,
            },
            fallback_as_of=as_of,
        )

        candidate_rows = []
        for item in candidates:
            candidate_id = str(item.get("candidate_id") or item.get("video_id") or item.get("row_id") or "")
            if not candidate_id:
                continue
            row = _to_runtime_row(
                row_id=candidate_id,
                payload={
                    "text": item.get("text")
                    or " ".join(
                        [
                            _safe_text(item.get("caption")),
                            " ".join(item.get("hashtags") or []),
                            " ".join(item.get("keywords") or []),
                        ]
                    ),
                    "topic_key": item.get("topic_key") or "general",
                    "author_id": item.get("author_id"),
                    "as_of_time": item.get("as_of_time") or item.get("posted_at") or as_of,
                },
                fallback_as_of=as_of,
            )
            candidate_rows.append(row)

        retrieved = self.retriever.retrieve(
            query_row=query_row,
            candidate_rows=candidate_rows,
            top_k=max(top_k, retrieve_k),
            index_cutoff_time=as_of,
        )

        ranker = self.rankers.get(effective_objective)
        fallback_mode = ranker is None
        ranked_items = []
        candidate_by_id = {row["row_id"]: row for row in candidate_rows}
        for item in retrieved:
            candidate_row = candidate_by_id.get(item["candidate_row_id"])
            if candidate_row is None:
                continue
            if ranker is None:
                final_score = float(item["fused_score"])
            else:
                pair_vec = _pair_vector(
                    query_row=query_row,
                    candidate_row=candidate_row,
                    similarity=float(item.get("fused_score") or 0.0),
                )
                final_score = float(ranker.predict_scores(pair_vec)[0])
            ranked_items.append(
                {
                    "candidate_id": candidate_row["row_id"],
                    "candidate_row_id": candidate_row["row_id"],
                    "score": float(final_score),
                    "similarity": {
                        "sparse": float(item.get("sparse_score") or 0.0),
                        "dense": float(item.get("dense_score") or 0.0),
                        "fused": float(item.get("fused_score") or 0.0),
                    },
                    "trace": {
                        "objective_model": effective_objective if ranker else None,
                        "ranker_backend": ranker.backend if ranker else None,
                    },
                }
            )
        ranked_items.sort(key=lambda item: item["score"], reverse=True)
        top_items = ranked_items[: max(1, top_k)]

        response: Dict[str, Any] = {
            "objective": requested_objective,
            "objective_effective": effective_objective,
            "generated_at": _to_utc_iso(datetime.now(timezone.utc)),
            "fallback_mode": fallback_mode,
            "items": [
                {
                    "candidate_id": item["candidate_id"],
                    "rank": idx + 1,
                    "score": round(item["score"], 6),
                    "similarity": {
                        "sparse": round(item["similarity"]["sparse"], 6),
                        "dense": round(item["similarity"]["dense"], 6),
                        "fused": round(item["similarity"]["fused"], 6),
                    },
                    "trace": item["trace"],
                }
                for idx, item in enumerate(top_items)
            ],
        }
        if debug:
            response["debug"] = {
                "candidate_pool_size": len(candidate_rows),
                "retrieved_count": len(retrieved),
                "ranked_count": len(ranked_items),
                "bundle_dir": str(self.bundle_dir),
            }
        return response
