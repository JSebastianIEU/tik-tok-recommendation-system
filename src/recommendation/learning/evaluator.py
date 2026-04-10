from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Sequence

import numpy as np


def recall_at_k(recommended_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = list(recommended_ids[: max(1, k)])
    hits = sum(1 for item in top if item in relevant_ids)
    return hits / max(1, len(relevant_ids))


def mrr_at_k(recommended_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    top = list(recommended_ids[: max(1, k)])
    for idx, item in enumerate(top, start=1):
        if item in relevant_ids:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(recommended_ids: Sequence[str], relevance: Dict[str, float], k: int) -> float:
    top = list(recommended_ids[: max(1, k)])
    dcg = 0.0
    for idx, item in enumerate(top, start=1):
        rel = max(0.0, float(relevance.get(item, 0.0)))
        dcg += (2**rel - 1.0) / math.log2(idx + 1.0)

    ideal = sorted([max(0.0, float(v)) for v in relevance.values()], reverse=True)[: len(top)]
    idcg = 0.0
    for idx, rel in enumerate(ideal, start=1):
        idcg += (2**rel - 1.0) / math.log2(idx + 1.0)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def aggregate(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return float(np.mean(vals))


def evaluate_retrieval(
    queries: Sequence[Dict[str, Any]],
    top_k: int = 200,
) -> Dict[str, float]:
    recalls_50 = []
    recalls_100 = []
    recalls_200 = []
    for query in queries:
        ranked = [str(item["candidate_row_id"]) for item in query.get("items", [])]
        relevant = {str(item) for item in query.get("relevant_ids", [])}
        recalls_50.append(recall_at_k(ranked, relevant, min(50, top_k)))
        recalls_100.append(recall_at_k(ranked, relevant, min(100, top_k)))
        recalls_200.append(recall_at_k(ranked, relevant, min(200, top_k)))
    return {
        "recall@50": aggregate(recalls_50),
        "recall@100": aggregate(recalls_100),
        "recall@200": aggregate(recalls_200),
    }


def evaluate_ranking(
    queries: Sequence[Dict[str, Any]],
    k_values: Sequence[int] = (10, 20),
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in k_values:
        ndcgs = []
        mrrs = []
        for query in queries:
            ranked = [str(item["candidate_row_id"]) for item in query.get("items", [])]
            relevance = {
                str(item["candidate_row_id"]): float(item.get("relevance", 0.0))
                for item in query.get("items", [])
            }
            relevant_ids = {
                key for key, value in relevance.items() if value > 0
            }
            ndcgs.append(ndcg_at_k(ranked, relevance, k))
            mrrs.append(mrr_at_k(ranked, relevant_ids, k))
        out[f"ndcg@{k}"] = aggregate(ndcgs)
        out[f"mrr@{k}"] = aggregate(mrrs)
    return out
