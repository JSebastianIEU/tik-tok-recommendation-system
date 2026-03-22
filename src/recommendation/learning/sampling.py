from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .temporal import row_text


def _tokenize(value: str) -> List[str]:
    normalized = (
        value.lower()
        .replace("#", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace("/", " ")
        .replace("-", " ")
        .replace("?", " ")
        .replace("!", " ")
    )
    return [token for token in normalized.split() if len(token) >= 2]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    inter = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return 0.0 if union == 0 else inter / union


def _is_row_censored(row: Dict[str, Any]) -> bool:
    labels = row.get("labels", {})
    if not isinstance(labels, dict):
        return True
    if "window_hours_observed" not in labels:
        return True
    try:
        observed = float(labels.get("window_hours_observed") or 0.0)
    except (TypeError, ValueError):
        return True
    return observed <= 0


@dataclass
class NegativeSamplerConfig:
    negatives_per_positive: int = 4
    hard_ratio: float = 0.5
    semihard_ratio: float = 0.3
    easy_ratio: float = 0.2
    max_per_author: int = 2
    max_per_topic: int = 4
    seed: int = 13


class NegativeSampler:
    def __init__(self, config: Optional[NegativeSamplerConfig] = None) -> None:
        self.config = config or NegativeSamplerConfig()
        total = self.config.hard_ratio + self.config.semihard_ratio + self.config.easy_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError("hard_ratio + semihard_ratio + easy_ratio must sum to 1.0")

    def _compatibility_score(
        self,
        query_row: Dict[str, Any],
        candidate_row: Dict[str, Any],
    ) -> float:
        query_tokens = _tokenize(row_text(query_row))
        candidate_tokens = _tokenize(row_text(candidate_row))
        sim = _jaccard(query_tokens, candidate_tokens)
        topic_bonus = 0.2 if query_row.get("topic_key") == candidate_row.get("topic_key") else 0.0
        author_penalty = 0.2 if query_row.get("author_id") == candidate_row.get("author_id") else 0.0
        return max(0.0, sim + topic_bonus - author_penalty)

    def sample(
        self,
        query_row: Dict[str, Any],
        positives: Sequence[Dict[str, Any]],
        candidate_pool: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        positive_count = len(positives)
        if positive_count <= 0:
            return []

        required_negatives = positive_count * max(1, self.config.negatives_per_positive)
        eligible_candidates: List[Dict[str, Any]] = []
        for row in candidate_pool:
            if row.get("row_id") == query_row.get("row_id"):
                continue
            if _is_row_censored(row):
                continue
            eligible_candidates.append(row)

        if not eligible_candidates:
            return []

        scored = [
            (self._compatibility_score(query_row, row), row)
            for row in eligible_candidates
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        hard_count = int(round(required_negatives * self.config.hard_ratio))
        semihard_count = int(round(required_negatives * self.config.semihard_ratio))
        easy_count = max(0, required_negatives - hard_count - semihard_count)

        hard_pool = [item[1] for item in scored[: max(1, len(scored) // 3)]]
        semihard_pool = [item[1] for item in scored[max(1, len(scored) // 3) : max(2, 2 * len(scored) // 3)]]
        easy_pool = [item[1] for item in scored[max(2, 2 * len(scored) // 3) :]]

        rng = random.Random(f"{self.config.seed}:{query_row.get('row_id')}")
        rng.shuffle(hard_pool)
        rng.shuffle(semihard_pool)
        rng.shuffle(easy_pool)

        raw_selected = (
            hard_pool[:hard_count]
            + semihard_pool[:semihard_count]
            + easy_pool[:easy_count]
        )

        selected: List[Dict[str, Any]] = []
        author_counts: Dict[str, int] = {}
        topic_counts: Dict[str, int] = {}
        seen_ids = set()
        for row in raw_selected:
            row_id = str(row.get("row_id"))
            if row_id in seen_ids:
                continue
            author_key = str(row.get("author_id") or "unknown")
            topic_key = str(row.get("topic_key") or "general")
            if author_counts.get(author_key, 0) >= self.config.max_per_author:
                continue
            if topic_counts.get(topic_key, 0) >= self.config.max_per_topic:
                continue
            seen_ids.add(row_id)
            author_counts[author_key] = author_counts.get(author_key, 0) + 1
            topic_counts[topic_key] = topic_counts.get(topic_key, 0) + 1
            selected.append(row)

        if len(selected) >= required_negatives:
            return selected[:required_negatives]

        for _, row in scored:
            row_id = str(row.get("row_id"))
            if row_id in seen_ids:
                continue
            selected.append(row)
            seen_ids.add(row_id)
            if len(selected) >= required_negatives:
                break
        return selected

