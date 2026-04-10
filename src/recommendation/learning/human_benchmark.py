from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .evaluator import mrr_at_k, ndcg_at_k, recall_at_k


HUMAN_COMPARABLE_BENCHMARK_VERSION = "recommender.human_comparable_benchmark.v1"
HUMAN_COMPARABLE_BENCHMARK_RUBRIC_VERSION = "comparable_label_rubric.v1"

LABEL_GOOD = "good"
LABEL_UNCLEAR = "unclear"
LABEL_BAD = "bad"
LABEL_VALUES = {LABEL_GOOD, LABEL_UNCLEAR, LABEL_BAD}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_rubric() -> Dict[str, Any]:
    return {
        "version": HUMAN_COMPARABLE_BENCHMARK_RUBRIC_VERSION,
        "scale": [LABEL_GOOD, LABEL_UNCLEAR, LABEL_BAD],
        "instructions": [
            "Judge each candidate as a historical comparable for the query draft under the stated objective.",
            "Prefer topical fit first, then creator intent, then format usefulness as a reference example.",
            "Use 'unclear' when the candidate is mixed or evidence is too thin to call confidently.",
        ],
        "definitions": {
            LABEL_GOOD: (
                "Strong comparable: topically close, intent-compatible, and useful as a reference example."
            ),
            LABEL_UNCLEAR: (
                "Mixed comparable: partially aligned but with meaningful topical, intent, or format mismatch."
            ),
            LABEL_BAD: (
                "Poor comparable: off-topic, generic-but-unhelpful, or not useful as evidence for this draft."
            ),
        },
        "pairwise_training_policy": {
            "positive_labels": [LABEL_GOOD],
            "negative_labels": [LABEL_BAD],
            "ignored_labels": [LABEL_UNCLEAR],
        },
    }


def label_to_relevance(label: Optional[str]) -> float:
    value = str(label or "").strip().lower()
    if value == LABEL_GOOD:
        return 2.0
    if value == LABEL_UNCLEAR:
        return 1.0
    return 0.0


def _safe_rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count) / float(total)


@dataclass
class BenchmarkQuery:
    query_id: str
    display: Dict[str, Any]
    query_payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "display": dict(self.display),
            "query_payload": dict(self.query_payload),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BenchmarkQuery":
        return cls(
            query_id=str(payload.get("query_id") or ""),
            display=dict(payload.get("display") or {}),
            query_payload=dict(payload.get("query_payload") or {}),
        )


@dataclass
class BenchmarkCandidate:
    candidate_id: str
    display: Dict[str, Any]
    candidate_payload: Dict[str, Any]
    baseline_rank: Optional[int] = None
    baseline_score: Optional[float] = None
    support_level: Optional[str] = None
    ranking_reasons: List[str] = field(default_factory=list)
    label: Optional[str] = None
    label_notes: str = ""

    def normalized_label(self) -> Optional[str]:
        value = str(self.label or "").strip().lower()
        return value if value in LABEL_VALUES else None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "display": dict(self.display),
            "candidate_payload": dict(self.candidate_payload),
            "baseline_rank": self.baseline_rank,
            "baseline_score": self.baseline_score,
            "support_level": self.support_level,
            "ranking_reasons": list(self.ranking_reasons),
            "label": self.label,
            "label_notes": self.label_notes,
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BenchmarkCandidate":
        return cls(
            candidate_id=str(payload.get("candidate_id") or ""),
            display=dict(payload.get("display") or {}),
            candidate_payload=dict(payload.get("candidate_payload") or {}),
            baseline_rank=int(payload["baseline_rank"])
            if payload.get("baseline_rank") is not None
            else None,
            baseline_score=float(payload["baseline_score"])
            if payload.get("baseline_score") is not None
            else None,
            support_level=str(payload["support_level"])
            if payload.get("support_level") is not None
            else None,
            ranking_reasons=[str(item) for item in list(payload.get("ranking_reasons") or [])],
            label=str(payload["label"]) if payload.get("label") is not None else None,
            label_notes=str(payload.get("label_notes") or ""),
        )


@dataclass
class BenchmarkCase:
    case_id: str
    objective: str
    query: BenchmarkQuery
    candidates: List[BenchmarkCandidate]
    retrieve_k: int
    label_pool_size: int
    source_candidate_pool_size: int
    notes: str = ""

    def labeled_candidates(self) -> List[BenchmarkCandidate]:
        return [candidate for candidate in self.candidates if candidate.normalized_label() is not None]

    def relevance_map(self) -> Dict[str, float]:
        return {
            candidate.candidate_id: label_to_relevance(candidate.normalized_label())
            for candidate in self.labeled_candidates()
        }

    def relevant_ids(self) -> set[str]:
        return {
            candidate.candidate_id
            for candidate in self.candidates
            if candidate.normalized_label() == LABEL_GOOD
        }

    def pairwise_preferences(self) -> List[Tuple[str, str]]:
        positives = [
            candidate.candidate_id
            for candidate in self.candidates
            if candidate.normalized_label() == LABEL_GOOD
        ]
        negatives = [
            candidate.candidate_id
            for candidate in self.candidates
            if candidate.normalized_label() == LABEL_BAD
        ]
        return [
            (positive_id, negative_id)
            for positive_id in positives
            for negative_id in negatives
            if positive_id != negative_id
        ]

    def baseline_ranked_ids(self) -> List[str]:
        ordered = sorted(
            self.candidates,
            key=lambda item: (
                item.baseline_rank is None,
                item.baseline_rank if item.baseline_rank is not None else 10_000,
                -(item.baseline_score or 0.0),
                item.candidate_id,
            ),
        )
        return [candidate.candidate_id for candidate in ordered]

    def label_summary(self) -> Dict[str, int]:
        counts = {LABEL_GOOD: 0, LABEL_UNCLEAR: 0, LABEL_BAD: 0, "unlabeled": 0}
        for candidate in self.candidates:
            label = candidate.normalized_label()
            if label is None:
                counts["unlabeled"] += 1
            else:
                counts[label] += 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "case_id": self.case_id,
            "objective": self.objective,
            "query": self.query.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "retrieve_k": self.retrieve_k,
            "label_pool_size": self.label_pool_size,
            "source_candidate_pool_size": self.source_candidate_pool_size,
        }
        if self.notes:
            payload["notes"] = self.notes
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BenchmarkCase":
        return cls(
            case_id=str(payload.get("case_id") or ""),
            objective=str(payload.get("objective") or ""),
            query=BenchmarkQuery.from_dict(dict(payload.get("query") or {})),
            candidates=[
                BenchmarkCandidate.from_dict(dict(item))
                for item in list(payload.get("candidates") or [])
            ],
            retrieve_k=int(payload.get("retrieve_k") or 0),
            label_pool_size=int(payload.get("label_pool_size") or 0),
            source_candidate_pool_size=int(payload.get("source_candidate_pool_size") or 0),
            notes=str(payload.get("notes") or ""),
        )


@dataclass
class BenchmarkDataset:
    version: str
    generated_at: str
    bundle_dir: str
    sample_metadata: Dict[str, Any]
    rubric: Dict[str, Any]
    cases: List[BenchmarkCase]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "bundle_dir": self.bundle_dir,
            "sample_metadata": dict(self.sample_metadata),
            "rubric": dict(self.rubric),
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BenchmarkDataset":
        return cls(
            version=str(payload.get("version") or HUMAN_COMPARABLE_BENCHMARK_VERSION),
            generated_at=str(payload.get("generated_at") or _iso_now()),
            bundle_dir=str(payload.get("bundle_dir") or ""),
            sample_metadata=dict(payload.get("sample_metadata") or {}),
            rubric=dict(payload.get("rubric") or default_rubric()),
            cases=[BenchmarkCase.from_dict(dict(item)) for item in list(payload.get("cases") or [])],
        )


def save_benchmark_dataset(dataset: BenchmarkDataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dataset.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_benchmark_dataset(path: Path) -> BenchmarkDataset:
    return BenchmarkDataset.from_dict(json.loads(path.read_text(encoding="utf-8")))


def summarize_benchmark_dataset(dataset: BenchmarkDataset) -> Dict[str, Any]:
    total_cases = len(dataset.cases)
    total_candidates = sum(len(case.candidates) for case in dataset.cases)
    label_counts = {LABEL_GOOD: 0, LABEL_UNCLEAR: 0, LABEL_BAD: 0, "unlabeled": 0}
    objective_counts: Dict[str, int] = {}
    pair_count = 0
    for case in dataset.cases:
        objective_counts[case.objective] = objective_counts.get(case.objective, 0) + 1
        pair_count += len(case.pairwise_preferences())
        for key, value in case.label_summary().items():
            label_counts[key] = label_counts.get(key, 0) + int(value)
    return {
        "version": dataset.version,
        "generated_at": dataset.generated_at,
        "case_count": total_cases,
        "candidate_count": total_candidates,
        "objective_counts": objective_counts,
        "label_counts": label_counts,
        "pairwise_preference_count": pair_count,
    }


def evaluate_case_ranked_ids(
    case: BenchmarkCase,
    ranked_ids: Sequence[str],
    *,
    k_values: Sequence[int] = (5, 10),
) -> Dict[str, Optional[float]]:
    relevance = case.relevance_map()
    relevant_ids = case.relevant_ids()
    label_map = {
        candidate.candidate_id: candidate.normalized_label()
        for candidate in case.labeled_candidates()
    }
    metrics: Dict[str, Optional[float]] = {
        "has_good_labels": 1.0 if relevant_ids else 0.0,
        "all_bad_case": 1.0
        if all(
            candidate.normalized_label() == LABEL_BAD
            for candidate in case.labeled_candidates()
        )
        and bool(case.labeled_candidates())
        else 0.0,
    }
    for k in k_values:
        cutoff = max(1, int(k))
        top_ids = [str(candidate_id) for candidate_id in list(ranked_ids)[:cutoff]]
        denominator = len(top_ids)
        good_count = sum(1 for candidate_id in top_ids if label_map.get(candidate_id) == LABEL_GOOD)
        unclear_count = sum(
            1 for candidate_id in top_ids if label_map.get(candidate_id) == LABEL_UNCLEAR
        )
        bad_count = sum(1 for candidate_id in top_ids if label_map.get(candidate_id) == LABEL_BAD)
        metrics[f"good_rate@{cutoff}"] = _safe_rate(good_count, denominator)
        metrics[f"unclear_rate@{cutoff}"] = _safe_rate(unclear_count, denominator)
        metrics[f"bad_rate@{cutoff}"] = _safe_rate(bad_count, denominator)
        metrics[f"ndcg@{cutoff}"] = ndcg_at_k(ranked_ids, relevance, cutoff)
        if relevant_ids:
            metrics[f"mrr@{cutoff}"] = mrr_at_k(ranked_ids, relevant_ids, cutoff)
            metrics[f"recall@{cutoff}"] = recall_at_k(ranked_ids, relevant_ids, cutoff)
        else:
            metrics[f"mrr@{cutoff}"] = None
            metrics[f"recall@{cutoff}"] = None
    return metrics


def aggregate_case_metrics(metric_rows: Iterable[Dict[str, Optional[float]]]) -> Dict[str, float]:
    rows = list(metric_rows)
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row.keys()})
    out: Dict[str, float] = {}
    for key in keys:
        values = [
            float(row[key])
            for row in rows
            if key in row and row[key] is not None
        ]
        if not values:
            continue
        out[key] = round(sum(values) / float(len(values)), 6)
    return out
