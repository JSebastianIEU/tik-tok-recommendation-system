from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .learned_reranker import PairwiseTrainingRow, candidate_feature_payload_from_item


EXPLICIT_POSITIVE_EVENTS = {
    "comparable_marked_relevant",
    "comparable_saved",
}
EXPLICIT_NEGATIVE_EVENTS = {
    "comparable_marked_not_relevant",
}
WEAK_POSITIVE_EVENTS = {
    "comparable_opened",
    "comparable_details_opened",
}
REQUEST_LEVEL_FEEDBACK_EVENTS = {
    "comparable_no_good_options",
}


@dataclass(frozen=True)
class CandidateLabelResolution:
    state: str
    priority: int
    conflict: bool = False


@dataclass(frozen=True)
class FeedbackTrainingSupportSummary:
    objective_effective: str
    request_count: int
    served_request_count: int
    explicit_positive_event_count: int
    explicit_negative_event_count: int
    explicit_positive_request_count: int
    explicit_negative_request_count: int
    no_good_option_request_count: int
    weak_positive_request_count: int
    conflict_request_count: int
    trainable_request_count: int
    skipped_unserved_feedback_event_count: int
    filtered_by_rank_feedback_event_count: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "objective_effective": self.objective_effective,
            "request_count": int(self.request_count),
            "served_request_count": int(self.served_request_count),
            "explicit_positive_event_count": int(self.explicit_positive_event_count),
            "explicit_negative_event_count": int(self.explicit_negative_event_count),
            "explicit_positive_request_count": int(self.explicit_positive_request_count),
            "explicit_negative_request_count": int(self.explicit_negative_request_count),
            "no_good_option_request_count": int(self.no_good_option_request_count),
            "weak_positive_request_count": int(self.weak_positive_request_count),
            "conflict_request_count": int(self.conflict_request_count),
            "trainable_request_count": int(self.trainable_request_count),
            "skipped_unserved_feedback_event_count": int(
                self.skipped_unserved_feedback_event_count
            ),
            "filtered_by_rank_feedback_event_count": int(
                self.filtered_by_rank_feedback_event_count
            ),
        }


@dataclass(frozen=True)
class _RequestFeedbackView:
    request_id: str
    objective_effective: str
    served: Dict[str, Dict[str, Any]]
    labeled: Dict[str, CandidateLabelResolution]
    explicit_positive_event_count: int
    explicit_negative_event_count: int
    has_no_good_options: bool
    has_weak_positive_only: bool
    conflict_present: bool
    skipped_unserved_feedback_event_count: int
    filtered_by_rank_feedback_event_count: int


def resolve_candidate_feedback_state(events: Sequence[Dict[str, Any]]) -> CandidateLabelResolution:
    seen_positive = False
    seen_negative = False
    seen_weak_positive = False
    saved = False
    relevant = False
    for event in events:
        event_name = str(event.get("event_name") or "").strip()
        if event_name == "comparable_saved":
            saved = True
            seen_positive = True
        elif event_name == "comparable_marked_relevant":
            relevant = True
            seen_positive = True
        elif event_name == "comparable_marked_not_relevant":
            seen_negative = True
        elif event_name in WEAK_POSITIVE_EVENTS:
            seen_weak_positive = True
    if seen_positive and seen_negative:
        return CandidateLabelResolution(state="conflict", priority=-1, conflict=True)
    if saved:
        return CandidateLabelResolution(state="saved", priority=3)
    if relevant:
        return CandidateLabelResolution(state="relevant", priority=2)
    if seen_negative:
        return CandidateLabelResolution(state="not_relevant", priority=0)
    if seen_weak_positive:
        return CandidateLabelResolution(state="weak_positive", priority=1)
    return CandidateLabelResolution(state="unlabeled", priority=1)


def _build_request_feedback_views(
    *,
    requests: Sequence[Dict[str, Any]],
    served_outputs: Sequence[Dict[str, Any]],
    feedback_events: Sequence[Dict[str, Any]],
    objectives: Optional[Sequence[str]] = None,
    max_served_rank: Optional[int] = None,
) -> List[_RequestFeedbackView]:
    objective_scope = {str(item) for item in list(objectives or []) if str(item).strip()}
    requests_by_id = {
        str(row.get("request_id") or ""): row
        for row in requests
        if str(row.get("request_id") or "").strip()
    }

    all_served_by_request: Dict[str, Dict[str, Dict[str, Any]]] = {}
    served_by_request: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in served_outputs:
        request_id = str(row.get("request_id") or "").strip()
        candidate_id = str(row.get("candidate_id") or "").strip()
        metadata = row.get("metadata")
        if not request_id or not candidate_id or not isinstance(metadata, dict):
            continue
        rank_raw = row.get("rank")
        try:
            served_rank = int(rank_raw) if rank_raw is not None else None
        except (TypeError, ValueError):
            served_rank = None
        served_metadata = dict(metadata)
        if served_rank is not None:
            served_metadata.setdefault("served_rank", served_rank)
            served_metadata.setdefault("visible_position", served_rank)
        served_metadata.setdefault("was_exposed", True)
        all_served_by_request.setdefault(request_id, {})[candidate_id] = served_metadata
        if max_served_rank is not None and served_rank is not None and served_rank > int(max_served_rank):
            continue
        served_by_request.setdefault(request_id, {})[candidate_id] = served_metadata

    comparable_feedback_by_request_candidate: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    request_level_feedback_flags: Dict[str, Dict[str, Any]] = {}
    for row in feedback_events:
        request_id = str(row.get("request_id") or "").strip()
        if not request_id:
            continue
        event_name = str(row.get("event_name") or "").strip()
        entity_type = str(row.get("entity_type") or "").strip()
        if event_name in REQUEST_LEVEL_FEEDBACK_EVENTS:
            request_level_feedback_flags.setdefault(request_id, {})[event_name] = True
            continue
        if entity_type != "comparable":
            continue
        candidate_id = str(row.get("entity_id") or "").strip()
        if not candidate_id:
            continue
        comparable_feedback_by_request_candidate.setdefault(request_id, {}).setdefault(
            candidate_id, []
        ).append(row)

    request_views: List[_RequestFeedbackView] = []
    for request_id, request_row in requests_by_id.items():
        objective_effective = str(request_row.get("objective_effective") or "").strip()
        if objective_scope and objective_effective not in objective_scope:
            continue
        served = served_by_request.get(request_id) or {}
        comparable_feedback = comparable_feedback_by_request_candidate.get(request_id) or {}
        if not served and not comparable_feedback:
            continue

        labeled: Dict[str, CandidateLabelResolution] = {}
        explicit_positive_event_count = 0
        explicit_negative_event_count = 0
        skipped_unserved_feedback_event_count = 0
        filtered_by_rank_feedback_event_count = 0
        conflict_present = False

        for candidate_id, events in comparable_feedback.items():
            if candidate_id in (all_served_by_request.get(request_id) or {}) and candidate_id not in served:
                filtered_by_rank_feedback_event_count += len(events)
                continue
            if candidate_id not in served:
                skipped_unserved_feedback_event_count += len(events)
                continue
            resolution = resolve_candidate_feedback_state(events)
            if resolution.conflict:
                conflict_present = True
                continue
            for event in events:
                event_name = str(event.get("event_name") or "").strip()
                if event_name in EXPLICIT_POSITIVE_EVENTS:
                    explicit_positive_event_count += 1
                elif event_name in EXPLICIT_NEGATIVE_EVENTS:
                    explicit_negative_event_count += 1
            labeled[candidate_id] = resolution

        request_flags = request_level_feedback_flags.get(request_id) or {}
        request_views.append(
            _RequestFeedbackView(
                request_id=request_id,
                objective_effective=objective_effective,
                served=served,
                labeled=labeled,
                explicit_positive_event_count=explicit_positive_event_count,
                explicit_negative_event_count=explicit_negative_event_count,
                has_no_good_options=bool(request_flags.get("comparable_no_good_options")),
                has_weak_positive_only=any(
                    resolution.state == "weak_positive" for resolution in labeled.values()
                ),
                conflict_present=conflict_present,
                skipped_unserved_feedback_event_count=skipped_unserved_feedback_event_count,
                filtered_by_rank_feedback_event_count=filtered_by_rank_feedback_event_count,
            )
        )

    return request_views


def summarize_feedback_training_support(
    *,
    requests: Sequence[Dict[str, Any]],
    served_outputs: Sequence[Dict[str, Any]],
    feedback_events: Sequence[Dict[str, Any]],
    objectives: Optional[Sequence[str]] = None,
    max_served_rank: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    summaries: Dict[str, Dict[str, int]] = {}
    request_views = _build_request_feedback_views(
        requests=requests,
        served_outputs=served_outputs,
        feedback_events=feedback_events,
        objectives=objectives,
        max_served_rank=max_served_rank,
    )
    request_counts_by_objective: Dict[str, int] = {}
    served_counts_by_objective: Dict[str, int] = {}
    for row in requests:
        request_id = str(row.get("request_id") or "").strip()
        objective_effective = str(row.get("objective_effective") or "").strip()
        if not request_id or not objective_effective:
            continue
        if objectives and objective_effective not in {str(item) for item in objectives}:
            continue
        request_counts_by_objective[objective_effective] = (
            request_counts_by_objective.get(objective_effective, 0) + 1
        )
    for view in request_views:
        served_counts_by_objective[view.objective_effective] = (
            served_counts_by_objective.get(view.objective_effective, 0) + 1
        )
        summary = summaries.setdefault(
            view.objective_effective,
            {
                "explicit_positive_event_count": 0,
                "explicit_negative_event_count": 0,
                "explicit_positive_request_count": 0,
                "explicit_negative_request_count": 0,
                "no_good_option_request_count": 0,
                "weak_positive_request_count": 0,
                "conflict_request_count": 0,
                "trainable_request_count": 0,
                "skipped_unserved_feedback_event_count": 0,
                "filtered_by_rank_feedback_event_count": 0,
            },
        )
        has_positive = any(
            resolution.state in {"saved", "relevant"} for resolution in view.labeled.values()
        )
        has_negative = any(
            resolution.state == "not_relevant" for resolution in view.labeled.values()
        )
        if has_positive:
            summary["explicit_positive_request_count"] += 1
        if has_negative:
            summary["explicit_negative_request_count"] += 1
        if view.has_no_good_options:
            summary["no_good_option_request_count"] += 1
        if view.has_weak_positive_only:
            summary["weak_positive_request_count"] += 1
        if view.conflict_present:
            summary["conflict_request_count"] += 1
        if has_positive and has_negative:
            summary["trainable_request_count"] += 1
        summary["explicit_positive_event_count"] += int(view.explicit_positive_event_count)
        summary["explicit_negative_event_count"] += int(view.explicit_negative_event_count)
        summary["skipped_unserved_feedback_event_count"] += int(
            view.skipped_unserved_feedback_event_count
        )
        summary["filtered_by_rank_feedback_event_count"] += int(
            view.filtered_by_rank_feedback_event_count
        )

    out: Dict[str, Dict[str, Any]] = {}
    for objective, request_count in request_counts_by_objective.items():
        summary = summaries.get(objective) or {}
        out[objective] = FeedbackTrainingSupportSummary(
            objective_effective=objective,
            request_count=request_count,
            served_request_count=served_counts_by_objective.get(objective, 0),
            explicit_positive_event_count=int(summary.get("explicit_positive_event_count", 0)),
            explicit_negative_event_count=int(summary.get("explicit_negative_event_count", 0)),
            explicit_positive_request_count=int(summary.get("explicit_positive_request_count", 0)),
            explicit_negative_request_count=int(summary.get("explicit_negative_request_count", 0)),
            no_good_option_request_count=int(summary.get("no_good_option_request_count", 0)),
            weak_positive_request_count=int(summary.get("weak_positive_request_count", 0)),
            conflict_request_count=int(summary.get("conflict_request_count", 0)),
            trainable_request_count=int(summary.get("trainable_request_count", 0)),
            skipped_unserved_feedback_event_count=int(
                summary.get("skipped_unserved_feedback_event_count", 0)
            ),
            filtered_by_rank_feedback_event_count=int(
                summary.get("filtered_by_rank_feedback_event_count", 0)
            ),
        ).as_dict()
    return out


def materialize_pairwise_rows(
    *,
    requests: Sequence[Dict[str, Any]],
    served_outputs: Sequence[Dict[str, Any]],
    feedback_events: Sequence[Dict[str, Any]],
    objectives: Optional[Sequence[str]] = None,
    include_saved_vs_relevant: bool = True,
    max_served_rank: Optional[int] = None,
) -> List[PairwiseTrainingRow]:
    out: List[PairwiseTrainingRow] = []
    request_views = _build_request_feedback_views(
        requests=requests,
        served_outputs=served_outputs,
        feedback_events=feedback_events,
        objectives=objectives,
        max_served_rank=max_served_rank,
    )
    for view in request_views:
        served = view.served
        labeled = view.labeled

        positives = [
            candidate_id
            for candidate_id, resolution in labeled.items()
            if resolution.state in {"saved", "relevant"}
        ]
        negatives = [
            candidate_id
            for candidate_id, resolution in labeled.items()
            if resolution.state == "not_relevant"
        ]

        for positive_id in positives:
            positive_resolution = labeled[positive_id]
            positive_features = candidate_feature_payload_from_item(served[positive_id])
            for negative_id in negatives:
                negative_features = candidate_feature_payload_from_item(served[negative_id])
                pair_source = (
                    "saved_vs_negative"
                    if positive_resolution.state == "saved"
                    else "positive_vs_negative"
                )
                pair_weight = 1.20 if positive_resolution.state == "saved" else 1.0
                out.append(
                    PairwiseTrainingRow(
                        request_id=view.request_id,
                        objective_effective=view.objective_effective,
                        candidate_a_id=positive_id,
                        candidate_b_id=negative_id,
                        label=1,
                        pair_source=pair_source,
                        pair_weight=pair_weight,
                        features_a=positive_features,
                        features_b=negative_features,
                    )
                )
                out.append(
                    PairwiseTrainingRow(
                        request_id=view.request_id,
                        objective_effective=view.objective_effective,
                        candidate_a_id=negative_id,
                        candidate_b_id=positive_id,
                        label=0,
                        pair_source=f"{pair_source}_reverse",
                        pair_weight=pair_weight,
                        features_a=negative_features,
                        features_b=positive_features,
                    )
                )

        if include_saved_vs_relevant:
            saved_ids = [
                candidate_id
                for candidate_id, resolution in labeled.items()
                if resolution.state == "saved"
            ]
            relevant_ids = [
                candidate_id
                for candidate_id, resolution in labeled.items()
                if resolution.state == "relevant"
            ]
            for saved_id in saved_ids:
                saved_features = candidate_feature_payload_from_item(served[saved_id])
                for relevant_id in relevant_ids:
                    relevant_features = candidate_feature_payload_from_item(served[relevant_id])
                    out.append(
                        PairwiseTrainingRow(
                            request_id=view.request_id,
                            objective_effective=view.objective_effective,
                            candidate_a_id=saved_id,
                            candidate_b_id=relevant_id,
                            label=1,
                            pair_source="saved_vs_relevant",
                            pair_weight=0.65,
                            features_a=saved_features,
                            features_b=relevant_features,
                        )
                    )
                    out.append(
                        PairwiseTrainingRow(
                            request_id=view.request_id,
                            objective_effective=view.objective_effective,
                            candidate_a_id=relevant_id,
                            candidate_b_id=saved_id,
                            label=0,
                            pair_source="saved_vs_relevant_reverse",
                            pair_weight=0.65,
                            features_a=relevant_features,
                            features_b=saved_features,
                        )
                    )

    return out


def group_rows_by_objective(
    rows: Iterable[PairwiseTrainingRow],
) -> Dict[str, List[PairwiseTrainingRow]]:
    out: Dict[str, List[PairwiseTrainingRow]] = {}
    for row in rows:
        out.setdefault(row.objective_effective, []).append(row)
    return out
