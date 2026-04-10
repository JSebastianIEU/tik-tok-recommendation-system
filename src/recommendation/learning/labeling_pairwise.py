from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .inference import RecommenderRuntime
from .learned_reranker import PairwiseTrainingRow, candidate_feature_payload_from_item
from .ranking_baseline import rank_shortlist
from .retrieval_baseline import retrieve_shortlist
from .temporal import parse_dt


LABELING_POSITIVE_STATES = {"saved", "relevant"}
LABELING_NEGATIVE_STATE = "not_relevant"
LABELING_ALLOWED_STATES = LABELING_POSITIVE_STATES | {LABELING_NEGATIVE_STATE}


def _normalize_label(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    return text if text in LABELING_ALLOWED_STATES else None


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _load_session_payload(session_json: Path | str) -> Dict[str, Any]:
    path = Path(session_json)
    return json.loads(path.read_text(encoding="utf-8"))


def _query_as_of_time(case_payload: Dict[str, Any]) -> Optional[str]:
    query = _as_dict(case_payload.get("query"))
    query_payload = _as_dict(query.get("query_payload"))
    display = _as_dict(query.get("display"))
    raw = (
        query_payload.get("as_of_time")
        or query_payload.get("created_at")
        or display.get("created_at")
    )
    parsed = parse_dt(raw)
    return parsed.isoformat() if parsed is not None else None


def _pair_rows_from_labels(
    *,
    request_id: str,
    objective: str,
    labels_by_candidate: Dict[str, str],
    feature_payload_by_candidate: Dict[str, Dict[str, float]],
    include_saved_vs_relevant: bool,
) -> List[PairwiseTrainingRow]:
    out: List[PairwiseTrainingRow] = []
    positives = [
        candidate_id
        for candidate_id, label in labels_by_candidate.items()
        if label in LABELING_POSITIVE_STATES and candidate_id in feature_payload_by_candidate
    ]
    negatives = [
        candidate_id
        for candidate_id, label in labels_by_candidate.items()
        if label == LABELING_NEGATIVE_STATE and candidate_id in feature_payload_by_candidate
    ]

    for positive_id in positives:
        positive_label = labels_by_candidate[positive_id]
        positive_features = feature_payload_by_candidate[positive_id]
        for negative_id in negatives:
            negative_features = feature_payload_by_candidate[negative_id]
            pair_source = (
                "labeling_saved_vs_not_relevant"
                if positive_label == "saved"
                else "labeling_relevant_vs_not_relevant"
            )
            pair_weight = 1.25 if positive_label == "saved" else 1.0
            out.append(
                PairwiseTrainingRow(
                    request_id=request_id,
                    objective_effective=objective,
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
                    request_id=request_id,
                    objective_effective=objective,
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
            for candidate_id, label in labels_by_candidate.items()
            if label == "saved" and candidate_id in feature_payload_by_candidate
        ]
        relevant_ids = [
            candidate_id
            for candidate_id, label in labels_by_candidate.items()
            if label == "relevant" and candidate_id in feature_payload_by_candidate
        ]
        for saved_id in saved_ids:
            saved_features = feature_payload_by_candidate[saved_id]
            for relevant_id in relevant_ids:
                relevant_features = feature_payload_by_candidate[relevant_id]
                out.append(
                    PairwiseTrainingRow(
                        request_id=request_id,
                        objective_effective=objective,
                        candidate_a_id=saved_id,
                        candidate_b_id=relevant_id,
                        label=1,
                        pair_source="labeling_saved_vs_relevant",
                        pair_weight=0.7,
                        features_a=saved_features,
                        features_b=relevant_features,
                    )
                )
                out.append(
                    PairwiseTrainingRow(
                        request_id=request_id,
                        objective_effective=objective,
                        candidate_a_id=relevant_id,
                        candidate_b_id=saved_id,
                        label=0,
                        pair_source="labeling_saved_vs_relevant_reverse",
                        pair_weight=0.7,
                        features_a=relevant_features,
                        features_b=saved_features,
                    )
                )
    return out


def materialize_labeling_session_rows(
    *,
    session_json_paths: Sequence[Path | str],
    bundle_dir: Path | str,
    objectives: Optional[Sequence[str]] = None,
    include_saved_vs_relevant: bool = True,
) -> List[PairwiseTrainingRow]:
    runtime = RecommenderRuntime(bundle_dir=Path(bundle_dir))
    objective_scope = {str(item).strip() for item in list(objectives or []) if str(item).strip()}
    out: List[PairwiseTrainingRow] = []

    for session_json_path in session_json_paths:
        session_payload = _load_session_payload(session_json_path)
        session_id = str(session_payload.get("session_id") or Path(session_json_path).stem).strip()
        for case_payload in list(session_payload.get("cases") or []):
            case = _as_dict(case_payload)
            objective = str(case.get("objective") or "").strip()
            if not objective:
                continue
            if objective_scope and objective not in objective_scope:
                continue

            query = _as_dict(case.get("query"))
            query_payload = _as_dict(query.get("query_payload"))
            as_of_time = _query_as_of_time(case)
            if not query_payload or not as_of_time:
                continue

            query_profile = runtime._build_query_profile(
                objective=objective,
                query=query_payload,
                fallback_language=None,
                fallback_locale=None,
                fallback_content_type=None,
            )
            as_of = parse_dt(as_of_time)
            if as_of is None:
                continue

            prepared_candidates: List[Dict[str, Any]] = []
            labels_by_candidate: Dict[str, str] = {}
            for candidate_payload in list(case.get("candidates") or []):
                candidate = _as_dict(candidate_payload)
                candidate_id = str(candidate.get("candidate_id") or "").strip()
                runtime_payload = _as_dict(candidate.get("candidate_payload"))
                label = _normalize_label(_as_dict(candidate.get("review")).get("label"))
                if not candidate_id or not runtime_payload:
                    continue
                prepared = runtime._prepare_candidate(
                    payload=runtime_payload,
                    as_of=as_of,
                    query_profile=query_profile,
                )
                if prepared is None:
                    continue
                prepared_candidates.append(prepared)
                if label is not None:
                    labels_by_candidate[str(prepared.get("candidate_id") or candidate_id)] = label

            usable_candidates = [
                item
                for item in prepared_candidates
                if str(item.get("support_level") or "") != "low"
            ]
            if len(usable_candidates) < 2:
                continue

            shortlist, _ = retrieve_shortlist(
                usable_candidates=usable_candidates,
                query_profile=query_profile,
                retrieve_k=len(usable_candidates),
            )
            ranked, _ = rank_shortlist(
                shortlist=shortlist,
                query_profile=query_profile,
                effective_objective=objective,
                portfolio=None,
                rankers_available=runtime.rankers.keys(),
            )
            feature_payload_by_candidate = {
                str(item.get("candidate_id") or ""): candidate_feature_payload_from_item(item)
                for item in ranked
                if str(item.get("candidate_id") or "").strip()
            }
            if len(feature_payload_by_candidate) < 2:
                continue

            request_id = f"labeling::{session_id}::{str(case.get('case_id') or '').strip()}"
            out.extend(
                _pair_rows_from_labels(
                    request_id=request_id,
                    objective=objective,
                    labels_by_candidate=labels_by_candidate,
                    feature_payload_by_candidate=feature_payload_by_candidate,
                    include_saved_vs_relevant=include_saved_vs_relevant,
                )
            )

    return out

