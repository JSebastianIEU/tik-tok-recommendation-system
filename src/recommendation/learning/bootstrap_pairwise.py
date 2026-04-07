from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..contracts import CanonicalDatasetBundle, load_bundle_from_manifest
from .inference import RecommenderRuntime
from .learned_reranker import PairwiseTrainingRow, candidate_feature_payload_from_item
from .ranking_baseline import rank_shortlist
from .retrieval_baseline import retrieve_shortlist
from .temporal import parse_dt


DEFAULT_BOOTSTRAP_ALLOWED_SPLITS = ("train", "validation")


def _normalize_dt(value: Any) -> datetime:
    parsed = parse_dt(value)
    if parsed is not None:
        return parsed
    return datetime.now(timezone.utc)


def _as_str_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for item in values:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _source_bundle_from_datamart(
    datamart: Dict[str, Any],
    *,
    source_bundle: Optional[CanonicalDatasetBundle],
) -> CanonicalDatasetBundle:
    if source_bundle is not None:
        return source_bundle
    manifest_ref = str(datamart.get("source_manifest_path") or "").strip()
    if not manifest_ref:
        raise ValueError(
            "bootstrap_datamart_missing_source_manifest: datamart.source_manifest_path is required "
            "unless source_bundle is provided explicitly."
        )
    return load_bundle_from_manifest(manifest_ref)


def _query_payload_from_bundle_row(
    bundle_row: Dict[str, Any],
    *,
    video: Any,
) -> Dict[str, Any]:
    keywords = _as_str_list(video.keywords)
    if str(bundle_row.get("topic_key") or "").strip() and str(bundle_row.get("topic_key")) not in keywords:
        keywords.append(str(bundle_row.get("topic_key")))
    return {
        "query_id": str(bundle_row.get("row_id") or video.video_id),
        "video_id": str(video.video_id),
        "text": str(video.caption or ""),
        "description": str(video.caption or ""),
        "hashtags": list(video.hashtags or []),
        "keywords": keywords,
        "author_id": str(video.author_id or ""),
        "language": str(video.language or "") or None,
        "as_of_time": bundle_row.get("as_of_time"),
    }


def _candidate_payload_from_bundle_row(
    bundle_row: Dict[str, Any],
    *,
    video: Any,
) -> Dict[str, Any]:
    features = bundle_row.get("features") if isinstance(bundle_row.get("features"), dict) else {}
    signal_hints: Dict[str, Any] = {}
    comment_intelligence = (
        features.get("comment_intelligence") if isinstance(features.get("comment_intelligence"), dict) else {}
    )
    if comment_intelligence:
        signal_hints["comment_intelligence"] = comment_intelligence
    keywords = _as_str_list(video.keywords)
    if str(bundle_row.get("topic_key") or "").strip() and str(bundle_row.get("topic_key")) not in keywords:
        keywords.append(str(bundle_row.get("topic_key")))
    payload: Dict[str, Any] = {
        "candidate_id": str(video.video_id),
        "video_id": str(video.video_id),
        "video_url": str(video.video_url) if video.video_url else "",
        "text": str(video.caption or ""),
        "caption": str(video.caption or ""),
        "hashtags": list(video.hashtags or []),
        "keywords": keywords,
        "posted_at": video.posted_at,
        "as_of_time": bundle_row.get("as_of_time"),
        "author_id": str(video.author_id or ""),
        "language": str(video.language or "") or None,
    }
    if signal_hints:
        payload["signal_hints"] = signal_hints
    return payload


def _bootstrap_pair_specs(
    *,
    labels_by_candidate: Dict[str, int],
    include_neutral_pairs: bool,
) -> List[Tuple[str, str, int, int, float, str]]:
    out: List[Tuple[str, str, int, int, float, str]] = []
    candidate_ids = sorted(labels_by_candidate.keys())
    for left_id in candidate_ids:
        left_label = int(labels_by_candidate[left_id])
        for right_id in candidate_ids:
            if left_id == right_id:
                continue
            right_label = int(labels_by_candidate[right_id])
            if left_label <= right_label:
                continue
            if left_label < 2:
                continue
            if right_label not in {0, 1}:
                continue
            if right_label == 1 and not include_neutral_pairs:
                continue
            gap = left_label - right_label
            if gap <= 0:
                continue
            base_weight = {
                (3, 0): 0.90,
                (2, 0): 0.70,
                (3, 1): 0.55,
                (2, 1): 0.40,
            }.get((left_label, right_label), 0.35)
            out.append(
                (
                    left_id,
                    right_id,
                    left_label,
                    right_label,
                    base_weight,
                    f"bootstrap_rel{left_label}_vs_rel{right_label}",
                )
            )
    return out


def materialize_datamart_bootstrap_rows(
    *,
    datamart: Dict[str, Any],
    bundle_dir: Path | str,
    source_bundle: Optional[CanonicalDatasetBundle] = None,
    objectives: Optional[Sequence[str]] = None,
    target_source: Optional[str] = None,
    include_neutral_pairs: bool = False,
    allowed_splits: Sequence[str] = DEFAULT_BOOTSTRAP_ALLOWED_SPLITS,
) -> List[PairwiseTrainingRow]:
    runtime = RecommenderRuntime(bundle_dir=Path(bundle_dir))
    source = _source_bundle_from_datamart(datamart, source_bundle=source_bundle)
    rows = list(datamart.get("rows") or [])
    pair_rows = list(datamart.get("pair_rows") or [])
    if not rows or not pair_rows:
        return []

    objective_scope = {str(item).strip() for item in list(objectives or []) if str(item).strip()}
    split_scope = {str(item).strip() for item in list(allowed_splits or []) if str(item).strip()}
    configured_target_source = str(
        target_source
        or ((datamart.get("config") or {}).get("pair_target_source"))
        or "scalar_v1"
    ).strip()

    rows_by_id = {
        str(row.get("row_id") or ""): row
        for row in rows
        if str(row.get("row_id") or "").strip()
    }
    videos_by_id = {str(video.video_id): video for video in list(source.videos or [])}

    pair_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for pair in pair_rows:
        objective = str(pair.get("objective") or "").strip()
        query_row_id = str(pair.get("query_row_id") or "").strip()
        if not objective or not query_row_id:
            continue
        if objective_scope and objective not in objective_scope:
            continue
        if str(pair.get("target_source") or "scalar_v1").strip() != configured_target_source:
            continue
        query_row = rows_by_id.get(query_row_id)
        candidate_row = rows_by_id.get(str(pair.get("candidate_row_id") or "").strip())
        if query_row is None or candidate_row is None:
            continue
        if split_scope and (
            str(query_row.get("split") or "").strip() not in split_scope
            or str(candidate_row.get("split") or "").strip() not in split_scope
        ):
            continue
        pair_groups[(objective, query_row_id)].append(pair)

    out: List[PairwiseTrainingRow] = []
    for (objective, query_row_id), query_pairs in pair_groups.items():
        query_row = rows_by_id.get(query_row_id)
        if query_row is None:
            continue
        query_video = videos_by_id.get(str(query_row.get("video_id") or "").strip())
        if query_video is None:
            continue
        query_payload = _query_payload_from_bundle_row(query_row, video=query_video)
        query_profile = runtime._build_query_profile(
            objective=objective,
            query=query_payload,
            fallback_language=str(query_video.language or "") or None,
            fallback_locale=None,
            fallback_content_type=None,
        )
        query_as_of = _normalize_dt(query_row.get("as_of_time"))

        prepared_candidates: List[Dict[str, Any]] = []
        labels_by_candidate: Dict[str, int] = {}
        for pair in query_pairs:
            candidate_row = rows_by_id.get(str(pair.get("candidate_row_id") or "").strip())
            if candidate_row is None:
                continue
            candidate_video = videos_by_id.get(str(pair.get("candidate_video_id") or "").strip())
            if candidate_video is None:
                continue
            candidate_payload = _candidate_payload_from_bundle_row(candidate_row, video=candidate_video)
            prepared = runtime._prepare_candidate(
                payload=candidate_payload,
                as_of=query_as_of,
                query_profile=query_profile,
            )
            if prepared is None:
                continue
            candidate_id = str(prepared.get("candidate_id") or "").strip()
            if not candidate_id:
                continue
            prepared_candidates.append(prepared)
            labels_by_candidate[candidate_id] = int(pair.get("relevance_label") or 0)

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

        pair_specs = _bootstrap_pair_specs(
            labels_by_candidate={
                candidate_id: label
                for candidate_id, label in labels_by_candidate.items()
                if candidate_id in feature_payload_by_candidate
            },
            include_neutral_pairs=include_neutral_pairs,
        )
        request_id = f"bootstrap::{objective}::{query_row_id}"
        for (
            candidate_a_id,
            candidate_b_id,
            left_label,
            right_label,
            pair_weight,
            pair_source,
        ) in pair_specs:
            features_a = feature_payload_by_candidate.get(candidate_a_id)
            features_b = feature_payload_by_candidate.get(candidate_b_id)
            if features_a is None or features_b is None:
                continue
            out.append(
                PairwiseTrainingRow(
                    request_id=request_id,
                    objective_effective=objective,
                    candidate_a_id=candidate_a_id,
                    candidate_b_id=candidate_b_id,
                    label=1,
                    pair_source=pair_source,
                    pair_weight=pair_weight,
                    features_a=features_a,
                    features_b=features_b,
                )
            )
            out.append(
                PairwiseTrainingRow(
                    request_id=request_id,
                    objective_effective=objective,
                    candidate_a_id=candidate_b_id,
                    candidate_b_id=candidate_a_id,
                    label=0,
                    pair_source=f"{pair_source}_reverse",
                    pair_weight=pair_weight,
                    features_a=features_b,
                    features_b=features_a,
                )
            )

    return out

