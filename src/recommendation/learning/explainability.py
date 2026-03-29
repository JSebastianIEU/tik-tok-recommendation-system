from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .calibration import ObjectiveSegmentCalibrators
from .ranker import (
    FEATURE_NAMES,
    SEGMENT_GLOBAL,
    RankerEnsembleModel,
    RankerFamilyModel,
    pair_feature_vector_array,
)
from .temporal import parse_dt


EXPLAINABILITY_VERSION = "explainability.v2"
_NORMAL_Z = 1.2815515655446004


@dataclass
class FeatureContributionRow:
    feature: str
    contribution: float
    value: float


@dataclass
class FeatureContributionCard:
    method: str
    fallback_used: bool
    input_trace_id: str
    top_positive: List[FeatureContributionRow]
    top_negative: List[FeatureContributionRow]


@dataclass
class NeighborEvidenceEntry:
    candidate_id: str
    similarity: float
    score: float
    rank: int
    feature_gaps: List[Dict[str, float]]


@dataclass
class NeighborEvidenceCard:
    method: str
    temporal_safe: bool
    winner_count: int
    loser_count: int
    winners: List[NeighborEvidenceEntry]
    losers: List[NeighborEvidenceEntry]


@dataclass
class TemporalConfidenceBand:
    p10: float
    p50: float
    p90: float
    source: str


@dataclass
class CounterfactualScenarioResult:
    scenario_id: str
    expected_rank_delta_band: TemporalConfidenceBand
    feasibility: str
    reason: str
    applied_deltas: Dict[str, float]
    trace: Dict[str, Any]


@dataclass
class ExplainabilityControls:
    enabled: bool = False
    top_features: int = 5
    neighbor_k: int = 3
    run_counterfactuals: bool = False

    @classmethod
    def from_payload(cls, payload: Optional[Dict[str, Any]]) -> "ExplainabilityControls":
        source = payload if isinstance(payload, dict) else {}
        top_features = int(source.get("top_features") or 5)
        neighbor_k = int(source.get("neighbor_k") or 3)
        return cls(
            enabled=bool(source.get("enabled", False)),
            top_features=max(1, min(top_features, 12)),
            neighbor_k=max(1, min(neighbor_k, 6)),
            run_counterfactuals=bool(source.get("run_counterfactuals", False)),
        )


@dataclass(frozen=True)
class _ScenarioDef:
    scenario_id: str
    deltas_std: Dict[str, float]
    reason: str


_SCENARIOS: Tuple[_ScenarioDef, ...] = (
    _ScenarioDef("hook_clarity_plus_1sd", {"keyword_count": 1.0}, "proxy_keyword_density"),
    _ScenarioDef("step_density_plus_0_5sd", {"keyword_count": 0.5}, "proxy_instruction_density"),
    _ScenarioDef("cta_keyword_count_plus_1sd", {"keyword_count": 1.0}, "direct_keyword_boost"),
    _ScenarioDef("caption_compactness_plus_0_5sd", {"caption_word_count": -0.5}, "shorter_caption_proxy"),
    _ScenarioDef("hashtag_specificity_plus_0_5sd", {"hashtag_count": -0.5}, "specificity_proxy"),
    _ScenarioDef(
        "comment_confusion_minus_0_5sd",
        {"comment_confusion_index": -0.5},
        "direct_comment_signal",
    ),
    _ScenarioDef(
        "comment_help_plus_0_5sd",
        {"comment_help_seeking_index": 0.5},
        "direct_comment_signal",
    ),
    _ScenarioDef("payoff_timing_earlier_0_5sd", {"caption_word_count": -0.25}, "timing_proxy"),
    _ScenarioDef("pacing_score_plus_0_5sd", {"hashtag_count": 0.25, "keyword_count": 0.25}, "pacing_proxy"),
    _ScenarioDef(
        "shareability_proxy_plus_0_5sd",
        {"hashtag_count": 0.3, "keyword_count": 0.2},
        "shareability_proxy",
    ),
)


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _round(value: float) -> float:
    return float(round(float(value), 6))


def _to_iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _pair_vec(query_row: Dict[str, Any], candidate_row: Dict[str, Any], similarity: float) -> np.ndarray:
    return pair_feature_vector_array(
        query_row=query_row,
        candidate_row=candidate_row,
        similarity=float(similarity),
    )


def _ensemble_contrib(
    ensemble: RankerEnsembleModel,
    pair_vec: np.ndarray,
) -> Optional[np.ndarray]:
    contribs: List[np.ndarray] = []
    for member in ensemble.models:
        backend = str(getattr(member, "backend", ""))
        if "lightgbm" not in backend:
            return None
        model = getattr(member, "model", None)
        if model is None:
            return None
        try:
            values = np.asarray(model.predict(pair_vec, pred_contrib=True), dtype=np.float64)
        except Exception:
            return None
        if values.ndim == 1:
            vec = values
        else:
            vec = values[0]
        if vec.shape[0] >= len(FEATURE_NAMES):
            contribs.append(vec[: len(FEATURE_NAMES)].astype(np.float64))
    if not contribs:
        return None
    matrix = np.vstack(contribs)
    return np.mean(matrix, axis=0).astype(np.float64)


def _score_from_vec(
    *,
    ranker: RankerFamilyModel,
    pair_vec: np.ndarray,
    selected_ranker_id: str,
) -> float:
    global_mean_arr, global_std_arr = ranker.global_ensemble.predict_stats(pair_vec)
    global_mean = float(global_mean_arr[0])
    global_std = float(global_std_arr[0])

    if (
        selected_ranker_id != SEGMENT_GLOBAL
        and selected_ranker_id in ranker.promoted_segments
        and selected_ranker_id in ranker.segment_ensembles
    ):
        segment_ensemble = ranker.segment_ensembles[selected_ranker_id]
        seg_mean_arr, seg_std_arr = segment_ensemble.predict_stats(pair_vec)
        seg_mean = float(seg_mean_arr[0])
        seg_std = float(seg_std_arr[0])
        blend_weight = _clip(1.0 - (seg_std / max(1e-6, ranker.std_ref)), 0.2, 1.0)
        return float((blend_weight * seg_mean) + ((1.0 - blend_weight) * global_mean))

    _ = global_std
    return float(global_mean)


def _local_perturbation_contrib(
    *,
    ranker: RankerFamilyModel,
    pair_vec: np.ndarray,
    selected_ranker_id: str,
) -> np.ndarray:
    base = _score_from_vec(
        ranker=ranker,
        pair_vec=pair_vec,
        selected_ranker_id=selected_ranker_id,
    )
    del base
    out = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    base_vec = np.asarray(pair_vec, dtype=np.float64)[0]
    for idx in range(len(FEATURE_NAMES)):
        epsilon = max(0.05, abs(base_vec[idx]) * 0.10)
        plus = np.asarray(base_vec, dtype=np.float64)
        minus = np.asarray(base_vec, dtype=np.float64)
        plus[idx] += epsilon
        minus[idx] -= epsilon
        plus_score = _score_from_vec(
            ranker=ranker,
            pair_vec=np.asarray([plus], dtype=np.float32),
            selected_ranker_id=selected_ranker_id,
        )
        minus_score = _score_from_vec(
            ranker=ranker,
            pair_vec=np.asarray([minus], dtype=np.float32),
            selected_ranker_id=selected_ranker_id,
        )
        out[idx] = (plus_score - minus_score) / 2.0
    return out


def feature_contribution_card(
    *,
    ranker: RankerFamilyModel,
    query_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
    similarity: float,
    selected_ranker_id: str,
    segment_blend_weight: float,
    top_features: int,
    input_trace_id: str,
) -> FeatureContributionCard:
    pair_vec = _pair_vec(query_row, candidate_row, similarity)
    method = "local_perturbation"
    fallback_used = True

    global_contrib = _ensemble_contrib(ranker.global_ensemble, pair_vec)
    mixed_contrib: Optional[np.ndarray] = None
    if (
        selected_ranker_id != SEGMENT_GLOBAL
        and selected_ranker_id in ranker.promoted_segments
        and selected_ranker_id in ranker.segment_ensembles
    ):
        seg_contrib = _ensemble_contrib(ranker.segment_ensembles[selected_ranker_id], pair_vec)
        if seg_contrib is not None and global_contrib is not None:
            weight = _clip(float(segment_blend_weight), 0.0, 1.0)
            mixed_contrib = (weight * seg_contrib) + ((1.0 - weight) * global_contrib)
    elif global_contrib is not None:
        mixed_contrib = global_contrib

    if mixed_contrib is not None:
        method = "lightgbm_pred_contrib"
        fallback_used = False
        contrib_values = mixed_contrib
    else:
        contrib_values = _local_perturbation_contrib(
            ranker=ranker,
            pair_vec=pair_vec,
            selected_ranker_id=selected_ranker_id,
        )

    feature_values = np.asarray(pair_vec, dtype=np.float64)[0]
    rows = [
        FeatureContributionRow(
            feature=name,
            contribution=_round(float(contrib_values[idx])),
            value=_round(float(feature_values[idx])),
        )
        for idx, name in enumerate(FEATURE_NAMES)
    ]
    positive = sorted(
        [row for row in rows if row.contribution > 0.0],
        key=lambda item: item.contribution,
        reverse=True,
    )[: max(1, top_features)]
    negative = sorted(
        [row for row in rows if row.contribution < 0.0],
        key=lambda item: item.contribution,
    )[: max(1, top_features)]
    return FeatureContributionCard(
        method=method,
        fallback_used=fallback_used,
        input_trace_id=input_trace_id,
        top_positive=positive,
        top_negative=negative,
    )


def _candidate_rank_map(items: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    sorted_items = sorted(items, key=lambda item: float(item.get("score_calibrated") or item.get("score") or 0.0), reverse=True)
    out: Dict[str, int] = {}
    for idx, item in enumerate(sorted_items, start=1):
        key = str(item.get("candidate_id") or item.get("candidate_row_id") or "")
        if key:
            out[key] = idx
    return out


def _feature_gap_rows(
    query_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
    top_features: int,
) -> List[Dict[str, float]]:
    q_features = query_row.get("features", {}) if isinstance(query_row.get("features"), dict) else {}
    c_features = candidate_row.get("features", {}) if isinstance(candidate_row.get("features"), dict) else {}
    q_comment = q_features.get("comment_intelligence", {}) if isinstance(q_features.get("comment_intelligence"), dict) else {}
    c_comment = c_features.get("comment_intelligence", {}) if isinstance(c_features.get("comment_intelligence"), dict) else {}
    comparisons = {
        "caption_word_count": (
            float(q_features.get("caption_word_count") or 0.0),
            float(c_features.get("caption_word_count") or 0.0),
        ),
        "hashtag_count": (
            float(q_features.get("hashtag_count") or 0.0),
            float(c_features.get("hashtag_count") or 0.0),
        ),
        "keyword_count": (
            float(q_features.get("keyword_count") or 0.0),
            float(c_features.get("keyword_count") or 0.0),
        ),
        "comment_confusion_index": (
            float(q_comment.get("confusion_index") or 0.0),
            float(c_comment.get("confusion_index") or 0.0),
        ),
        "comment_help_seeking_index": (
            float(q_comment.get("help_seeking_index") or 0.0),
            float(c_comment.get("help_seeking_index") or 0.0),
        ),
        "comment_sentiment_volatility": (
            float(q_comment.get("sentiment_volatility") or 0.0),
            float(c_comment.get("sentiment_volatility") or 0.0),
        ),
    }
    rows = []
    for key, (qv, cv) in comparisons.items():
        rows.append(
            {
                "feature": key,
                "query_value": _round(qv),
                "candidate_value": _round(cv),
                "delta": _round(cv - qv),
                "abs_delta": _round(abs(cv - qv)),
            }
        )
    ranked = sorted(rows, key=lambda item: float(item.get("abs_delta") or 0.0), reverse=True)
    return ranked[: max(1, top_features)]


def neighbor_evidence_card(
    *,
    query_row: Dict[str, Any],
    ranked_items: Sequence[Dict[str, Any]],
    candidate_by_id: Dict[str, Dict[str, Any]],
    query_as_of: datetime,
    neighbor_k: int,
    top_features: int,
) -> NeighborEvidenceCard:
    candidates: List[Dict[str, Any]] = []
    for item in ranked_items:
        candidate_key = str(item.get("candidate_id") or item.get("candidate_row_id") or "")
        candidate_row = candidate_by_id.get(candidate_key) or candidate_by_id.get(
            str(item.get("candidate_row_id") or "")
        )
        if candidate_row is None:
            continue
        candidate_time = parse_dt(candidate_row.get("as_of_time"))
        if candidate_time is None or candidate_time >= query_as_of:
            continue
        with_row = dict(item)
        with_row["_candidate_row"] = candidate_row
        candidates.append(with_row)

    if not candidates:
        return NeighborEvidenceCard(
            method="quartile_similarity",
            temporal_safe=True,
            winner_count=0,
            loser_count=0,
            winners=[],
            losers=[],
        )

    scores = np.asarray(
        [float(item.get("score_calibrated") or item.get("score") or 0.0) for item in candidates],
        dtype=np.float64,
    )
    p75 = float(np.percentile(scores, 75))
    p25 = float(np.percentile(scores, 25))
    rank_map = _candidate_rank_map(candidates)

    winners = [
        item for item in candidates if float(item.get("score_calibrated") or item.get("score") or 0.0) >= p75
    ]
    losers = [
        item for item in candidates if float(item.get("score_calibrated") or item.get("score") or 0.0) <= p25
    ]
    winners.sort(key=lambda item: float((item.get("similarity") or {}).get("fused") or 0.0), reverse=True)
    losers.sort(key=lambda item: float((item.get("similarity") or {}).get("fused") or 0.0), reverse=True)

    def _rows(items: Sequence[Dict[str, Any]]) -> List[NeighborEvidenceEntry]:
        out: List[NeighborEvidenceEntry] = []
        for item in list(items)[: max(1, neighbor_k)]:
            candidate_id = str(item.get("candidate_id") or item.get("candidate_row_id") or "")
            candidate_row = item.get("_candidate_row")
            if not isinstance(candidate_row, dict) or not candidate_id:
                continue
            out.append(
                NeighborEvidenceEntry(
                    candidate_id=candidate_id,
                    similarity=_round(float((item.get("similarity") or {}).get("fused") or 0.0)),
                    score=_round(float(item.get("score_calibrated") or item.get("score") or 0.0)),
                    rank=int(rank_map.get(candidate_id) or 0),
                    feature_gaps=_feature_gap_rows(
                        query_row=query_row,
                        candidate_row=candidate_row,
                        top_features=top_features,
                    ),
                )
            )
        return out

    return NeighborEvidenceCard(
        method="quartile_similarity",
        temporal_safe=True,
        winner_count=len(winners),
        loser_count=len(losers),
        winners=_rows(winners),
        losers=_rows(losers),
    )


def temporal_confidence_band(
    *,
    score_std: float,
    score_raw: float,
    score_calibrated: float,
    rank_count: int,
    score_spread: float,
    source: str = "ensemble_plus_calibration",
) -> TemporalConfidenceBand:
    calibrated_shift = abs(float(score_calibrated) - float(score_raw))
    sigma = (float(score_std) + (0.5 * calibrated_shift)) * (max(2, rank_count) / max(1e-4, score_spread))
    sigma = _clip(sigma, 0.25, float(max(2, rank_count)))
    return TemporalConfidenceBand(
        p10=_round(-_NORMAL_Z * sigma),
        p50=0.0,
        p90=_round(_NORMAL_Z * sigma),
        source=source,
    )


def _mutable_feature_stats(
    query_row: Dict[str, Any],
    candidate_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Tuple[float, float]]:
    keys = (
        "caption_word_count",
        "hashtag_count",
        "keyword_count",
        "comment_confusion_index",
        "comment_help_seeking_index",
        "comment_sentiment_volatility",
    )
    values: Dict[str, List[float]] = {key: [] for key in keys}

    def _append_from_row(row: Dict[str, Any]) -> None:
        features = row.get("features", {}) if isinstance(row.get("features"), dict) else {}
        comment = features.get("comment_intelligence", {}) if isinstance(features.get("comment_intelligence"), dict) else {}
        values["caption_word_count"].append(float(features.get("caption_word_count") or 0.0))
        values["hashtag_count"].append(float(features.get("hashtag_count") or 0.0))
        values["keyword_count"].append(float(features.get("keyword_count") or 0.0))
        values["comment_confusion_index"].append(float(comment.get("confusion_index") or 0.0))
        values["comment_help_seeking_index"].append(float(comment.get("help_seeking_index") or 0.0))
        values["comment_sentiment_volatility"].append(float(comment.get("sentiment_volatility") or 0.0))

    _append_from_row(query_row)
    for row in candidate_rows:
        _append_from_row(row)

    out: Dict[str, Tuple[float, float]] = {}
    for key, items in values.items():
        arr = np.asarray(items if items else [0.0], dtype=np.float64)
        out[key] = (float(np.mean(arr)), max(1e-4, float(np.std(arr))))
    return out


def _apply_delta(
    *,
    row: Dict[str, Any],
    feature: str,
    delta_std: float,
    stats: Dict[str, Tuple[float, float]],
) -> Tuple[bool, float]:
    features = row.get("features", {})
    if not isinstance(features, dict):
        return False, 0.0
    comment = features.get("comment_intelligence")
    if not isinstance(comment, dict):
        comment = {}
        features["comment_intelligence"] = comment

    mean, std = stats.get(feature, (0.0, 1.0))
    delta = float(delta_std) * max(1e-4, float(std))
    if feature in {"caption_word_count", "hashtag_count", "keyword_count"}:
        current = float(features.get(feature) or 0.0)
        updated = max(0.0, current + delta)
        if feature in {"caption_word_count", "hashtag_count", "keyword_count"}:
            features[feature] = int(round(updated))
        return True, float(updated - current)
    if feature == "comment_confusion_index":
        current = float(comment.get("confusion_index") or 0.0)
        updated = _clip(current + delta, 0.0, 1.0)
        comment["confusion_index"] = float(updated)
        return True, float(updated - current)
    if feature == "comment_help_seeking_index":
        current = float(comment.get("help_seeking_index") or 0.0)
        updated = _clip(current + delta, 0.0, 1.0)
        comment["help_seeking_index"] = float(updated)
        return True, float(updated - current)
    if feature == "comment_sentiment_volatility":
        current = float(comment.get("sentiment_volatility") or 0.0)
        updated = max(0.0, current + delta)
        comment["sentiment_volatility"] = float(updated)
        return True, float(updated - current)
    _ = mean
    return False, 0.0


def _calibrated_score(
    *,
    ranker: RankerFamilyModel,
    query_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
    similarity: float,
    calibrator: Optional[ObjectiveSegmentCalibrators],
) -> float:
    ranker_score = ranker.score_pair(
        query_row=query_row,
        candidate_row=candidate_row,
        similarity=float(similarity),
    )
    score_raw = float(ranker_score.get("final_score") or 0.0)
    if calibrator is None:
        return score_raw
    calibration = calibrator.calibrate(
        score_raw=score_raw,
        segment_id=str(ranker_score.get("selected_ranker_id") or SEGMENT_GLOBAL),
        include_debug=False,
    )
    return float(calibration.get("score_calibrated") or score_raw)


def counterfactual_scenarios_for_items(
    *,
    objective: str,
    ranker: RankerFamilyModel,
    calibrator: Optional[ObjectiveSegmentCalibrators],
    query_row: Dict[str, Any],
    top_items: Sequence[Dict[str, Any]],
    ranked_items: Sequence[Dict[str, Any]],
    candidate_by_id: Dict[str, Dict[str, Any]],
    score_spread: float,
) -> Dict[str, List[CounterfactualScenarioResult]]:
    del objective
    if not top_items or not ranked_items:
        return {}

    ranked_by_score = sorted(
        ranked_items,
        key=lambda item: float(item.get("score_calibrated") or item.get("score") or 0.0),
        reverse=True,
    )
    baseline_rank = _candidate_rank_map(ranked_by_score)
    candidate_rows: List[Dict[str, Any]] = []
    for item in ranked_by_score:
        row_id = str(item.get("candidate_row_id") or item.get("candidate_id") or "")
        candidate_row = candidate_by_id.get(row_id) or candidate_by_id.get(
            str(item.get("candidate_id") or "")
        )
        if isinstance(candidate_row, dict):
            candidate_rows.append(candidate_row)
    stats = _mutable_feature_stats(query_row, candidate_rows)

    results: Dict[str, List[CounterfactualScenarioResult]] = {
        str(item.get("candidate_id") or item.get("candidate_row_id") or ""): []
        for item in top_items
        if str(item.get("candidate_id") or item.get("candidate_row_id") or "")
    }

    for scenario in _SCENARIOS:
        query_variant = copy.deepcopy(query_row)
        applied_deltas: Dict[str, float] = {}
        applied_count = 0
        for feature, delta_std in scenario.deltas_std.items():
            applied, observed_delta = _apply_delta(
                row=query_variant,
                feature=feature,
                delta_std=delta_std,
                stats=stats,
            )
            if applied:
                applied_count += 1
                applied_deltas[feature] = _round(observed_delta)

        if applied_count == 0:
            feasibility = "not_applicable"
            reason = "no_mapped_features"
            rescored_rank: Dict[str, int] = baseline_rank
        elif applied_count < len(scenario.deltas_std):
            feasibility = "partially_applied"
            reason = "partial_feature_mapping"
            rescored_rank = {}
        else:
            feasibility = "applied"
            reason = scenario.reason
            rescored_rank = {}

        if applied_count > 0:
            rescored: List[Tuple[str, float]] = []
            for item in ranked_by_score:
                candidate_id = str(item.get("candidate_id") or item.get("candidate_row_id") or "")
                row_id = str(item.get("candidate_row_id") or candidate_id)
                candidate_row = candidate_by_id.get(row_id) or candidate_by_id.get(candidate_id)
                if not isinstance(candidate_row, dict):
                    continue
                similarity = float((item.get("similarity") or {}).get("fused") or 0.0)
                score = _calibrated_score(
                    ranker=ranker,
                    query_row=query_variant,
                    candidate_row=candidate_row,
                    similarity=similarity,
                    calibrator=calibrator,
                )
                rescored.append((candidate_id, score))
            rescored.sort(key=lambda pair: pair[1], reverse=True)
            rescored_rank = {candidate_id: idx for idx, (candidate_id, _) in enumerate(rescored, start=1)}

        for top_item in top_items:
            candidate_id = str(top_item.get("candidate_id") or top_item.get("candidate_row_id") or "")
            if not candidate_id:
                continue
            base_rank = int(baseline_rank.get(candidate_id) or (len(ranked_by_score) + 1))
            next_rank = int(rescored_rank.get(candidate_id) or (len(ranked_by_score) + 1))
            delta = float(next_rank - base_rank)
            band = temporal_confidence_band(
                score_std=float(top_item.get("score_std") or 0.0),
                score_raw=float(top_item.get("score_raw") or top_item.get("score") or 0.0),
                score_calibrated=float(top_item.get("score_calibrated") or top_item.get("score") or 0.0),
                rank_count=len(ranked_by_score),
                score_spread=score_spread,
                source="counterfactual_rank_delta",
            )
            expected = TemporalConfidenceBand(
                p10=_round(delta + float(band.p10)),
                p50=_round(delta),
                p90=_round(delta + float(band.p90)),
                source=band.source,
            )
            results.setdefault(candidate_id, []).append(
                CounterfactualScenarioResult(
                    scenario_id=scenario.scenario_id,
                    expected_rank_delta_band=expected,
                    feasibility=feasibility,
                    reason=reason,
                    applied_deltas=dict(applied_deltas),
                    trace={
                        "baseline_rank": base_rank,
                        "counterfactual_rank": next_rank,
                        "scenario_size": len(scenario.deltas_std),
                    },
                )
            )
    return results


def as_payload(value: Any) -> Any:
    if isinstance(value, list):
        return [as_payload(item) for item in value]
    if isinstance(value, dict):
        return {str(key): as_payload(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return as_payload(asdict(value))
    if isinstance(value, float):
        return _round(value)
    return value


def metadata_payload(
    *,
    objective: str,
    methods: Sequence[str],
    fallback_counts: Dict[str, int],
) -> Dict[str, Any]:
    return {
        "version": EXPLAINABILITY_VERSION,
        "objective": objective,
        "generated_at": _to_iso_utc_now(),
        "methods": sorted({str(method) for method in methods if str(method).strip()}),
        "fallback_counts": dict(fallback_counts),
    }
