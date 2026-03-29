from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .artifacts import ArtifactRegistry
from .calibration import ObjectiveSegmentCalibrators
from .evaluator import evaluate_ranking, evaluate_retrieval, recall_at_k
from .objectives import OBJECTIVE_SPECS, map_objective
from .ranker import (
    FEATURE_NAMES,
    ObjectiveRankerConfig,
    ObjectiveRankerModel,
    RankerEnsembleModel,
    RankerFamilyConfig,
    RankerFamilyModel,
    SEGMENT_GLOBAL,
    SEGMENT_CREATOR_COLD_START,
    SEGMENT_CREATOR_MATURE,
    SEGMENT_FORMAT_ENTERTAINMENT,
    SEGMENT_FORMAT_TUTORIAL,
    segment_candidates_for_pair,
    pair_feature_vector_array,
)
from .retriever import HybridRetriever, HybridRetrieverTrainerConfig
from .graph import (
    GraphBuildConfig,
    annotate_rows_with_graph_features,
    build_creator_video_dna_graph,
)
from .trajectory import (
    TRAJECTORY_VERSION,
    TrajectoryBuildConfig,
    TrajectoryBundle,
    annotate_rows_with_trajectory_features,
    build_trajectory_bundle,
)
from .sampling import (
    AdaptiveNegativeMiner,
    AdaptiveNegativeMiningConfig,
    NegativeSampler,
    NegativeSamplerConfig,
)
from .temporal import TemporalCandidatePool, TemporalCandidatePoolConfig, split_rows
from .policy import PolicyRerankerConfig
from ..fabric import FeatureFabric

VALID_PAIR_TARGET_SOURCES = {"scalar_v1", "trajectory_v2_composite"}
VALID_NEGATIVE_MINING_MODES = {"fixed_v1", "adaptive_v2"}
RANKER_SEGMENTS = (
    SEGMENT_CREATOR_COLD_START,
    SEGMENT_CREATOR_MATURE,
    SEGMENT_FORMAT_TUTORIAL,
    SEGMENT_FORMAT_ENTERTAINMENT,
)


def _to_utc_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class RecommenderTrainingConfig:
    objectives: Sequence[str] = ("reach", "engagement", "conversion")
    retrieve_k: int = 200
    max_age_days: int = 180
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    random_seed: int = 13
    run_name: str = "recommender-v1"
    pair_target_source: str = "scalar_v1"
    negative_mining_mode: str = "fixed_v1"
    adaptive_negative_mining: AdaptiveNegativeMiningConfig = field(
        default_factory=AdaptiveNegativeMiningConfig
    )
    ranker_ensemble_size: int = 5
    ranker_uncertainty_std_ref: float = 0.15
    segment_min_train_pairs: int = 120
    segment_min_validation_pairs: int = 20
    creator_cold_start_threshold: int = 10
    ranker_calibration_min_support: int = 25
    feature_snapshot_manifest_path: Optional[str] = None
    graph_enabled: bool = True
    graph_embedding_dim: int = 32
    graph_walk_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "walk_length": 12,
            "num_walks": 20,
            "context_size": 4,
            "seed": 13,
        }
    )
    graph_weighting_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "recency_half_life_days": 45.0,
            "include_creator_similarity": True,
            "creator_similarity_top_k": 5,
            "creator_similarity_min_jaccard": 0.15,
            "branch_weight": 0.10,
        }
    )
    trajectory_enabled: bool = True
    trajectory_embedding_dim: int = 16
    trajectory_feature_version: str = "trajectory_features.v2"
    trajectory_branch_weight: float = 0.08
    trajectory_encoder_mode: str = "feature_only"
    trajectory_manifest_path: Optional[str] = None
    contract_version: str = "contract.v2"
    datamart_version: str = "datamart.v1"

    def __post_init__(self) -> None:
        if self.pair_target_source not in VALID_PAIR_TARGET_SOURCES:
            raise ValueError(
                "pair_target_source must be one of "
                f"{sorted(VALID_PAIR_TARGET_SOURCES)}; got '{self.pair_target_source}'"
            )
        if self.negative_mining_mode not in VALID_NEGATIVE_MINING_MODES:
            raise ValueError(
                "negative_mining_mode must be one of "
                f"{sorted(VALID_NEGATIVE_MINING_MODES)}; got '{self.negative_mining_mode}'"
            )
        if self.ranker_ensemble_size < 1:
            raise ValueError("ranker_ensemble_size must be >= 1.")
        if self.ranker_uncertainty_std_ref <= 0:
            raise ValueError("ranker_uncertainty_std_ref must be > 0.")
        if self.segment_min_train_pairs < 1 or self.segment_min_validation_pairs < 1:
            raise ValueError("segment_min_train_pairs and segment_min_validation_pairs must be >= 1.")
        if self.creator_cold_start_threshold < 1:
            raise ValueError("creator_cold_start_threshold must be >= 1.")
        if self.ranker_calibration_min_support < 1:
            raise ValueError("ranker_calibration_min_support must be >= 1.")
        if self.graph_embedding_dim < 4:
            raise ValueError("graph_embedding_dim must be >= 4.")
        if self.trajectory_embedding_dim < 4:
            raise ValueError("trajectory_embedding_dim must be >= 4.")
        if self.trajectory_branch_weight < 0:
            raise ValueError("trajectory_branch_weight must be >= 0.")
        if self.trajectory_encoder_mode not in {
            "feature_only",
            "sequence_encoder_shadow",
        }:
            raise ValueError(
                "trajectory_encoder_mode must be one of: feature_only, sequence_encoder_shadow."
            )


class HybridRetrieverTrainer:
    def __init__(
        self,
        config: Optional[HybridRetrieverTrainerConfig] = None,
    ) -> None:
        self.config = config or HybridRetrieverTrainerConfig()

    def train(
        self,
        rows: Sequence[Dict[str, Any]],
        multimodal_vectors: Optional[Dict[str, Sequence[float]]] = None,
        graph_vectors: Optional[Dict[str, Sequence[float]]] = None,
        graph_lookup: Optional[Dict[str, Dict[str, Sequence[float]]]] = None,
        graph_metadata: Optional[Dict[str, Any]] = None,
        trajectory_vectors: Optional[Dict[str, Sequence[float]]] = None,
        trajectory_lookup: Optional[Dict[str, Dict[str, Sequence[float]]]] = None,
        trajectory_metadata: Optional[Dict[str, Any]] = None,
    ) -> HybridRetriever:
        return HybridRetriever.train(
            rows=rows,
            config=self.config,
            multimodal_vectors=multimodal_vectors,
            graph_vectors=graph_vectors,
            graph_lookup=graph_lookup,
            graph_metadata=graph_metadata,
            trajectory_vectors=trajectory_vectors,
            trajectory_lookup=trajectory_lookup,
            trajectory_metadata=trajectory_metadata,
        )


def _resolve_manifest_path(manifest_ref: str) -> Path:
    path = Path(manifest_ref)
    if path.is_dir():
        return path / "manifest.json"
    return path


def _load_feature_snapshot_vectors(
    manifest_ref: Optional[str],
) -> tuple[Optional[str], Dict[str, List[float]]]:
    if not manifest_ref:
        return None, {}
    manifest_path = _resolve_manifest_path(manifest_ref)
    if not manifest_path.exists():
        return None, {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_file = Path(str(payload.get("feature_file") or ""))
    if feature_file and not feature_file.is_absolute():
        feature_file = (manifest_path.parent / feature_file).resolve()
    if not feature_file.exists():
        return str(payload.get("feature_manifest_id") or ""), {}
    file_format = str(payload.get("format") or "").lower()
    rows: List[Dict[str, Any]] = []
    if file_format == "parquet":
        try:
            import pandas as pd  # type: ignore

            frame = pd.read_parquet(feature_file)
            rows = frame.to_dict(orient="records")
        except Exception:
            rows = []
    else:
        rows = []
        for line in feature_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    out: Dict[str, List[float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            continue
        out[video_id] = [
            float(row.get("text_clarity_score") or 0.0),
            float(row.get("structure_hook_timing_seconds") or 0.0),
            float(row.get("structure_payoff_timing_seconds") or 0.0),
            float(row.get("audio_speech_ratio") or 0.0),
            float(row.get("visual_shot_change_rate") or 0.0),
            float(row.get("text_token_count") or 0.0),
        ]
    return str(payload.get("feature_manifest_id") or ""), out


def _load_trajectory_bundle_from_manifest(
    manifest_ref: Optional[str],
) -> Optional[TrajectoryBundle]:
    if not manifest_ref:
        return None
    manifest_path = _resolve_manifest_path(manifest_ref)
    if not manifest_path.exists():
        return None
    bundle_dir = manifest_path.parent
    try:
        return TrajectoryBundle.load(bundle_dir)
    except Exception:
        return None


class ObjectiveRankerTrainer:
    def __init__(self, objective: str, random_seed: int = 13) -> None:
        _, self.objective = map_objective(objective)
        self.random_seed = random_seed

    def train(
        self,
        rows_by_id: Dict[str, Dict[str, Any]],
        pair_rows: Sequence[Dict[str, Any]],
    ) -> ObjectiveRankerModel:
        return ObjectiveRankerModel.train(
            config=ObjectiveRankerConfig(
                objective=self.objective,
                random_state=self.random_seed,
            ),
            rows_by_id=rows_by_id,
            pair_rows=pair_rows,
        )


def _rows_by_id(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("row_id")): row for row in rows}


def _relevance_label_from_score(score: float) -> int:
    if score >= 1.0:
        return 3
    if score >= 0.3:
        return 2
    if score >= -0.3:
        return 1
    return 0


def _pair_rows_for_objective(
    rows_by_id: Dict[str, Dict[str, Any]],
    pair_rows: Sequence[Dict[str, Any]],
    objective: str,
    pair_target_source: str = "scalar_v1",
) -> List[Dict[str, Any]]:
    direct = [
        row
        for row in pair_rows
        if str(row.get("objective")) == objective
        and str(row.get("target_source", "scalar_v1")) == pair_target_source
    ]
    if direct:
        return direct

    out: List[Dict[str, Any]] = []
    for pair in pair_rows:
        candidate_id = str(pair.get("candidate_row_id"))
        candidate_row = rows_by_id.get(candidate_id)
        if candidate_row is None:
            continue
        if pair_target_source == "trajectory_v2_composite":
            trajectory_targets = (
                candidate_row.get("targets_trajectory_z", {})
                if isinstance(candidate_row.get("targets_trajectory_z"), dict)
                else {}
            ).get(objective, {})
            score = trajectory_targets.get("composite_z") if isinstance(trajectory_targets, dict) else None
            if score is None:
                continue
            score = float(score)
            components = (
                trajectory_targets.get("components_z", {})
                if isinstance(trajectory_targets, dict)
                else {}
            )
            objective_score_components = {
                key: float(value)
                for key, value in (components.items() if isinstance(components, dict) else [])
                if value is not None and key in {"early_velocity", "stability", "late_lift"}
            }
            availability_payload = (
                candidate_row.get("target_availability", {})
                if isinstance(candidate_row.get("target_availability"), dict)
                else {}
            ).get(objective, {})
            component_mask = (
                availability_payload.get("components", {})
                if isinstance(availability_payload, dict)
                else {}
            )
            availability_mask = {
                "candidate_objective_available": bool(
                    availability_payload.get("objective_available", False)
                ),
                "candidate_early_available": bool(
                    component_mask.get("early_velocity", False)
                ),
                "candidate_stability_available": bool(
                    component_mask.get("stability", False)
                ),
                "candidate_late_available": bool(
                    component_mask.get("late_lift", False)
                ),
            }
        else:
            score = float(
                (
                    candidate_row.get("targets_z", {})
                    if isinstance(candidate_row.get("targets_z"), dict)
                    else {}
                ).get(objective, 0.0)
            )
            objective_score_components = None
            availability_mask = None
        out.append(
            {
                **pair,
                "objective": objective,
                "target_source": pair_target_source,
                "objective_score": round(score, 6),
                "objective_score_components": objective_score_components,
                "availability_mask": availability_mask,
                "relevance_label": _relevance_label_from_score(score),
            }
        )
    return out


def _group_relevance_by_query(
    pair_rows: Sequence[Dict[str, Any]],
    objective: str,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for pair in pair_rows:
        if str(pair.get("objective")) != objective:
            continue
        qid = str(pair.get("query_row_id"))
        cid = str(pair.get("candidate_row_id"))
        rel = float(pair.get("relevance_label") or 0.0)
        out.setdefault(qid, {})[cid] = rel
    return out


def _parse_as_of(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def _annotate_creator_prior_video_counts(rows: Sequence[Dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: _parse_as_of(row.get("as_of_time")))
    author_counts: Dict[str, int] = {}
    for row in ordered:
        author_id = str(row.get("author_id") or "unknown")
        prior = int(author_counts.get(author_id, 0))
        row["_creator_prior_video_count"] = prior
        features = row.get("features")
        if isinstance(features, dict):
            features["creator_prior_video_count"] = prior
        author_counts[author_id] = prior + 1


def _pair_applies_segment(
    *,
    query_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
    segment_id: str,
    creator_cold_start_threshold: int,
) -> bool:
    return segment_id in segment_candidates_for_pair(
        query_row=query_row,
        candidate_row=candidate_row,
        creator_cold_threshold=creator_cold_start_threshold,
    )


def _segment_support_counts(
    *,
    rows_by_id: Dict[str, Dict[str, Any]],
    pair_rows: Sequence[Dict[str, Any]],
    objective: str,
    segment_id: str,
    creator_cold_start_threshold: int,
) -> Dict[str, int]:
    train_count = 0
    validation_count = 0
    for pair in pair_rows:
        if str(pair.get("objective")) != objective:
            continue
        qid = str(pair.get("query_row_id"))
        cid = str(pair.get("candidate_row_id"))
        query_row = rows_by_id.get(qid)
        candidate_row = rows_by_id.get(cid)
        if query_row is None or candidate_row is None:
            continue
        if str(candidate_row.get("split")) != "train":
            continue
        if not _pair_applies_segment(
            query_row=query_row,
            candidate_row=candidate_row,
            segment_id=segment_id,
            creator_cold_start_threshold=creator_cold_start_threshold,
        ):
            continue
        if str(query_row.get("split")) == "train":
            train_count += 1
        elif str(query_row.get("split")) == "validation":
            validation_count += 1
    return {
        "train_pairs": train_count,
        "validation_pairs": validation_count,
    }


def _pair_rows_for_segment(
    *,
    rows_by_id: Dict[str, Dict[str, Any]],
    pair_rows: Sequence[Dict[str, Any]],
    objective: str,
    segment_id: str,
    creator_cold_start_threshold: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pair in pair_rows:
        if str(pair.get("objective")) != objective:
            continue
        qid = str(pair.get("query_row_id"))
        cid = str(pair.get("candidate_row_id"))
        query_row = rows_by_id.get(qid)
        candidate_row = rows_by_id.get(cid)
        if query_row is None or candidate_row is None:
            continue
        if not _pair_applies_segment(
            query_row=query_row,
            candidate_row=candidate_row,
            segment_id=segment_id,
            creator_cold_start_threshold=creator_cold_start_threshold,
        ):
            continue
        out.append(pair)
    return out


def _ranker_score(
    *,
    ranker: Any,
    query_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
    similarity: float,
) -> Tuple[float, float]:
    pair_vector = pair_feature_vector_array(
        query_row=query_row,
        candidate_row=candidate_row,
        similarity=similarity,
    )
    if isinstance(ranker, RankerEnsembleModel):
        mean_scores, std_scores = ranker.predict_stats(pair_vector)
        return float(mean_scores[0]), float(std_scores[0])
    if isinstance(ranker, ObjectiveRankerModel):
        return float(ranker.predict_scores(pair_vector)[0]), 0.0
    return float(similarity), 0.0


def _evaluate_ranker_objective(
    ranker: Any,
    objective: str,
    rows_by_id: Dict[str, Dict[str, Any]],
    pair_rows: Sequence[Dict[str, Any]],
    *,
    segment_id: Optional[str] = None,
    creator_cold_start_threshold: int = 10,
) -> Dict[str, float]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for pair in pair_rows:
        if str(pair.get("objective")) != objective:
            continue
        qid = str(pair.get("query_row_id"))
        query_row = rows_by_id.get(qid)
        candidate_row = rows_by_id.get(str(pair.get("candidate_row_id")))
        if query_row is None or candidate_row is None:
            continue
        if str(query_row.get("split")) not in {"validation", "test"}:
            continue
        if segment_id is not None and not _pair_applies_segment(
            query_row=query_row,
            candidate_row=candidate_row,
            segment_id=segment_id,
            creator_cold_start_threshold=creator_cold_start_threshold,
        ):
            continue
        score, _ = _ranker_score(
            ranker=ranker,
            query_row=query_row,
            candidate_row=candidate_row,
            similarity=float(pair.get("similarity") or 0.0),
        )
        grouped.setdefault(qid, []).append(
            {
                "candidate_row_id": str(pair.get("candidate_row_id")),
                "score": score,
                "relevance": float(pair.get("relevance_label") or 0.0),
            }
        )

    query_payloads = []
    for qid, items in grouped.items():
        items.sort(key=lambda item: item["score"], reverse=True)
        query_payloads.append({"query_id": qid, "items": items})
    return evaluate_ranking(query_payloads, k_values=(10, 20))


def _evaluate_segment_blended_objective(
    *,
    global_ranker: RankerEnsembleModel,
    segment_ranker: RankerEnsembleModel,
    objective: str,
    rows_by_id: Dict[str, Dict[str, Any]],
    pair_rows: Sequence[Dict[str, Any]],
    segment_id: str,
    creator_cold_start_threshold: int,
) -> Dict[str, float]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for pair in pair_rows:
        if str(pair.get("objective")) != objective:
            continue
        qid = str(pair.get("query_row_id"))
        query_row = rows_by_id.get(qid)
        candidate_row = rows_by_id.get(str(pair.get("candidate_row_id")))
        if query_row is None or candidate_row is None:
            continue
        if str(query_row.get("split")) not in {"validation", "test"}:
            continue
        similarity = float(pair.get("similarity") or 0.0)
        global_score, _ = _ranker_score(
            ranker=global_ranker,
            query_row=query_row,
            candidate_row=candidate_row,
            similarity=similarity,
        )
        use_segment = _pair_applies_segment(
            query_row=query_row,
            candidate_row=candidate_row,
            segment_id=segment_id,
            creator_cold_start_threshold=creator_cold_start_threshold,
        )
        if use_segment:
            segment_mean, segment_std = _ranker_score(
                ranker=segment_ranker,
                query_row=query_row,
                candidate_row=candidate_row,
                similarity=similarity,
            )
            blend_weight = max(0.2, min(1.0, 1.0 - (segment_std / max(1e-6, segment_ranker.std_ref))))
            score = (blend_weight * segment_mean) + ((1.0 - blend_weight) * global_score)
        else:
            score = global_score
        grouped.setdefault(qid, []).append(
            {
                "candidate_row_id": str(pair.get("candidate_row_id")),
                "score": float(score),
                "relevance": float(pair.get("relevance_label") or 0.0),
            }
        )

    query_payloads = []
    for qid, items in grouped.items():
        items.sort(key=lambda item: item["score"], reverse=True)
        query_payloads.append({"query_id": qid, "items": items})
    return evaluate_ranking(query_payloads, k_values=(10, 20))


def _collect_calibration_samples(
    *,
    ranker_family: RankerFamilyModel,
    objective: str,
    rows_by_id: Dict[str, Dict[str, Any]],
    pair_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pair in pair_rows:
        if str(pair.get("objective")) != objective:
            continue
        qid = str(pair.get("query_row_id"))
        cid = str(pair.get("candidate_row_id"))
        query_row = rows_by_id.get(qid)
        candidate_row = rows_by_id.get(cid)
        if query_row is None or candidate_row is None:
            continue
        if str(query_row.get("split")) != "validation":
            continue
        score_payload = ranker_family.score_pair(
            query_row=query_row,
            candidate_row=candidate_row,
            similarity=float(pair.get("similarity") or 0.0),
        )
        out.append(
            {
                "segment_id": str(score_payload.get("selected_ranker_id") or SEGMENT_GLOBAL),
                "score_raw": float(score_payload.get("final_score") or 0.0),
                "label": float(pair.get("relevance_label") or 0.0),
            }
        )
    return out


def _calibration_fallback_summary(
    *,
    calibration_bundle: ObjectiveSegmentCalibrators,
    samples: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    total = 0
    fallback_total = 0
    by_segment_total: Dict[str, int] = {}
    by_segment_fallback: Dict[str, int] = {}
    for sample in samples:
        segment_id = str(sample.get("segment_id") or SEGMENT_GLOBAL)
        score_raw = float(sample.get("score_raw") or 0.0)
        result = calibration_bundle.calibrate(score_raw=score_raw, segment_id=segment_id)
        total += 1
        by_segment_total[segment_id] = by_segment_total.get(segment_id, 0) + 1
        if bool(result.get("calibration_fallback_used")):
            fallback_total += 1
            by_segment_fallback[segment_id] = by_segment_fallback.get(segment_id, 0) + 1
    return {
        "total": total,
        "fallback_total": fallback_total,
        "fallback_rate": round(fallback_total / max(1, total), 6),
        "by_segment": {
            segment_id: {
                "total": count,
                "fallback_total": by_segment_fallback.get(segment_id, 0),
                "fallback_rate": round(
                    by_segment_fallback.get(segment_id, 0) / max(1, count),
                    6,
                ),
            }
            for segment_id, count in sorted(by_segment_total.items())
        },
    }


def _evaluate_retriever_objective(
    retriever: HybridRetriever,
    objective: str,
    rows_split: Dict[str, List[Dict[str, Any]]],
    relevance_by_query: Dict[str, Dict[str, float]],
    retrieve_k: int,
    max_age_days: int,
) -> Dict[str, float]:
    pool_builder = TemporalCandidatePool(
        TemporalCandidatePoolConfig(
            max_age_days=max_age_days,
            min_pool_size=30,
            enforce_index_cutoff=True,
        )
    )
    eval_queries = rows_split["validation"] + rows_split["test"]
    query_payloads: List[Dict[str, Any]] = []
    for query in eval_queries:
        query_id = str(query.get("row_id"))
        rel = relevance_by_query.get(query_id, {})
        relevant_ids = [cid for cid, score in rel.items() if score >= 2.0]
        if not relevant_ids:
            continue
        pool = pool_builder.for_query(
            query_row=query,
            candidate_rows=rows_split["train"] + rows_split["validation"],
            index_cutoff_time=query.get("as_of_time"),
        )
        items = retriever.retrieve(
            query_row=query,
            candidate_rows=pool,
            top_k=retrieve_k,
            index_cutoff_time=query.get("as_of_time"),
            objective=objective,
            retrieval_constraints={"max_age_days": max_age_days},
        )
        query_payloads.append(
            {
                "query_id": query_id,
                "items": items,
                "relevant_ids": relevant_ids,
            }
        )
    return evaluate_retrieval(query_payloads, top_k=retrieve_k)


def _learn_objective_blend_weights(
    retriever: HybridRetriever,
    objective: str,
    rows_split: Dict[str, List[Dict[str, Any]]],
    relevance_by_query: Dict[str, Dict[str, float]],
    retrieve_k: int,
    max_age_days: int,
) -> Dict[str, float]:
    pool_builder = TemporalCandidatePool(
        TemporalCandidatePoolConfig(
            max_age_days=max_age_days,
            min_pool_size=30,
            enforce_index_cutoff=True,
        )
    )
    eval_queries = rows_split["validation"] or rows_split["test"]
    if not eval_queries:
        return retriever.branch_weights(objective)

    step_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    candidates: List[Dict[str, float]] = []
    for lexical in step_values:
        for dense in step_values:
            for multimodal in step_values:
                for trajectory_dense in step_values:
                    graph_dense = 1.0 - lexical - dense - multimodal - trajectory_dense
                    if graph_dense < 0:
                        continue
                    candidates.append(
                        {
                            "lexical": lexical,
                            "dense_text": dense,
                            "multimodal": multimodal,
                            "graph_dense": graph_dense,
                            "trajectory_dense": trajectory_dense,
                        }
                    )
    if not candidates:
        return retriever.branch_weights(objective)

    best = retriever.branch_weights(objective)
    best_score = -1.0
    for weights in candidates:
        recalls: List[float] = []
        for query in eval_queries:
            query_id = str(query.get("row_id"))
            rel = relevance_by_query.get(query_id, {})
            relevant_ids = {
                cid for cid, score in rel.items() if float(score) >= 2.0
            }
            if not relevant_ids:
                continue
            pool = pool_builder.for_query(
                query_row=query,
                candidate_rows=rows_split["train"] + rows_split["validation"],
                index_cutoff_time=query.get("as_of_time"),
            )
            items = retriever.retrieve(
                query_row=query,
                candidate_rows=pool,
                top_k=retrieve_k,
                index_cutoff_time=query.get("as_of_time"),
                objective=objective,
                retrieval_constraints={"max_age_days": max_age_days},
                weight_override=weights,
            )
            ranked = [str(item.get("candidate_row_id")) for item in items]
            recalls.append(recall_at_k(ranked, relevant_ids, min(100, retrieve_k)))
        mean_score = sum(recalls) / len(recalls) if recalls else 0.0
        if mean_score > best_score:
            best_score = mean_score
            best = weights
    return best


def train_recommender_from_datamart(
    datamart: Dict[str, Any],
    artifact_root: Path,
    config: Optional[RecommenderTrainingConfig] = None,
) -> Dict[str, Any]:
    cfg = config or RecommenderTrainingConfig()
    rows = list(datamart.get("rows") or [])
    pair_rows = list(datamart.get("pair_rows") or [])
    if not rows:
        raise ValueError("Training data mart has no rows.")

    _annotate_creator_prior_video_counts(rows)
    graph_bundle = None
    graph_vectors_by_row: Dict[str, Sequence[float]] = {}
    graph_lookup: Dict[str, Dict[str, Sequence[float]]] = {}
    graph_manifest_payload: Optional[Dict[str, Any]] = None
    trajectory_bundle: Optional[TrajectoryBundle] = None
    trajectory_vectors_by_row: Dict[str, Sequence[float]] = {}
    trajectory_lookup: Dict[str, Dict[str, Sequence[float]]] = {}
    trajectory_manifest_payload: Optional[Dict[str, Any]] = None
    if cfg.graph_enabled:
        graph_cfg = GraphBuildConfig(
            embedding_dim=cfg.graph_embedding_dim,
            walk_length=int(cfg.graph_walk_params.get("walk_length", 12)),
            num_walks=int(cfg.graph_walk_params.get("num_walks", 20)),
            context_size=int(cfg.graph_walk_params.get("context_size", 4)),
            seed=int(cfg.graph_walk_params.get("seed", cfg.random_seed)),
            recency_half_life_days=float(
                cfg.graph_weighting_params.get("recency_half_life_days", 45.0)
            ),
            include_creator_similarity=bool(
                cfg.graph_weighting_params.get("include_creator_similarity", True)
            ),
            creator_similarity_top_k=int(
                cfg.graph_weighting_params.get("creator_similarity_top_k", 5)
            ),
            creator_similarity_min_jaccard=float(
                cfg.graph_weighting_params.get("creator_similarity_min_jaccard", 0.15)
            ),
        )
        graph_bundle = build_creator_video_dna_graph(
            rows=rows,
            as_of_time=datamart.get("generated_at"),
            run_cutoff_time=datamart.get("generated_at"),
            config=graph_cfg,
        )
        annotate_rows_with_graph_features(rows, graph_bundle)
    if cfg.trajectory_enabled:
        trajectory_bundle = _load_trajectory_bundle_from_manifest(
            cfg.trajectory_manifest_path
        )
        if trajectory_bundle is None:
            trajectory_cfg = TrajectoryBuildConfig(
                windows_hours=(6, 24, 96),
                embedding_dim=cfg.trajectory_embedding_dim,
                feature_version=cfg.trajectory_feature_version,
                encoder_mode=cfg.trajectory_encoder_mode,
            )
            trajectory_bundle = build_trajectory_bundle(
                rows=rows,
                as_of_time=datamart.get("generated_at"),
                run_cutoff_time=datamart.get("generated_at"),
                config=trajectory_cfg,
            )
        annotate_rows_with_trajectory_features(rows, trajectory_bundle)
    rows_split = split_rows(rows)
    if not rows_split["train"]:
        raise ValueError("Training data mart has no train rows.")

    registry = ArtifactRegistry(artifact_root)
    bundle_dir = registry.create_bundle_dir(cfg.run_name)
    retriever_dir = bundle_dir / "retriever"
    rankers_dir = bundle_dir / "rankers"
    metrics_dir = bundle_dir / "metrics"
    diagnostics_dir = bundle_dir / "diagnostics"
    retriever_dir.mkdir(parents=True, exist_ok=True)
    rankers_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = bundle_dir / "graph"
    trajectory_dir = bundle_dir / "trajectory"
    if graph_bundle is not None:
        graph_bundle.save(graph_dir)
        graph_manifest = json.loads((graph_dir / "graph_manifest.json").read_text(encoding="utf-8"))
        graph_manifest_payload = {
            "graph_bundle_id": graph_manifest.get("graph_bundle_id"),
            "graph_version": graph_manifest.get("version"),
            "graph_schema_hash": graph_manifest.get("graph_schema_hash"),
            "graph_node_count": graph_manifest.get("node_count"),
            "graph_edge_count": graph_manifest.get("edge_count"),
            "path": str(graph_dir / "graph_manifest.json"),
        }
    if trajectory_bundle is not None:
        trajectory_bundle.save(trajectory_dir)
        trajectory_manifest = json.loads(
            (trajectory_dir / "manifest.json").read_text(encoding="utf-8")
        )
        trajectory_manifest_payload = {
            "trajectory_manifest_id": trajectory_manifest.get("trajectory_manifest_id"),
            "trajectory_version": trajectory_manifest.get("version"),
            "trajectory_schema_hash": trajectory_manifest.get("trajectory_schema_hash"),
            "profile_count": trajectory_manifest.get("profile_count"),
            "embedding_count": trajectory_manifest.get("embedding_count"),
            "path": str(trajectory_dir / "manifest.json"),
        }
    feature_schema_hash = registry.feature_schema_hash(FEATURE_NAMES)

    feature_manifest_id, feature_vectors_by_video = _load_feature_snapshot_vectors(
        cfg.feature_snapshot_manifest_path
    )
    multimodal_vectors_by_row: Dict[str, Sequence[float]] = {}
    for row in rows_split["train"]:
        row_id = str(row.get("row_id"))
        video_id = str(row.get("video_id") or row_id.split("::", 1)[0])
        vector = feature_vectors_by_video.get(video_id)
        if vector is not None:
            multimodal_vectors_by_row[row_id] = vector
        if graph_bundle is not None:
            graph_vector = graph_bundle.video_embeddings.get(video_id)
            if graph_vector is not None:
                graph_vectors_by_row[row_id] = graph_vector
        if trajectory_bundle is not None:
            trajectory_vector = trajectory_bundle.embeddings_by_video.get(video_id)
            if trajectory_vector is not None:
                trajectory_vectors_by_row[row_id] = trajectory_vector

    if graph_bundle is not None:
        graph_lookup = {
            "video": {
                key: list(value)
                for key, value in graph_bundle.video_embeddings.items()
            },
            "creator": {
                key: list(value)
                for key, value in graph_bundle.creator_embeddings.items()
            },
            "hashtag": {
                key: list(value)
                for key, value in graph_bundle.hashtag_embeddings.items()
            },
            "audio_motif": {
                key: list(value)
                for key, value in graph_bundle.audio_embeddings.items()
            },
            "style_signature": {
                key: list(value)
                for key, value in graph_bundle.style_embeddings.items()
            },
        }
    if trajectory_bundle is not None:
        trajectory_lookup = {
            "video": {
                key: list(value)
                for key, value in trajectory_bundle.embeddings_by_video.items()
            }
        }

    retriever = HybridRetrieverTrainer(
        HybridRetrieverTrainerConfig(
            dense_model_name=cfg.dense_model_name,
            graph_weight=float(cfg.graph_weighting_params.get("branch_weight", 0.10)),
            trajectory_weight=float(cfg.trajectory_branch_weight),
        )
    ).train(
        rows_split["train"],
        multimodal_vectors=multimodal_vectors_by_row or None,
        graph_vectors=graph_vectors_by_row or None,
        graph_lookup=graph_lookup or None,
        graph_metadata={
            "graph_bundle_id": (
                graph_manifest_payload.get("graph_bundle_id")
                if isinstance(graph_manifest_payload, dict)
                else None
            ),
            "graph_version": (
                graph_manifest_payload.get("graph_version")
                if isinstance(graph_manifest_payload, dict)
                else None
            ),
            "graph_schema_hash": (
                graph_manifest_payload.get("graph_schema_hash")
                if isinstance(graph_manifest_payload, dict)
                else None
            ),
        },
        trajectory_vectors=trajectory_vectors_by_row or None,
        trajectory_lookup=trajectory_lookup or None,
        trajectory_metadata={
            "trajectory_manifest_id": (
                trajectory_manifest_payload.get("trajectory_manifest_id")
                if isinstance(trajectory_manifest_payload, dict)
                else None
            ),
            "trajectory_version": (
                trajectory_manifest_payload.get("trajectory_version")
                if isinstance(trajectory_manifest_payload, dict)
                else None
            ),
            "trajectory_schema_hash": (
                trajectory_manifest_payload.get("trajectory_schema_hash")
                if isinstance(trajectory_manifest_payload, dict)
                else None
            ),
        },
    )

    rows_by_id = _rows_by_id(rows)

    # Diagnostics from negative sampler policy
    neg_sampler = NegativeSampler(
        NegativeSamplerConfig(seed=cfg.random_seed)
    )
    sample_diagnostics: Dict[str, Any] = {}
    if rows_split["validation"]:
        query_row = rows_split["validation"][0]
        pool_builder = TemporalCandidatePool(
            TemporalCandidatePoolConfig(max_age_days=cfg.max_age_days)
        )
        pool = pool_builder.for_query(
            query_row=query_row,
            candidate_rows=rows_split["train"] + rows_split["validation"],
            index_cutoff_time=query_row.get("as_of_time"),
        )
        positives = pool[:3]
        negatives = neg_sampler.sample(query_row=query_row, positives=positives, candidate_pool=pool)
        sample_diagnostics = {
            "query_row_id": query_row.get("row_id"),
            "pool_size": len(pool),
            "sample_positive_count": len(positives),
            "sample_negative_count": len(negatives),
            "policy": {
                "negatives_per_positive": neg_sampler.config.negatives_per_positive,
                "hard_ratio": neg_sampler.config.hard_ratio,
                "semihard_ratio": neg_sampler.config.semihard_ratio,
                "easy_ratio": neg_sampler.config.easy_ratio,
            },
        }

    objective_metrics: Dict[str, Dict[str, Any]] = {}
    trained_objectives: List[str] = []
    learned_objective_blend: Dict[str, Dict[str, float]] = {}
    adaptive_selected_by_objective: Dict[str, bool] = {}
    objective_diagnostics_manifest: Dict[str, Dict[str, str]] = {}
    objective_calibration_manifest: Dict[str, Dict[str, str]] = {}
    segment_support_by_objective: Dict[str, Dict[str, Dict[str, int]]] = {}
    segment_promotion_by_objective: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for objective in cfg.objectives:
        _, effective_objective = map_objective(objective)
        if effective_objective in trained_objectives:
            continue

        objective_pair_rows = _pair_rows_for_objective(
            rows_by_id=rows_by_id,
            pair_rows=pair_rows,
            objective=effective_objective,
            pair_target_source=cfg.pair_target_source,
        )
        spec = OBJECTIVE_SPECS[effective_objective]
        relevance_by_query = _group_relevance_by_query(
            pair_rows=objective_pair_rows,
            objective=effective_objective,
        )
        learned_weights = _learn_objective_blend_weights(
            retriever=retriever,
            objective=effective_objective,
            rows_split=rows_split,
            relevance_by_query=relevance_by_query,
            retrieve_k=cfg.retrieve_k,
            max_age_days=cfg.max_age_days,
        )
        learned_objective_blend[effective_objective] = learned_weights
        retriever.set_objective_blend({effective_objective: learned_weights})
        retrieval_eval = _evaluate_retriever_objective(
            retriever=retriever,
            objective=effective_objective,
            rows_split=rows_split,
            relevance_by_query=relevance_by_query,
            retrieve_k=cfg.retrieve_k,
            max_age_days=cfg.max_age_days,
        )

        baseline_ranker = ObjectiveRankerTrainer(
            objective=effective_objective,
            random_seed=cfg.random_seed,
        ).train(rows_by_id=rows_by_id, pair_rows=objective_pair_rows)
        baseline_ranker_eval = _evaluate_ranker_objective(
            ranker=baseline_ranker,
            objective=effective_objective,
            rows_by_id=rows_by_id,
            pair_rows=objective_pair_rows,
        )

        selected_pair_rows = list(objective_pair_rows)
        selected_ranker_eval = dict(baseline_ranker_eval)
        selected_variant = "baseline"
        adaptive_ranker_eval: Optional[Dict[str, float]] = None
        adaptive_gate: Dict[str, Any] = {
            "enabled": False,
            "selected": False,
            "reason": "adaptive_mining_disabled",
            "threshold_ratio": 0.995,
        }
        mining_diagnostics: Dict[str, Any] = {
            "enabled": False,
            "mode": "fixed_v1",
            "objective": effective_objective,
        }

        adaptive_enabled = (
            cfg.negative_mining_mode == "adaptive_v2"
            and bool(cfg.adaptive_negative_mining.enabled)
        )
        if adaptive_enabled:
            adaptive_gate["enabled"] = True
            miner = AdaptiveNegativeMiner(config=cfg.adaptive_negative_mining)
            mined_rows, mining_diagnostics = miner.mine(
                objective=effective_objective,
                target_source=cfg.pair_target_source,
                rows_by_id=rows_by_id,
                train_rows=rows_split["train"],
                base_pair_rows=objective_pair_rows,
                retriever=retriever,
                baseline_ranker=baseline_ranker,
                max_age_days=cfg.max_age_days,
            )
            augmented_pair_rows = [*objective_pair_rows, *mined_rows]
            adaptive_gate["mined_rows_total"] = len(mined_rows)
            if mined_rows:
                try:
                    adaptive_ranker = ObjectiveRankerTrainer(
                        objective=effective_objective,
                        random_seed=cfg.random_seed,
                    ).train(rows_by_id=rows_by_id, pair_rows=augmented_pair_rows)
                    adaptive_ranker_eval = _evaluate_ranker_objective(
                        ranker=adaptive_ranker,
                        objective=effective_objective,
                        rows_by_id=rows_by_id,
                        pair_rows=objective_pair_rows,
                    )
                    baseline_ndcg = float(baseline_ranker_eval.get("ndcg@10", 0.0))
                    baseline_mrr = float(baseline_ranker_eval.get("mrr@20", 0.0))
                    adaptive_ndcg = float(adaptive_ranker_eval.get("ndcg@10", 0.0))
                    adaptive_mrr = float(adaptive_ranker_eval.get("mrr@20", 0.0))
                    adaptive_ok = (
                        adaptive_ndcg >= (baseline_ndcg * 0.995)
                        and adaptive_mrr >= (baseline_mrr * 0.995)
                    )
                    adaptive_gate["baseline"] = {
                        "ndcg@10": baseline_ndcg,
                        "mrr@20": baseline_mrr,
                    }
                    adaptive_gate["adaptive"] = {
                        "ndcg@10": adaptive_ndcg,
                        "mrr@20": adaptive_mrr,
                    }
                    adaptive_gate["selected"] = bool(adaptive_ok)
                    adaptive_gate["reason"] = (
                        "adaptive_passed_gate" if adaptive_ok else "adaptive_failed_gate"
                    )
                    if adaptive_ok:
                        selected_pair_rows = list(augmented_pair_rows)
                        selected_ranker_eval = dict(adaptive_ranker_eval)
                        selected_variant = "adaptive"
                except Exception as error:
                    adaptive_gate["selected"] = False
                    adaptive_gate["reason"] = "adaptive_training_failed"
                    adaptive_gate["error"] = str(error)
            else:
                adaptive_gate["selected"] = False
                adaptive_gate["reason"] = "no_mined_rows"

        family_config = RankerFamilyConfig(
            objective=effective_objective,
            ensemble_size=cfg.ranker_ensemble_size,
            random_seed=cfg.random_seed,
            std_ref=cfg.ranker_uncertainty_std_ref,
            creator_cold_threshold=cfg.creator_cold_start_threshold,
        )
        global_ensemble = RankerEnsembleModel.train(
            config=family_config,
            rows_by_id=rows_by_id,
            pair_rows=selected_pair_rows,
            segment_id=SEGMENT_GLOBAL,
        )
        global_ranker_eval = _evaluate_ranker_objective(
            ranker=global_ensemble,
            objective=effective_objective,
            rows_by_id=rows_by_id,
            pair_rows=objective_pair_rows,
        )

        segment_support: Dict[str, Dict[str, int]] = {}
        segment_promotion: Dict[str, Dict[str, Any]] = {}
        segment_ensembles: Dict[str, RankerEnsembleModel] = {}
        promoted_segments: List[str] = []
        for segment_id in RANKER_SEGMENTS:
            support = _segment_support_counts(
                rows_by_id=rows_by_id,
                pair_rows=objective_pair_rows,
                objective=effective_objective,
                segment_id=segment_id,
                creator_cold_start_threshold=cfg.creator_cold_start_threshold,
            )
            segment_support[segment_id] = support
            eligible = (
                support["train_pairs"] >= cfg.segment_min_train_pairs
                and support["validation_pairs"] >= cfg.segment_min_validation_pairs
            )
            decision: Dict[str, Any] = {
                "eligible": eligible,
                "promoted": False,
                "reason": "insufficient_support",
                "support": support,
                "slice_global_metrics": None,
                "slice_segment_metrics": None,
                "overall_blended_metrics": None,
            }
            if not eligible:
                segment_promotion[segment_id] = decision
                continue

            try:
                segment_train_pairs = _pair_rows_for_segment(
                    rows_by_id=rows_by_id,
                    pair_rows=selected_pair_rows,
                    objective=effective_objective,
                    segment_id=segment_id,
                    creator_cold_start_threshold=cfg.creator_cold_start_threshold,
                )
                segment_ensemble = RankerEnsembleModel.train(
                    config=family_config,
                    rows_by_id=rows_by_id,
                    pair_rows=segment_train_pairs,
                    segment_id=segment_id,
                )
                segment_ensembles[segment_id] = segment_ensemble
                slice_global_eval = _evaluate_ranker_objective(
                    ranker=global_ensemble,
                    objective=effective_objective,
                    rows_by_id=rows_by_id,
                    pair_rows=objective_pair_rows,
                    segment_id=segment_id,
                    creator_cold_start_threshold=cfg.creator_cold_start_threshold,
                )
                slice_segment_eval = _evaluate_ranker_objective(
                    ranker=segment_ensemble,
                    objective=effective_objective,
                    rows_by_id=rows_by_id,
                    pair_rows=objective_pair_rows,
                    segment_id=segment_id,
                    creator_cold_start_threshold=cfg.creator_cold_start_threshold,
                )
                overall_blended_eval = _evaluate_segment_blended_objective(
                    global_ranker=global_ensemble,
                    segment_ranker=segment_ensemble,
                    objective=effective_objective,
                    rows_by_id=rows_by_id,
                    pair_rows=objective_pair_rows,
                    segment_id=segment_id,
                    creator_cold_start_threshold=cfg.creator_cold_start_threshold,
                )
                decision["slice_global_metrics"] = slice_global_eval
                decision["slice_segment_metrics"] = slice_segment_eval
                decision["overall_blended_metrics"] = overall_blended_eval

                slice_pass = (
                    float(slice_segment_eval.get("ndcg@10", 0.0))
                    >= (float(slice_global_eval.get("ndcg@10", 0.0)) * 1.005)
                    and float(slice_segment_eval.get("mrr@20", 0.0))
                    >= (float(slice_global_eval.get("mrr@20", 0.0)) * 1.005)
                )
                overall_pass = (
                    float(overall_blended_eval.get("ndcg@10", 0.0))
                    >= (float(global_ranker_eval.get("ndcg@10", 0.0)) * 0.995)
                    and float(overall_blended_eval.get("mrr@20", 0.0))
                    >= (float(global_ranker_eval.get("mrr@20", 0.0)) * 0.995)
                )
                decision["slice_gate"] = bool(slice_pass)
                decision["overall_gate"] = bool(overall_pass)
                if slice_pass and overall_pass:
                    promoted_segments.append(segment_id)
                    decision["promoted"] = True
                    decision["reason"] = "promotion_gate_passed"
                else:
                    decision["promoted"] = False
                    decision["reason"] = "promotion_gate_failed"
            except Exception as error:
                decision["promoted"] = False
                decision["reason"] = "segment_training_failed"
                decision["error"] = str(error)

            segment_promotion[segment_id] = decision

        ranker_family = RankerFamilyModel(
            objective=effective_objective,
            global_ensemble=global_ensemble,
            segment_ensembles=segment_ensembles,
            promoted_segments=promoted_segments,
            std_ref=cfg.ranker_uncertainty_std_ref,
            creator_cold_threshold=cfg.creator_cold_start_threshold,
        )
        ranker_output_dir = rankers_dir / effective_objective
        ranker_family.save(ranker_output_dir)
        calibration_samples = _collect_calibration_samples(
            ranker_family=ranker_family,
            objective=effective_objective,
            rows_by_id=rows_by_id,
            pair_rows=objective_pair_rows,
        )
        calibration_bundle = ObjectiveSegmentCalibrators.fit(
            objective=effective_objective,
            samples=calibration_samples,
            known_segments=(SEGMENT_GLOBAL, *RANKER_SEGMENTS),
            min_support=cfg.ranker_calibration_min_support,
            compatibility={
                "objective": effective_objective,
                "feature_schema_hash": feature_schema_hash,
                "ranker_family_schema_hash": _sha256_text(json.dumps(FEATURE_NAMES)),
                "ranker_family_version": "ranker_family.v2",
            },
        )
        calibration_path = ranker_output_dir / "calibration.json"
        calibration_bundle.save(calibration_path)
        calibration_fallback = _calibration_fallback_summary(
            calibration_bundle=calibration_bundle,
            samples=calibration_samples,
        )
        calibration_report_payload = {
            "objective": effective_objective,
            "target_definition": "p_relevance_ge_2",
            "validation_samples": len(calibration_samples),
            "summary": calibration_bundle.summary(),
            "fallback": calibration_fallback,
        }
        calibration_rel_path = f"diagnostics/objective_{effective_objective}_calibration.json"
        calibration_report_path = bundle_dir / calibration_rel_path
        calibration_text = json.dumps(calibration_report_payload, ensure_ascii=False, indent=2)
        calibration_report_path.write_text(calibration_text, encoding="utf-8")
        objective_calibration_manifest[effective_objective] = {
            "path": calibration_rel_path,
            "sha256": _sha256_text(calibration_text),
        }
        adaptive_selected_by_objective[effective_objective] = selected_variant == "adaptive"
        segment_support_by_objective[effective_objective] = segment_support
        segment_promotion_by_objective[effective_objective] = segment_promotion

        objective_diagnostics_payload = {
            "objective": effective_objective,
            "pair_target_source": cfg.pair_target_source,
            "negative_mining_mode": cfg.negative_mining_mode,
            "mining": mining_diagnostics,
            "gate": adaptive_gate,
            "ranker_baseline": baseline_ranker_eval,
            "ranker_adaptive": adaptive_ranker_eval,
            "ranker_selected": selected_ranker_eval,
            "selected_variant": selected_variant,
            "ranker_global": global_ranker_eval,
            "ranker_family": {
                "promoted_segments": promoted_segments,
                "segment_support": segment_support,
                "segment_promotion": segment_promotion,
                "creator_cold_start_threshold": cfg.creator_cold_start_threshold,
                "ensemble_size": cfg.ranker_ensemble_size,
                "uncertainty_std_ref": cfg.ranker_uncertainty_std_ref,
            },
            "calibration": {
                "summary": calibration_bundle.summary(),
                "fallback": calibration_fallback,
                "report_path": calibration_rel_path,
            },
        }
        diagnostics_rel_path = f"diagnostics/objective_{effective_objective}_negative_mining.json"
        diagnostics_path = bundle_dir / diagnostics_rel_path
        diagnostics_text = json.dumps(
            objective_diagnostics_payload,
            ensure_ascii=False,
            indent=2,
        )
        diagnostics_path.write_text(diagnostics_text, encoding="utf-8")
        objective_diagnostics_manifest[effective_objective] = {
            "path": diagnostics_rel_path,
            "sha256": _sha256_text(diagnostics_text),
        }

        objective_metrics[effective_objective] = {
            "spec": {
                "objective_id": spec.objective_id,
                "label_key": spec.label_key,
                "primary_metric": spec.primary_metric,
                "training_loss": spec.training_loss,
                "calibration": spec.calibration,
            },
            "retriever": retrieval_eval,
            "ranker": global_ranker_eval,
            "ranker_baseline": baseline_ranker_eval,
            "ranker_adaptive": adaptive_ranker_eval,
            "ranker_selected_variant": selected_variant,
            "adaptive_gate": adaptive_gate,
            "backend": global_ensemble.models[0].backend,
            "retriever_blend_weights": learned_weights,
            "ranker_family": {
                "global": global_ranker_eval,
                "promoted_segments": promoted_segments,
                "segment_support": segment_support,
                "segment_promotion": segment_promotion,
                "ensemble_size": cfg.ranker_ensemble_size,
                "uncertainty_std_ref": cfg.ranker_uncertainty_std_ref,
                "creator_cold_start_threshold": cfg.creator_cold_start_threshold,
            },
            "ranker_calibration": calibration_bundle.summary(),
            "ranker_calibration_fallback": calibration_fallback,
        }
        trained_objectives.append(effective_objective)

    retriever.save(retriever_dir)

    (metrics_dir / "objective_metrics.json").write_text(
        json.dumps(objective_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (diagnostics_dir / "negative_sampler.json").write_text(
        json.dumps(sample_diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fabric = FeatureFabric()
    fabric_schema_hashes = {
        name: fabric.registry.get(name).output_schema_hash
        for name in ("text", "structure", "audio", "visual")
    }
    manifest = {
        "component": "recommender-learning-v1",
        "contract_version": cfg.contract_version,
        "datamart_version": cfg.datamart_version,
        "feature_schema_hash": feature_schema_hash,
        "ranker_family_schema_hash": _sha256_text(json.dumps(FEATURE_NAMES)),
        "feature_manifest_id": datamart.get("source_manifest_id"),
        "feature_manifest_path": datamart.get("source_manifest_path"),
        "feature_snapshot_manifest_id": feature_manifest_id,
        "feature_snapshot_manifest_path": cfg.feature_snapshot_manifest_path,
        "graph": graph_manifest_payload,
        "graph_enabled": cfg.graph_enabled,
        "graph_embedding_dim": cfg.graph_embedding_dim,
        "graph_walk_params": cfg.graph_walk_params,
        "graph_weighting_params": cfg.graph_weighting_params,
        "trajectory": trajectory_manifest_payload,
        "trajectory_enabled": cfg.trajectory_enabled,
        "trajectory_embedding_dim": cfg.trajectory_embedding_dim,
        "trajectory_feature_version": cfg.trajectory_feature_version,
        "trajectory_branch_weight": cfg.trajectory_branch_weight,
        "trajectory_encoder_mode": cfg.trajectory_encoder_mode,
        "trajectory_manifest_path": cfg.trajectory_manifest_path,
        "trajectory_version": TRAJECTORY_VERSION,
        "comment_feature_manifest_id": datamart.get("comment_feature_manifest_id"),
        "comment_feature_manifest_path": datamart.get("comment_feature_manifest_path"),
        "comment_priors_manifest_id": datamart.get("comment_priors_manifest_id"),
        "comment_priors_manifest_path": datamart.get("comment_priors_manifest_path"),
        "comment_intelligence_version": datamart.get(
            "comment_intelligence_version", "comment_intelligence.v2"
        ),
        "fabric_version": "fabric.v2",
        "fabric_registry_signature": fabric.registry.signature(),
        "fabric_schema_hashes": fabric_schema_hashes,
        "objectives": trained_objectives,
        "retrieve_k": cfg.retrieve_k,
        "max_age_days": cfg.max_age_days,
        "dense_model_name": cfg.dense_model_name,
        "random_seed": cfg.random_seed,
        "pair_target_source": cfg.pair_target_source,
        "negative_mining_mode": cfg.negative_mining_mode,
        "adaptive_negative_mining_config": asdict(cfg.adaptive_negative_mining),
        "adaptive_selected_by_objective": adaptive_selected_by_objective,
        "ranker_family": {
            "version": "ranker_family.v2",
            "segments": list(RANKER_SEGMENTS),
            "creator_cold_start_threshold": cfg.creator_cold_start_threshold,
            "ensemble_size": cfg.ranker_ensemble_size,
            "uncertainty_std_ref": cfg.ranker_uncertainty_std_ref,
            "ranker_calibration_min_support": cfg.ranker_calibration_min_support,
            "segment_support_by_objective": segment_support_by_objective,
            "segment_promotion_by_objective": segment_promotion_by_objective,
        },
        "ranker_calibration": {
            "version": "ranker_calibration.v2",
            "min_support": cfg.ranker_calibration_min_support,
        },
        "policy_reranker": PolicyRerankerConfig().to_payload(),
        "objective_diagnostics": objective_diagnostics_manifest,
        "objective_calibration_reports": objective_calibration_manifest,
        "rows_total": len(rows),
        "pair_rows_total": len(pair_rows),
        "train_rows": len(rows_split["train"]),
        "validation_rows": len(rows_split["validation"]),
        "test_rows": len(rows_split["test"]),
        "retriever": {
            "artifact_version": retriever.artifact_version,
            "sparse_backend": retriever.sparse_backend,
            "dense_backend": retriever.dense_backend,
            "multimodal_backend": retriever.multimodal_backend,
            "graph_backend": retriever.graph_backend,
            "trajectory_backend": retriever.trajectory_backend,
            "objective_blend": learned_objective_blend or retriever.objective_blend,
            "temporal_shards": sorted(set(retriever.row_shards.values())),
            "index_cutoff_time": _to_utc_iso(datamart.get("generated_at")),
            "graph_bundle_id": retriever.graph_bundle_id,
            "graph_version": retriever.graph_version,
            "trajectory_manifest_id": retriever.trajectory_manifest_id,
            "trajectory_version": retriever.trajectory_version,
        },
    }
    registry.write_manifest(bundle_dir, manifest)

    return {
        "bundle_dir": str(bundle_dir),
        "manifest": manifest,
        "objective_metrics": objective_metrics,
    }
