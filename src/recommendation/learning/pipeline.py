from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .artifacts import ArtifactRegistry
from .baseline_common import (
    BASELINE_GRAPH_VERSION,
    BASELINE_RANKER_VERSION,
    BASELINE_RETRIEVER_VERSION,
    BASELINE_TRAJECTORY_VERSION,
    DEFAULT_RANKING_WEIGHTS,
    OBJECTIVE_RANKING_WEIGHTS,
    round_score,
)
from .candidate_support import prepare_candidate
from .evaluator import evaluate_ranking, evaluate_retrieval, recall_at_k
from .graph import (
    GraphBuildConfig,
    annotate_rows_with_graph_features,
    build_creator_video_dna_graph,
)
from .objectives import OBJECTIVE_SPECS, map_objective
from .policy import PolicyRerankerConfig
from .query_contract import build_query_profile
from .ranking_baseline import rank_shortlist
from .retrieval_baseline import retrieve_shortlist
from .retriever import HybridRetriever, HybridRetrieverTrainerConfig
from .sampling import NegativeSampler, NegativeSamplerConfig
from .temporal import TemporalCandidatePool, TemporalCandidatePoolConfig, parse_dt, row_text, split_rows
from .trajectory import (
    TRAJECTORY_VERSION,
    TrajectoryBuildConfig,
    TrajectoryBundle,
    annotate_rows_with_trajectory_features,
    build_trajectory_bundle,
)
from ..fabric import FeatureFabric

VALID_PAIR_TARGET_SOURCES = {"scalar_v1", "trajectory_v2_composite"}
RETRIEVAL_BRANCH_KEYS = (
    "lexical",
    "dense_text",
    "multimodal",
    "graph_dense",
    "trajectory_dense",
)
BASELINE_RANKER_FEATURE_NAMES = [
    "semantic_relevance",
    "intent_alignment",
    "reference_usefulness",
    "support_confidence",
    "retrieval_fused",
    "support_score",
]


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
        if self.retrieve_k < 1:
            raise ValueError("retrieve_k must be >= 1.")
        if self.max_age_days < 1:
            raise ValueError("max_age_days must be >= 1.")
        if self.graph_embedding_dim < 4:
            raise ValueError("graph_embedding_dim must be >= 4.")
        if self.trajectory_embedding_dim < 4:
            raise ValueError("trajectory_embedding_dim must be >= 4.")
        if self.trajectory_encoder_mode not in {"feature_only", "sequence_encoder_shadow"}:
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
    try:
        return TrajectoryBundle.load(manifest_path.parent)
    except Exception:
        return None


def _rows_by_id(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("row_id")): row for row in rows}


def _group_relevance_by_query(
    pair_rows: Sequence[Dict[str, Any]],
    objective: str,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for pair in pair_rows:
        if str(pair.get("objective")) != objective:
            continue
        query_id = str(pair.get("query_row_id") or "").strip()
        candidate_id = str(pair.get("candidate_row_id") or "").strip()
        if not query_id or not candidate_id:
            continue
        out.setdefault(query_id, {})[candidate_id] = float(pair.get("relevance_label") or 0.0)
    return out


def _normalize_retrieval_weights(weights: Dict[str, float]) -> Dict[str, float]:
    normalized = {
        key: max(0.0, float(weights.get(key, 0.0)))
        for key in RETRIEVAL_BRANCH_KEYS
    }
    denom = sum(normalized.values())
    if denom <= 0.0:
        uniform = 1.0 / float(len(RETRIEVAL_BRANCH_KEYS))
        return {key: uniform for key in RETRIEVAL_BRANCH_KEYS}
    return {key: value / denom for key, value in normalized.items()}


def _sparse_only_retrieval_weights() -> Dict[str, float]:
    return {
        "lexical": 1.0,
        "dense_text": 0.0,
        "multimodal": 0.0,
        "graph_dense": 0.0,
        "trajectory_dense": 0.0,
    }


def _select_retriever_weight_variant(
    *,
    learned_weights: Dict[str, float],
    learned_validation: Dict[str, float],
    sparse_baseline_weights: Dict[str, float],
    sparse_validation: Dict[str, float],
    learned_test: Optional[Dict[str, float]] = None,
    sparse_test: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    learned_val = float(learned_validation.get("recall@100") or 0.0)
    sparse_val = float(sparse_validation.get("recall@100") or 0.0)
    selected_variant = "learned" if learned_val >= sparse_val else "sparse_baseline"
    selected_weights = (
        dict(learned_weights)
        if selected_variant == "learned"
        else dict(sparse_baseline_weights)
    )
    selected_val = learned_val if selected_variant == "learned" else sparse_val
    competitor_val = sparse_val if selected_variant == "learned" else learned_val
    validation_non_regression = selected_val >= competitor_val

    test_non_regression: Optional[bool] = None
    if isinstance(learned_test, dict) and isinstance(sparse_test, dict):
        learned_test_val = float(learned_test.get("recall@100") or 0.0)
        sparse_test_val = float(sparse_test.get("recall@100") or 0.0)
        selected_test_val = (
            learned_test_val if selected_variant == "learned" else sparse_test_val
        )
        competitor_test_val = (
            sparse_test_val if selected_variant == "learned" else learned_test_val
        )
        test_non_regression = selected_test_val >= competitor_test_val

    return {
        "metric_key": "recall@100",
        "selected_variant": selected_variant,
        "selected_weights": _normalize_retrieval_weights(selected_weights),
        "validation": {
            "learned": {
                key: float(learned_validation.get(key) or 0.0)
                for key in ("recall@50", "recall@100", "recall@200")
            },
            "sparse_baseline": {
                key: float(sparse_validation.get(key) or 0.0)
                for key in ("recall@50", "recall@100", "recall@200")
            },
            "selected_not_worse_than_competitor": bool(validation_non_regression),
        },
        "test": {
            "learned": (
                {
                    key: float(learned_test.get(key) or 0.0)
                    for key in ("recall@50", "recall@100", "recall@200")
                }
                if isinstance(learned_test, dict)
                else None
            ),
            "sparse_baseline": (
                {
                    key: float(sparse_test.get(key) or 0.0)
                    for key in ("recall@50", "recall@100", "recall@200")
                }
                if isinstance(sparse_test, dict)
                else None
            ),
            "selected_not_worse_than_competitor": test_non_regression,
        },
    }


def _branch_dropout_weights(
    base_weights: Dict[str, float],
    dropped_branches: Sequence[str],
) -> Dict[str, float]:
    dropped = {str(item) for item in dropped_branches}
    proposal = {}
    for key in RETRIEVAL_BRANCH_KEYS:
        if key in dropped:
            proposal[key] = 0.0
            continue
        proposal[key] = float(base_weights.get(key, 0.0))

    surviving_keys = [key for key in RETRIEVAL_BRANCH_KEYS if key not in dropped]
    surviving_total = sum(float(proposal.get(key, 0.0)) for key in surviving_keys)
    if surviving_total <= 0.0:
        if not surviving_keys:
            return {key: 0.0 for key in RETRIEVAL_BRANCH_KEYS}
        uniform = 1.0 / float(len(surviving_keys))
        return {
            key: (0.0 if key in dropped else uniform)
            for key in RETRIEVAL_BRANCH_KEYS
        }

    normalized = _normalize_retrieval_weights(proposal)
    for key in dropped:
        if key in normalized:
            normalized[key] = 0.0
    surviving_norm = sum(normalized.get(key, 0.0) for key in surviving_keys)
    if surviving_norm > 0.0:
        for key in surviving_keys:
            normalized[key] = float(normalized.get(key, 0.0)) / surviving_norm
    return normalized


def _retriever_ablation_variants(
    base_weights: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    variants: Dict[str, Dict[str, Any]] = {
        "full": {
            "dropped_branches": [],
            "weights": _normalize_retrieval_weights(base_weights),
        }
    }
    for branch in RETRIEVAL_BRANCH_KEYS:
        variants[f"no_{branch}"] = {
            "dropped_branches": [branch],
            "weights": _branch_dropout_weights(base_weights, [branch]),
        }
    return variants


def _metric_delta_block(
    *,
    baseline_metrics: Dict[str, float],
    variant_metrics: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key in ("recall@50", "recall@100", "recall@200"):
        baseline = float(baseline_metrics.get(key) or 0.0)
        variant = float(variant_metrics.get(key) or 0.0)
        delta = variant - baseline
        out[key] = {
            "baseline": round(baseline, 6),
            "variant": round(variant, 6),
            "delta": round(delta, 6),
            "delta_pp": round(delta * 100.0, 3),
            "relative_delta_pct": round(((delta / baseline) * 100.0) if baseline > 0.0 else 0.0, 3),
        }
    return out


def _pair_rows_for_objective(
    pair_rows: Sequence[Dict[str, Any]],
    objective: str,
    target_source: str,
) -> List[Dict[str, Any]]:
    direct = [
        row
        for row in pair_rows
        if str(row.get("objective")) == objective
        and str(row.get("target_source") or "scalar_v1") == target_source
    ]
    if direct:
        return direct
    return [row for row in pair_rows if str(row.get("objective")) == objective]


def _evaluate_retriever_objective(
    retriever: HybridRetriever,
    objective: str,
    rows_split: Dict[str, List[Dict[str, Any]]],
    relevance_by_query: Dict[str, Dict[str, float]],
    retrieve_k: int,
    max_age_days: int,
    weight_override: Optional[Dict[str, float]] = None,
    eval_split: Optional[str] = None,
) -> Dict[str, float]:
    pool_builder = TemporalCandidatePool(
        TemporalCandidatePoolConfig(
            max_age_days=max_age_days,
            min_pool_size=30,
            enforce_index_cutoff=True,
        )
    )
    if eval_split in {"train", "validation", "test"}:
        eval_queries = list(rows_split.get(str(eval_split), []))
    else:
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
            weight_override=weight_override,
        )
        query_payloads.append(
            {
                "query_id": query_id,
                "items": items,
                "relevant_ids": relevant_ids,
            }
        )
    return evaluate_retrieval(query_payloads, top_k=retrieve_k)


def _evaluate_retriever_ablation_objective(
    *,
    retriever: HybridRetriever,
    objective: str,
    rows_split: Dict[str, List[Dict[str, Any]]],
    relevance_by_query: Dict[str, Dict[str, float]],
    retrieve_k: int,
    max_age_days: int,
) -> Dict[str, Any]:
    base_weights = retriever.branch_weights(objective)
    variants = _retriever_ablation_variants(base_weights)
    evaluated: Dict[str, Any] = {}
    for name, variant in variants.items():
        weights = variant["weights"]
        metrics = _evaluate_retriever_objective(
            retriever=retriever,
            objective=objective,
            rows_split=rows_split,
            relevance_by_query=relevance_by_query,
            retrieve_k=retrieve_k,
            max_age_days=max_age_days,
            weight_override=weights,
        )
        evaluated[name] = {
            "dropped_branches": list(variant["dropped_branches"]),
            "weights": {key: round(float(weights.get(key, 0.0)), 6) for key in RETRIEVAL_BRANCH_KEYS},
            "metrics": {
                "recall@50": round(float(metrics.get("recall@50") or 0.0), 6),
                "recall@100": round(float(metrics.get("recall@100") or 0.0), 6),
                "recall@200": round(float(metrics.get("recall@200") or 0.0), 6),
            },
        }

    baseline_metrics = dict(evaluated["full"]["metrics"])
    for payload in evaluated.values():
        payload["delta_vs_full"] = _metric_delta_block(
            baseline_metrics=baseline_metrics,
            variant_metrics=payload["metrics"],
        )

    branch_lift = []
    for branch in RETRIEVAL_BRANCH_KEYS:
        variant = evaluated.get(f"no_{branch}")
        if not isinstance(variant, dict):
            continue
        delta = (
            variant.get("delta_vs_full", {})
            .get("recall@100", {})
            .get("delta")
        )
        branch_lift.append(
            {
                "branch": branch,
                "recall@100_delta_without_branch": round(float(delta or 0.0), 6),
            }
        )
    branch_lift.sort(key=lambda item: float(item["recall@100_delta_without_branch"]))

    return {
        "metric_key": "recall@100",
        "variants": evaluated,
        "branch_importance": branch_lift,
    }


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


def _row_signal_hints(row: Dict[str, Any]) -> Dict[str, Any]:
    features = row.get("features")
    if not isinstance(features, dict):
        return {}
    out: Dict[str, Any] = {}
    comment_intelligence = features.get("comment_intelligence")
    if isinstance(comment_intelligence, dict):
        out["comment_intelligence"] = comment_intelligence
    trajectory = features.get("trajectory_features")
    if isinstance(trajectory, dict):
        out["trajectory_features"] = trajectory
    return out


def _row_candidate_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    topic_key = str(row.get("topic_key") or "").strip()
    features = row.get("features")
    if not isinstance(features, dict):
        features = {}
    payload = {
        "candidate_id": str(row.get("row_id") or ""),
        "row_id": str(row.get("row_id") or ""),
        "video_id": str(row.get("video_id") or ""),
        "author_id": str(row.get("author_id") or ""),
        "text": row_text(row),
        "caption": str(row.get("caption") or row_text(row)),
        "hashtags": list(row.get("hashtags") or ([] if not topic_key else [f"#{topic_key}"])),
        "keywords": list(row.get("keywords") or ([] if not topic_key else [topic_key])),
        "search_query": row.get("search_query"),
        "posted_at": row.get("posted_at"),
        "as_of_time": row.get("as_of_time"),
        "language": row.get("language") or features.get("language"),
        "locale": row.get("locale"),
        "content_type": row.get("content_type"),
        "topic_key": topic_key or None,
    }
    signal_hints = _row_signal_hints(row)
    if signal_hints:
        payload["signal_hints"] = signal_hints
    return payload


def _row_query_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    topic_key = str(row.get("topic_key") or "").strip()
    features = row.get("features")
    if not isinstance(features, dict):
        features = {}
    return {
        "query_id": str(row.get("row_id") or "query"),
        "text": row_text(row),
        "description": str(row.get("caption") or row_text(row)),
        "hashtags": list(row.get("hashtags") or ([] if not topic_key else [f"#{topic_key}"])),
        "mentions": [],
        "content_type": row.get("content_type"),
        "primary_cta": "none",
        "language": row.get("language") or features.get("language"),
        "locale": row.get("locale"),
        "topic_key": topic_key or None,
        "keywords": list(row.get("keywords") or []),
    }


def _evaluate_baseline_ranker_objective(
    *,
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
    query_payloads: List[Dict[str, Any]] = []
    for query_row in rows_split["validation"] + rows_split["test"]:
        query_id = str(query_row.get("row_id") or "")
        if not query_id:
            continue
        relevance = relevance_by_query.get(query_id, {})
        if not relevance:
            continue
        query_as_of = parse_dt(query_row.get("as_of_time"))
        if query_as_of is None:
            continue
        query_payload = _row_query_payload(query_row)
        query_profile = build_query_profile(
            objective=objective,
            query=query_payload,
            fallback_language=query_payload.get("language"),
            fallback_locale=query_payload.get("locale"),
            fallback_content_type=query_payload.get("content_type"),
        )
        pool = pool_builder.for_query(
            query_row=query_row,
            candidate_rows=rows_split["train"] + rows_split["validation"],
            index_cutoff_time=query_row.get("as_of_time"),
        )
        prepared: List[Dict[str, Any]] = []
        for candidate_row in pool:
            prepared_candidate = prepare_candidate(
                payload=_row_candidate_payload(candidate_row),
                as_of=query_as_of,
                query_profile=query_profile,
                manifest_comment_lookup=lambda _row_id, _point_in_time: None,
            )
            if prepared_candidate is None or prepared_candidate.get("support_level") == "low":
                continue
            prepared.append(prepared_candidate)
        shortlist, _ = retrieve_shortlist(
            usable_candidates=prepared,
            query_profile=query_profile,
            retrieve_k=retrieve_k,
        )
        ranked, _ = rank_shortlist(
            shortlist=shortlist,
            query_profile=query_profile,
            effective_objective=objective,
            portfolio=None,
            rankers_available=[objective],
        )
        items = [
            {
                "candidate_row_id": str(item.get("candidate_id") or ""),
                "score": float(item.get("score") or 0.0),
                "relevance": float(relevance.get(str(item.get("candidate_id") or ""), 0.0)),
            }
            for item in ranked
            if str(item.get("candidate_id") or "")
        ]
        if items:
            query_payloads.append({"query_id": query_id, "items": items})
    return evaluate_ranking(query_payloads, k_values=(10, 20))


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
            "video": {key: list(value) for key, value in graph_bundle.video_embeddings.items()},
            "creator": {key: list(value) for key, value in graph_bundle.creator_embeddings.items()},
            "hashtag": {key: list(value) for key, value in graph_bundle.hashtag_embeddings.items()},
            "audio_motif": {key: list(value) for key, value in graph_bundle.audio_embeddings.items()},
            "style_signature": {key: list(value) for key, value in graph_bundle.style_embeddings.items()},
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

    # Negative-sampling diagnostics are retained as a utility trace even though
    # the old learned ranker family has been removed.
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
        negatives = NegativeSampler(
            NegativeSamplerConfig(seed=cfg.random_seed)
        ).sample(query_row=query_row, positives=positives, candidate_pool=pool)
        sample_diagnostics = {
            "query_row_id": query_row.get("row_id"),
            "pool_size": len(pool),
            "sample_positive_count": len(positives),
            "sample_negative_count": len(negatives),
            "policy": asdict(NegativeSamplerConfig(seed=cfg.random_seed)),
        }

    ranker_feature_schema_hash = registry.feature_schema_hash(BASELINE_RANKER_FEATURE_NAMES)
    objective_metrics: Dict[str, Dict[str, Any]] = {}
    objective_diagnostics_manifest: Dict[str, Dict[str, str]] = {}
    objective_ablation_manifest: Dict[str, Dict[str, str]] = {}
    trained_objectives: List[str] = []
    learned_objective_blend: Dict[str, Dict[str, float]] = {}

    for objective in cfg.objectives:
        _, effective_objective = map_objective(objective)
        if effective_objective in trained_objectives:
            continue

        objective_pair_rows = _pair_rows_for_objective(
            pair_rows=pair_rows,
            objective=effective_objective,
            target_source=cfg.pair_target_source,
        )
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
        sparse_baseline_weights = _sparse_only_retrieval_weights()
        learned_validation_eval = _evaluate_retriever_objective(
            retriever=retriever,
            objective=effective_objective,
            rows_split=rows_split,
            relevance_by_query=relevance_by_query,
            retrieve_k=cfg.retrieve_k,
            max_age_days=cfg.max_age_days,
            weight_override=learned_weights,
            eval_split="validation",
        )
        sparse_validation_eval = _evaluate_retriever_objective(
            retriever=retriever,
            objective=effective_objective,
            rows_split=rows_split,
            relevance_by_query=relevance_by_query,
            retrieve_k=cfg.retrieve_k,
            max_age_days=cfg.max_age_days,
            weight_override=sparse_baseline_weights,
            eval_split="validation",
        )
        learned_test_eval = _evaluate_retriever_objective(
            retriever=retriever,
            objective=effective_objective,
            rows_split=rows_split,
            relevance_by_query=relevance_by_query,
            retrieve_k=cfg.retrieve_k,
            max_age_days=cfg.max_age_days,
            weight_override=learned_weights,
            eval_split="test",
        )
        sparse_test_eval = _evaluate_retriever_objective(
            retriever=retriever,
            objective=effective_objective,
            rows_split=rows_split,
            relevance_by_query=relevance_by_query,
            retrieve_k=cfg.retrieve_k,
            max_age_days=cfg.max_age_days,
            weight_override=sparse_baseline_weights,
            eval_split="test",
        )
        retriever_weight_gate = _select_retriever_weight_variant(
            learned_weights=learned_weights,
            learned_validation=learned_validation_eval,
            sparse_baseline_weights=sparse_baseline_weights,
            sparse_validation=sparse_validation_eval,
            learned_test=learned_test_eval,
            sparse_test=sparse_test_eval,
        )
        selected_retriever_weights = dict(retriever_weight_gate["selected_weights"])
        learned_objective_blend[effective_objective] = selected_retriever_weights
        retriever.set_objective_blend({effective_objective: selected_retriever_weights})

        retrieval_eval = _evaluate_retriever_objective(
            retriever=retriever,
            objective=effective_objective,
            rows_split=rows_split,
            relevance_by_query=relevance_by_query,
            retrieve_k=cfg.retrieve_k,
            max_age_days=cfg.max_age_days,
        )
        retriever_ablation = _evaluate_retriever_ablation_objective(
            retriever=retriever,
            objective=effective_objective,
            rows_split=rows_split,
            relevance_by_query=relevance_by_query,
            retrieve_k=cfg.retrieve_k,
            max_age_days=cfg.max_age_days,
        )
        ranker_eval = _evaluate_baseline_ranker_objective(
            objective=effective_objective,
            rows_split=rows_split,
            relevance_by_query=relevance_by_query,
            retrieve_k=cfg.retrieve_k,
            max_age_days=cfg.max_age_days,
        )

        ranker_output_dir = rankers_dir / effective_objective
        ranker_output_dir.mkdir(parents=True, exist_ok=True)
        baseline_manifest = {
            "ranker_id": "baseline_weighted",
            "version": BASELINE_RANKER_VERSION,
            "objective": effective_objective,
            "feature_schema_hash": ranker_feature_schema_hash,
            "weights": OBJECTIVE_RANKING_WEIGHTS.get(
                effective_objective,
                DEFAULT_RANKING_WEIGHTS,
            ),
            "retriever_blend_weights": selected_retriever_weights,
        }
        (ranker_output_dir / "baseline_manifest.json").write_text(
            json.dumps(baseline_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        diagnostics_payload = {
            "objective": effective_objective,
            "ranker_id": "baseline_weighted",
            "ranker_version": BASELINE_RANKER_VERSION,
            "feature_schema_hash": ranker_feature_schema_hash,
            "retriever_weight_gate": retriever_weight_gate,
            "retriever_ablation": retriever_ablation,
            "retriever_eval": retrieval_eval,
            "ranker_eval": ranker_eval,
            "pair_row_count": len(objective_pair_rows),
            "pair_target_source": cfg.pair_target_source,
        }
        diagnostics_rel_path = f"diagnostics/objective_{effective_objective}_baseline.json"
        diagnostics_path = bundle_dir / diagnostics_rel_path
        diagnostics_text = json.dumps(diagnostics_payload, ensure_ascii=False, indent=2)
        diagnostics_path.write_text(diagnostics_text, encoding="utf-8")
        objective_diagnostics_manifest[effective_objective] = {
            "path": diagnostics_rel_path,
            "sha256": _sha256_text(diagnostics_text),
        }

        ablation_rel_path = f"diagnostics/objective_{effective_objective}_ablation.json"
        ablation_payload = {
            "objective": effective_objective,
            "retriever_weight_gate": retriever_weight_gate,
            "retriever_ablation": retriever_ablation,
            "retriever_blend_weights": selected_retriever_weights,
            "retriever_eval": retrieval_eval,
            "ranker_eval": ranker_eval,
        }
        ablation_text = json.dumps(ablation_payload, ensure_ascii=False, indent=2)
        (bundle_dir / ablation_rel_path).write_text(ablation_text, encoding="utf-8")
        objective_ablation_manifest[effective_objective] = {
            "path": ablation_rel_path,
            "sha256": _sha256_text(ablation_text),
        }

        spec = OBJECTIVE_SPECS[effective_objective]
        objective_metrics[effective_objective] = {
            "spec": {
                "objective_id": spec.objective_id,
                "label_key": spec.label_key,
                "primary_metric": spec.primary_metric,
                "training_loss": spec.training_loss,
                "calibration": spec.calibration,
            },
            "retriever": retrieval_eval,
            "ranker": ranker_eval,
            "backend": "baseline_weighted",
            "retriever_blend_weights": selected_retriever_weights,
            "retriever_weight_gate": retriever_weight_gate,
            "retriever_ablation": retriever_ablation,
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
        "feature_schema_hash": ranker_feature_schema_hash,
        "ranker_family_schema_hash": ranker_feature_schema_hash,
        "ranker_family_version": BASELINE_RANKER_VERSION,
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
        "policy_reranker": PolicyRerankerConfig().to_payload(),
        "objective_diagnostics": objective_diagnostics_manifest,
        "objective_ablation_reports": objective_ablation_manifest,
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
            "graph_version": retriever.graph_version or BASELINE_GRAPH_VERSION,
            "trajectory_manifest_id": retriever.trajectory_manifest_id,
            "trajectory_version": retriever.trajectory_version or BASELINE_TRAJECTORY_VERSION,
        },
    }
    registry.write_manifest(bundle_dir, manifest)

    return {
        "bundle_dir": str(bundle_dir),
        "manifest": manifest,
        "objective_metrics": objective_metrics,
    }
