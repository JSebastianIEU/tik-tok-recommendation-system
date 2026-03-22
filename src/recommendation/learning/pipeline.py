from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .artifacts import ArtifactRegistry
from .evaluator import evaluate_ranking, evaluate_retrieval
from .objectives import OBJECTIVE_SPECS, map_objective
from .ranker import FEATURE_NAMES, ObjectiveRankerConfig, ObjectiveRankerModel
from .retriever import HybridRetriever, HybridRetrieverTrainerConfig
from .sampling import NegativeSampler, NegativeSamplerConfig
from .temporal import TemporalCandidatePool, TemporalCandidatePoolConfig, split_rows


def _to_utc_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


@dataclass
class RecommenderTrainingConfig:
    objectives: Sequence[str] = ("reach", "engagement", "conversion")
    retrieve_k: int = 200
    max_age_days: int = 180
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    random_seed: int = 13
    run_name: str = "recommender-v1"
    contract_version: str = "contract.v1"
    datamart_version: str = "datamart.v1"


class HybridRetrieverTrainer:
    def __init__(
        self,
        config: Optional[HybridRetrieverTrainerConfig] = None,
    ) -> None:
        self.config = config or HybridRetrieverTrainerConfig()

    def train(self, rows: Sequence[Dict[str, Any]]) -> HybridRetriever:
        return HybridRetriever.train(rows=rows, config=self.config)


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
) -> List[Dict[str, Any]]:
    direct = [row for row in pair_rows if str(row.get("objective")) == objective]
    if direct:
        return direct

    out: List[Dict[str, Any]] = []
    for pair in pair_rows:
        candidate_id = str(pair.get("candidate_row_id"))
        candidate_row = rows_by_id.get(candidate_id)
        if candidate_row is None:
            continue
        score = float(
            (
                candidate_row.get("targets_z", {})
                if isinstance(candidate_row.get("targets_z"), dict)
                else {}
            ).get(objective, 0.0)
        )
        out.append(
            {
                **pair,
                "objective": objective,
                "objective_score": round(score, 6),
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
        )
        query_payloads.append(
            {
                "query_id": query_id,
                "items": items,
                "relevant_ids": relevant_ids,
            }
        )
    return evaluate_retrieval(query_payloads, top_k=retrieve_k)


def _evaluate_ranker_objective(
    ranker: ObjectiveRankerModel,
    objective: str,
    rows_by_id: Dict[str, Dict[str, Any]],
    pair_rows: Sequence[Dict[str, Any]],
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
        feature_vector = np.asarray(
            [
                [
                    float(pair.get("similarity") or 0.0),
                    float(query_row.get("features", {}).get("caption_word_count") or 0.0),
                    float(query_row.get("features", {}).get("hashtag_count") or 0.0),
                    float(query_row.get("features", {}).get("keyword_count") or 0.0),
                    float(candidate_row.get("features", {}).get("caption_word_count") or 0.0),
                    float(candidate_row.get("features", {}).get("hashtag_count") or 0.0),
                    float(candidate_row.get("features", {}).get("keyword_count") or 0.0),
                    abs(
                        float(query_row.get("features", {}).get("caption_word_count") or 0.0)
                        - float(candidate_row.get("features", {}).get("caption_word_count") or 0.0)
                    ),
                    abs(
                        float(query_row.get("features", {}).get("hashtag_count") or 0.0)
                        - float(candidate_row.get("features", {}).get("hashtag_count") or 0.0)
                    ),
                    abs(
                        float(query_row.get("features", {}).get("keyword_count") or 0.0)
                        - float(candidate_row.get("features", {}).get("keyword_count") or 0.0)
                    ),
                    1.0 if query_row.get("author_id") == candidate_row.get("author_id") else 0.0,
                    1.0 if query_row.get("topic_key") == candidate_row.get("topic_key") else 0.0,
                ]
            ],
            dtype=np.float32,
        )
        score = float(ranker.predict_scores(feature_vector)[0])
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

    retriever = HybridRetrieverTrainer(
        HybridRetrieverTrainerConfig(dense_model_name=cfg.dense_model_name)
    ).train(rows_split["train"])
    retriever.save(retriever_dir)

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
    for objective in cfg.objectives:
        _, effective_objective = map_objective(objective)
        if effective_objective in trained_objectives:
            continue
        objective_pair_rows = _pair_rows_for_objective(
            rows_by_id=rows_by_id,
            pair_rows=pair_rows,
            objective=effective_objective,
        )
        spec = OBJECTIVE_SPECS[effective_objective]
        ranker = ObjectiveRankerTrainer(
            objective=effective_objective,
            random_seed=cfg.random_seed,
        ).train(rows_by_id=rows_by_id, pair_rows=objective_pair_rows)
        ranker_output_dir = rankers_dir / effective_objective
        ranker.save(ranker_output_dir)
        relevance_by_query = _group_relevance_by_query(
            pair_rows=objective_pair_rows,
            objective=effective_objective,
        )
        retrieval_eval = _evaluate_retriever_objective(
            retriever=retriever,
            objective=effective_objective,
            rows_split=rows_split,
            relevance_by_query=relevance_by_query,
            retrieve_k=cfg.retrieve_k,
            max_age_days=cfg.max_age_days,
        )
        ranker_eval = _evaluate_ranker_objective(
            ranker=ranker,
            objective=effective_objective,
            rows_by_id=rows_by_id,
            pair_rows=objective_pair_rows,
        )
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
            "backend": ranker.backend,
        }
        trained_objectives.append(effective_objective)

    (metrics_dir / "objective_metrics.json").write_text(
        json.dumps(objective_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (diagnostics_dir / "negative_sampler.json").write_text(
        json.dumps(sample_diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    feature_schema_hash = registry.feature_schema_hash(FEATURE_NAMES)
    manifest = {
        "component": "recommender-learning-v1",
        "contract_version": cfg.contract_version,
        "datamart_version": cfg.datamart_version,
        "feature_schema_hash": feature_schema_hash,
        "objectives": trained_objectives,
        "retrieve_k": cfg.retrieve_k,
        "dense_model_name": cfg.dense_model_name,
        "random_seed": cfg.random_seed,
        "rows_total": len(rows),
        "pair_rows_total": len(pair_rows),
        "train_rows": len(rows_split["train"]),
        "validation_rows": len(rows_split["validation"]),
        "test_rows": len(rows_split["test"]),
        "retriever": {
            "sparse_backend": retriever.sparse_backend,
            "dense_backend": retriever.dense_backend,
            "index_cutoff_time": _to_utc_iso(datamart.get("generated_at")),
        },
    }
    registry.write_manifest(bundle_dir, manifest)

    return {
        "bundle_dir": str(bundle_dir),
        "manifest": manifest,
        "objective_metrics": objective_metrics,
    }
