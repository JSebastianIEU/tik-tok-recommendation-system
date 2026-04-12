from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    lgb = None

from .artifacts import ArtifactRegistry
from .baseline_common import as_float, round_score, sanitize_probability


LEARNED_RERANKER_VERSION = "recommender.ranker.learned_pairwise.v1"
LEARNED_RERANKER_ID = "learned_pairwise_lgbm"
LEARNED_RERANKER_LABEL_POLICY_VERSION = "pairwise_supervision.v2"
LEARNED_RERANKER_MIN_SHORTLIST_SIGNAL = 0.06
LEARNED_RERANKER_FULL_AUTHORITY_PAIR_COUNT = 1500.0
LEARNED_RERANKER_MAX_RESIDUAL_SHIFT = 0.12
LEARNED_RERANKER_BASELINE_HEAD_GUARD_RANK = 3
LEARNED_RERANKER_HEAD_GUARD_MIN_AUTHORITY = 0.75

LEARNED_RERANKER_FEATURE_NAMES = [
    "baseline_score",
    "score_component_semantic_relevance",
    "score_component_intent_alignment",
    "score_component_performance_quality",
    "score_component_reference_usefulness",
    "score_component_support_confidence",
    "retrieval_semantic",
    "retrieval_hashtag_topic",
    "retrieval_structured_compatibility",
    "retrieval_fused",
    "similarity_sparse",
    "similarity_dense",
    "similarity_fused",
    "support_score",
    "support_is_full",
    "support_is_partial",
    "confidence",
    "comment_alignment_score",
    "comment_value_prop_coverage",
    "comment_on_topic_ratio",
    "comment_artifact_drift_ratio",
    "comment_alignment_confidence",
    "trajectory_similarity",
    "trajectory_regime_confidence",
    "has_reason_semantic",
    "has_reason_intent",
    "has_reason_performance",
    "has_reason_reference",
    "has_reason_support",
    "has_reason_multi_branch",
    "has_reason_full_support",
    "retrieval_branch_count",
]


def learned_reranker_feature_schema_hash() -> str:
    return ArtifactRegistry(Path(".")).feature_schema_hash(LEARNED_RERANKER_FEATURE_NAMES)


def _bool_flag(value: bool) -> float:
    return 1.0 if bool(value) else 0.0


def _authority_from_pair_count(pair_count: Any) -> float:
    count = max(0.0, float(pair_count or 0.0))
    if LEARNED_RERANKER_FULL_AUTHORITY_PAIR_COUNT <= 0.0:
        return 1.0
    return sanitize_probability(count / LEARNED_RERANKER_FULL_AUTHORITY_PAIR_COUNT, 0.0)


def candidate_feature_payload_from_item(item: Dict[str, Any]) -> Dict[str, float]:
    score_components = item.get("score_components") if isinstance(item.get("score_components"), dict) else {}
    retrieval_branch_scores = (
        item.get("retrieval_branch_scores") if isinstance(item.get("retrieval_branch_scores"), dict) else {}
    )
    similarity = item.get("similarity") if isinstance(item.get("similarity"), dict) else {}
    comment_trace = item.get("comment_trace") if isinstance(item.get("comment_trace"), dict) else {}
    trajectory_trace = item.get("trajectory_trace") if isinstance(item.get("trajectory_trace"), dict) else {}
    reasons = {str(reason) for reason in list(item.get("ranking_reasons") or [])}
    support_level = str(item.get("support_level") or "")

    return {
        "baseline_score": sanitize_probability(item.get("score"), 0.0),
        "score_component_semantic_relevance": sanitize_probability(
            score_components.get("semantic_relevance"), 0.0
        ),
        "score_component_intent_alignment": sanitize_probability(
            score_components.get("intent_alignment"), 0.0
        ),
        "score_component_performance_quality": sanitize_probability(
            score_components.get("performance_quality"), 0.0
        ),
        "score_component_reference_usefulness": sanitize_probability(
            score_components.get("reference_usefulness"), 0.0
        ),
        "score_component_support_confidence": sanitize_probability(
            score_components.get("support_confidence"), 0.0
        ),
        "retrieval_semantic": sanitize_probability(
            retrieval_branch_scores.get("semantic"), 0.0
        ),
        "retrieval_hashtag_topic": sanitize_probability(
            retrieval_branch_scores.get("hashtag_topic"), 0.0
        ),
        "retrieval_structured_compatibility": sanitize_probability(
            retrieval_branch_scores.get("structured_compatibility"), 0.0
        ),
        "retrieval_fused": sanitize_probability(
            retrieval_branch_scores.get("fused_retrieval")
            if "fused_retrieval" in retrieval_branch_scores
            else retrieval_branch_scores.get("fused"),
            0.0,
        ),
        "similarity_sparse": sanitize_probability(similarity.get("sparse"), 0.0),
        "similarity_dense": sanitize_probability(similarity.get("dense"), 0.0),
        "similarity_fused": sanitize_probability(similarity.get("fused"), 0.0),
        "support_score": sanitize_probability(item.get("support_score"), 0.0),
        "support_is_full": _bool_flag(support_level == "full"),
        "support_is_partial": _bool_flag(support_level == "partial"),
        "confidence": sanitize_probability(item.get("confidence"), 0.0),
        "comment_alignment_score": sanitize_probability(
            comment_trace.get("alignment_score"), 0.0
        ),
        "comment_value_prop_coverage": sanitize_probability(
            comment_trace.get("value_prop_coverage"), 0.0
        ),
        "comment_on_topic_ratio": sanitize_probability(
            comment_trace.get("on_topic_ratio"), 0.0
        ),
        "comment_artifact_drift_ratio": sanitize_probability(
            comment_trace.get("artifact_drift_ratio"), 0.0
        ),
        "comment_alignment_confidence": sanitize_probability(
            comment_trace.get("alignment_confidence", comment_trace.get("confidence")),
            0.0,
        ),
        "trajectory_similarity": sanitize_probability(
            item.get("trajectory_similarity", trajectory_trace.get("similarity")), 0.0
        ),
        "trajectory_regime_confidence": sanitize_probability(
            item.get("trajectory_regime_confidence", trajectory_trace.get("regime_confidence")),
            0.0,
        ),
        "has_reason_semantic": _bool_flag("strong_semantic_relevance" in reasons),
        "has_reason_intent": _bool_flag("strong_intent_alignment" in reasons),
        "has_reason_performance": _bool_flag("strong_performance_quality" in reasons),
        "has_reason_reference": _bool_flag("strong_reference_usefulness" in reasons),
        "has_reason_support": _bool_flag("strong_support_confidence" in reasons),
        "has_reason_multi_branch": _bool_flag("multi_branch_retrieval_match" in reasons),
        "has_reason_full_support": _bool_flag("fully_supported_reference" in reasons),
        "retrieval_branch_count": float(
            len(list(item.get("retrieval_branches") or []))
            if isinstance(item.get("retrieval_branches"), list)
            else sum(
                1
                for branch in (
                    "semantic",
                    "hashtag_topic",
                    "structured_compatibility",
                )
                if sanitize_probability(retrieval_branch_scores.get(branch), 0.0) > 0.0
            )
        ),
    }


def feature_vector_from_payload(payload: Dict[str, float]) -> np.ndarray:
    return np.array(
        [float(payload.get(name, 0.0)) for name in LEARNED_RERANKER_FEATURE_NAMES],
        dtype=np.float32,
    )


def delta_feature_vector(
    left_payload: Dict[str, float],
    right_payload: Dict[str, float],
) -> np.ndarray:
    left = feature_vector_from_payload(left_payload)
    right = feature_vector_from_payload(right_payload)
    return left - right


@dataclass
class PairwiseTrainingRow:
    request_id: str
    objective_effective: str
    candidate_a_id: str
    candidate_b_id: str
    label: int
    pair_source: str
    pair_weight: float
    features_a: Dict[str, float]
    features_b: Dict[str, float]

    def delta_vector(self) -> np.ndarray:
        return delta_feature_vector(self.features_a, self.features_b)


class LearnedPairwiseReranker:
    def __init__(
        self,
        *,
        objective: str,
        model: Any,
        model_type: str,
        train_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.objective = str(objective)
        self.model = model
        self.model_type = str(model_type)
        self.train_summary = dict(train_summary or {})

    @classmethod
    def train(
        cls,
        *,
        objective: str,
        rows: Sequence[PairwiseTrainingRow],
        random_state: int = 7,
    ) -> "LearnedPairwiseReranker":
        if not rows:
            raise ValueError("Cannot train learned reranker without pairwise rows.")

        x = np.stack([row.delta_vector() for row in rows], axis=0)
        y = np.array([int(row.label) for row in rows], dtype=np.int32)
        sample_weight = np.array(
            [max(1e-6, float(row.pair_weight)) for row in rows], dtype=np.float32
        )

        use_lightgbm = lgb is not None and len(rows) >= 24
        if use_lightgbm:
            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=180,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=4,
                min_data_in_bin=1,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=1,
                verbosity=-1,
                random_state=int(random_state),
            )
            model.fit(x, y, sample_weight=sample_weight)
            model_type = "lightgbm_classifier"
            train_scores = model.predict_proba(x)[:, 1]
        else:
            model = LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                random_state=int(random_state),
            )
            model.fit(x, y, sample_weight=sample_weight)
            model_type = "logistic_regression_classifier"
            train_scores = model.predict_proba(x)[:, 1]

        weighted_correct = float(
            np.sum(((train_scores >= 0.5) == (y >= 1)).astype(np.float32) * sample_weight)
        )
        weighted_total = float(np.sum(sample_weight))
        train_accuracy = 0.0 if weighted_total <= 0.0 else weighted_correct / weighted_total
        train_summary = {
            "pair_count": int(len(rows)),
            "positive_pair_count": int(np.sum(y == 1)),
            "negative_pair_count": int(np.sum(y == 0)),
            "weighted_train_accuracy": round(float(train_accuracy), 6),
            "label_policy_version": LEARNED_RERANKER_LABEL_POLICY_VERSION,
            "feature_schema_hash": learned_reranker_feature_schema_hash(),
        }
        return cls(
            objective=objective,
            model=model,
            model_type=model_type,
            train_summary=train_summary,
        )

    def _pairwise_probability(
        self,
        *,
        left_payload: Dict[str, float],
        right_payload: Dict[str, float],
    ) -> float:
        vec = delta_feature_vector(left_payload, right_payload).reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(vec)[0][1])
        else:  # pragma: no cover - defensive
            raw = float(self.model.predict(vec)[0])
            proba = 1.0 / (1.0 + math.exp(-raw))
        return sanitize_probability(proba, 0.5)

    def _pairwise_probabilities_batch(
        self,
        *,
        pair_payloads: Sequence[Tuple[Dict[str, float], Dict[str, float]]],
    ) -> np.ndarray:
        if not pair_payloads:
            return np.zeros((0,), dtype=np.float32)
        matrix = np.stack(
            [
                delta_feature_vector(left_payload, right_payload)
                for left_payload, right_payload in pair_payloads
            ],
            axis=0,
        )
        if hasattr(self.model, "predict_proba"):
            out = self.model.predict_proba(matrix)[:, 1]
        else:  # pragma: no cover - defensive
            raw = self.model.predict(matrix)
            out = 1.0 / (1.0 + np.exp(-raw))
        return np.array([sanitize_probability(item, 0.5) for item in out], dtype=np.float32)

    def rerank_items(
        self,
        *,
        items: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not items:
            return [], {
                "enabled": True,
                "applied": False,
                "reason": "empty_shortlist",
                "ranker_id": LEARNED_RERANKER_ID,
                "version": LEARNED_RERANKER_VERSION,
            }
        if len(items) == 1:
            single = dict(items[0])
            baseline_score = sanitize_probability(single.get("score"), 0.0)
            single["baseline_score"] = round_score(baseline_score, 6)
            single["learned_score"] = round_score(baseline_score, 6)
            single["score_raw"] = round_score(baseline_score, 6)
            single["score_calibrated"] = round_score(baseline_score, 6)
            single["score"] = round_score(baseline_score, 6)
            single["score_mean"] = round_score(baseline_score, 6)
            single["score_std"] = 0.0
            single["confidence"] = sanitize_probability(single.get("confidence"), 0.0)
            single["selected_ranker_id"] = LEARNED_RERANKER_ID
            single["global_score_mean"] = round_score(baseline_score, 6)
            single["segment_blend_weight"] = 1.0
            single["calibration_trace"] = {
                "score_raw": round_score(baseline_score, 6),
                "score_calibrated": round_score(baseline_score, 6),
                "calibrator_segment_id": "learned_identity",
                "requested_segment_id": LEARNED_RERANKER_ID,
                "calibrator_method": "identity",
                "calibrator_support_count": 0,
                "calibration_fallback_used": False,
                "target_definition": "pairwise_win_rate",
            }
            return [single], {
                "enabled": True,
                "applied": True,
                "reason": None,
                "ranker_id": LEARNED_RERANKER_ID,
                "version": LEARNED_RERANKER_VERSION,
                "shortlist_size": 1,
            }

        payloads = [candidate_feature_payload_from_item(item) for item in items]
        pairwise_matrix = np.zeros((len(items), len(items)), dtype=np.float32)
        pair_specs: List[Tuple[int, int]] = []
        pair_payloads: List[Tuple[Dict[str, float], Dict[str, float]]] = []
        for left_idx in range(len(items)):
            for right_idx in range(left_idx + 1, len(items)):
                pair_specs.append((left_idx, right_idx))
                pair_payloads.append((payloads[left_idx], payloads[right_idx]))
        probabilities = self._pairwise_probabilities_batch(pair_payloads=pair_payloads)
        for (left_idx, right_idx), prob in zip(pair_specs, probabilities):
                pairwise_matrix[left_idx, right_idx] = float(prob)
                pairwise_matrix[right_idx, left_idx] = float(1.0 - prob)

        shortlist_signal_strength = 0.0
        if len(probabilities) > 0:
            shortlist_signal_strength = float(
                np.mean(np.abs(probabilities - 0.5)) * 2.0
            )
        train_pair_authority = _authority_from_pair_count(
            self.train_summary.get("pair_count", 0)
        )
        provisional_win_rates = [
            float(np.sum(pairwise_matrix[idx])) / float(max(1, len(items) - 1))
            for idx in range(len(items))
        ]
        learned_top_idx = max(
            range(len(provisional_win_rates)),
            key=lambda idx: provisional_win_rates[idx],
        )
        residual_budget = (
            LEARNED_RERANKER_MAX_RESIDUAL_SHIFT
            * shortlist_signal_strength
            * train_pair_authority
        )

        if shortlist_signal_strength < LEARNED_RERANKER_MIN_SHORTLIST_SIGNAL:
            baseline_items: List[Dict[str, Any]] = []
            for item in items:
                baseline_score = sanitize_probability(item.get("score"), 0.0)
                updated = dict(item)
                updated["baseline_score"] = round_score(baseline_score, 6)
                updated["learned_score"] = round_score(baseline_score, 6)
                updated["learned_trace"] = {
                    "version": LEARNED_RERANKER_VERSION,
                    "model_type": self.model_type,
                    "applied": False,
                    "reason": "shortlist_signal_below_threshold",
                    "baseline_score": round_score(baseline_score, 6),
                    "shortlist_signal_strength": round_score(shortlist_signal_strength, 6),
                    "train_pair_authority": round_score(train_pair_authority, 6),
                }
                baseline_items.append(updated)
            return baseline_items, {
                "enabled": True,
                "applied": False,
                "reason": "shortlist_signal_below_threshold",
                "ranker_id": LEARNED_RERANKER_ID,
                "version": LEARNED_RERANKER_VERSION,
                "shortlist_size": len(items),
                "model_type": self.model_type,
                "shortlist_signal_strength": round_score(shortlist_signal_strength, 6),
                "train_pair_authority": round_score(train_pair_authority, 6),
                "residual_budget": round_score(residual_budget, 6),
            }

        if (
            train_pair_authority < LEARNED_RERANKER_HEAD_GUARD_MIN_AUTHORITY
            and learned_top_idx >= LEARNED_RERANKER_BASELINE_HEAD_GUARD_RANK
        ):
            baseline_items: List[Dict[str, Any]] = []
            for item in items:
                baseline_score = sanitize_probability(item.get("score"), 0.0)
                updated = dict(item)
                updated["baseline_score"] = round_score(baseline_score, 6)
                updated["learned_score"] = round_score(baseline_score, 6)
                updated["learned_trace"] = {
                    "version": LEARNED_RERANKER_VERSION,
                    "model_type": self.model_type,
                    "applied": False,
                    "reason": "baseline_head_guard",
                    "baseline_score": round_score(baseline_score, 6),
                    "shortlist_signal_strength": round_score(shortlist_signal_strength, 6),
                    "train_pair_authority": round_score(train_pair_authority, 6),
                    "learned_top_baseline_rank": int(learned_top_idx + 1),
                }
                baseline_items.append(updated)
            return baseline_items, {
                "enabled": True,
                "applied": False,
                "reason": "baseline_head_guard",
                "ranker_id": LEARNED_RERANKER_ID,
                "version": LEARNED_RERANKER_VERSION,
                "shortlist_size": len(items),
                "model_type": self.model_type,
                "blend_mode": "baseline_plus_learned_residual",
                "shortlist_signal_strength": round_score(shortlist_signal_strength, 6),
                "train_pair_authority": round_score(train_pair_authority, 6),
                "residual_budget": round_score(residual_budget, 6),
                "learned_top_baseline_rank": int(learned_top_idx + 1),
            }

        reranked: List[Dict[str, Any]] = []
        for idx, item in enumerate(items):
            baseline_score = sanitize_probability(item.get("score"), 0.0)
            win_rate = provisional_win_rates[idx]
            learned_margin = (win_rate - 0.5) * 2.0
            score_shift = residual_budget * learned_margin * abs(learned_margin)
            final_score = sanitize_probability(baseline_score + score_shift, baseline_score)
            updated = dict(item)
            updated["baseline_score"] = round_score(baseline_score, 6)
            updated["learned_score"] = round_score(win_rate, 6)
            updated["score_raw"] = round_score(final_score, 6)
            updated["score_calibrated"] = round_score(final_score, 6)
            updated["score"] = round_score(final_score, 6)
            updated["score_mean"] = round_score(final_score, 6)
            updated["score_std"] = 0.0
            updated["confidence"] = round_score(
                max(
                    sanitize_probability(item.get("confidence"), 0.0),
                    abs(score_shift) / max(1e-6, LEARNED_RERANKER_MAX_RESIDUAL_SHIFT),
                ),
                6,
            )
            updated["selected_ranker_id"] = LEARNED_RERANKER_ID
            updated["global_score_mean"] = round_score(final_score, 6)
            updated["segment_blend_weight"] = 1.0
            updated["calibration_trace"] = {
                "score_raw": round_score(final_score, 6),
                "score_calibrated": round_score(final_score, 6),
                "calibrator_segment_id": "learned_identity",
                "requested_segment_id": LEARNED_RERANKER_ID,
                "calibrator_method": "identity",
                "calibrator_support_count": 0,
                "calibration_fallback_used": False,
                "target_definition": "baseline_plus_learned_residual",
            }
            updated["learned_trace"] = {
                "version": LEARNED_RERANKER_VERSION,
                "model_type": self.model_type,
                "win_rate": round_score(win_rate, 6),
                "baseline_score": round_score(baseline_score, 6),
                "final_score": round_score(final_score, 6),
                "learned_margin": round_score(learned_margin, 6),
                "score_shift": round_score(score_shift, 6),
                "shortlist_signal_strength": round_score(shortlist_signal_strength, 6),
                "train_pair_authority": round_score(train_pair_authority, 6),
                "residual_budget": round_score(residual_budget, 6),
                "applied": True,
                "learned_top_baseline_rank": int(learned_top_idx + 1),
            }
            reranked.append(updated)

        reranked.sort(
            key=lambda item: (
                -float(item.get("score", 0.0) or 0.0),
                -float(item.get("baseline_score", item.get("score", 0.0)) or 0.0),
                -float(
                    (
                        (item.get("retrieval_branch_scores") or {}).get("fused_retrieval")
                        if isinstance(item.get("retrieval_branch_scores"), dict)
                        else 0.0
                    )
                    or 0.0
                ),
            )
        )
        return reranked, {
            "enabled": True,
            "applied": True,
            "reason": None,
            "ranker_id": LEARNED_RERANKER_ID,
            "version": LEARNED_RERANKER_VERSION,
            "shortlist_size": len(items),
            "model_type": self.model_type,
            "blend_mode": "baseline_plus_learned_residual",
            "shortlist_signal_strength": round_score(shortlist_signal_strength, 6),
            "train_pair_authority": round_score(train_pair_authority, 6),
            "residual_budget": round_score(residual_budget, 6),
            "learned_top_baseline_rank": int(learned_top_idx + 1),
        }

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "model.pkl").open("wb") as handle:
            pickle.dump(self.model, handle)
        manifest = {
            "ranker_id": LEARNED_RERANKER_ID,
            "version": LEARNED_RERANKER_VERSION,
            "objective": self.objective,
            "model_type": self.model_type,
            "feature_names": LEARNED_RERANKER_FEATURE_NAMES,
            "feature_schema_hash": learned_reranker_feature_schema_hash(),
            "label_policy_version": LEARNED_RERANKER_LABEL_POLICY_VERSION,
            "train_summary": self.train_summary,
            "inference_policy": {
                "blend_mode": "baseline_plus_learned_residual",
                "min_shortlist_signal": LEARNED_RERANKER_MIN_SHORTLIST_SIGNAL,
                "full_authority_pair_count": LEARNED_RERANKER_FULL_AUTHORITY_PAIR_COUNT,
                "max_residual_shift": LEARNED_RERANKER_MAX_RESIDUAL_SHIFT,
                "baseline_head_guard_rank": LEARNED_RERANKER_BASELINE_HEAD_GUARD_RANK,
                "head_guard_min_authority": LEARNED_RERANKER_HEAD_GUARD_MIN_AUTHORITY,
            },
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, output_dir: Path) -> "LearnedPairwiseReranker":
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        expected_hash = learned_reranker_feature_schema_hash()
        actual_hash = str(manifest.get("feature_schema_hash") or "")
        if actual_hash and actual_hash != expected_hash:
            raise ValueError(
                f"learned_reranker_incompatible: expected feature schema {expected_hash}, got {actual_hash}"
            )
        with (output_dir / "model.pkl").open("rb") as handle:
            model = pickle.load(handle)
        return cls(
            objective=str(manifest.get("objective") or ""),
            model=model,
            model_type=str(manifest.get("model_type") or "unknown"),
            train_summary=dict(manifest.get("train_summary") or {}),
        )


def summarize_pairwise_rows(rows: Sequence[PairwiseTrainingRow]) -> Dict[str, Any]:
    by_source: Dict[str, int] = {}
    by_objective: Dict[str, int] = {}
    for row in rows:
        by_source[row.pair_source] = by_source.get(row.pair_source, 0) + 1
        by_objective[row.objective_effective] = (
            by_objective.get(row.objective_effective, 0) + 1
        )
    return {
        "pair_count": int(len(rows)),
        "by_source": by_source,
        "by_objective": by_objective,
    }


def rows_to_json_ready(rows: Iterable[PairwiseTrainingRow]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for row in rows:
        payload.append(
            {
                "request_id": row.request_id,
                "objective_effective": row.objective_effective,
                "candidate_a_id": row.candidate_a_id,
                "candidate_b_id": row.candidate_b_id,
                "label": int(row.label),
                "pair_source": row.pair_source,
                "pair_weight": round_score(row.pair_weight, 6),
                "features_a": row.features_a,
                "features_b": row.features_b,
                "delta_features": {
                    name: round_score(
                        row.features_a.get(name, 0.0) - row.features_b.get(name, 0.0),
                        6,
                    )
                    for name in LEARNED_RERANKER_FEATURE_NAMES
                },
            }
        )
    return payload
