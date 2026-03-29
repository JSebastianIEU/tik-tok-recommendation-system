from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .temporal import parse_dt, row_text


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


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _relevance_label(score: float) -> int:
    if score >= 1.0:
        return 3
    if score >= 0.3:
        return 2
    if score >= -0.3:
        return 1
    return 0


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    idx = _clip(pct, 0.0, 1.0) * (len(ordered) - 1)
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    if lower == upper:
        return ordered[lower]
    frac = idx - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def _minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [0.0 for _ in values]
    denom = hi - lo
    return [float((value - lo) / denom) for value in values]


def _stable_noise(seed: str) -> float:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 1000000) / 1000000.0


def _era_bucket(value: Any, bucket: str) -> str:
    parsed = parse_dt(value)
    if parsed is None:
        return "unknown"
    if bucket == "month":
        return parsed.strftime("%Y-%m")
    return "unknown"


def _objective_payload(
    candidate_row: Dict[str, Any],
    objective: str,
    target_source: str,
) -> Optional[Tuple[float, Optional[Dict[str, float]], Optional[Dict[str, bool]]]]:
    if target_source == "trajectory_v2_composite":
        objective_payload = (
            candidate_row.get("targets_trajectory_z", {})
            if isinstance(candidate_row.get("targets_trajectory_z"), dict)
            else {}
        ).get(objective, {})
        if not isinstance(objective_payload, dict):
            return None
        score = objective_payload.get("composite_z")
        if score is None:
            return None
        availability_payload = (
            candidate_row.get("target_availability", {})
            if isinstance(candidate_row.get("target_availability"), dict)
            else {}
        ).get(objective, {})
        if not isinstance(availability_payload, dict):
            return None
        if not bool(availability_payload.get("objective_available", False)):
            return None
        components_z = (
            objective_payload.get("components_z", {})
            if isinstance(objective_payload.get("components_z"), dict)
            else {}
        )
        components = {
            key: float(value)
            for key, value in components_z.items()
            if key in {"early_velocity", "stability", "late_lift"} and value is not None
        }
        component_mask = (
            availability_payload.get("components", {})
            if isinstance(availability_payload.get("components"), dict)
            else {}
        )
        availability_mask = {
            "candidate_objective_available": bool(availability_payload.get("objective_available", False)),
            "candidate_early_available": bool(component_mask.get("early_velocity", False)),
            "candidate_stability_available": bool(component_mask.get("stability", False)),
            "candidate_late_available": bool(component_mask.get("late_lift", False)),
        }
        return float(score), components, availability_mask
    scalar_targets = (
        candidate_row.get("targets_z", {})
        if isinstance(candidate_row.get("targets_z"), dict)
        else {}
    )
    return float(scalar_targets.get(objective, 0.0)), None, None


@dataclass
class AdaptiveNegativeMiningConfig:
    enabled: bool = False
    mode: str = "adaptive_v2"
    mining_candidate_k: int = 400
    negatives_per_positive: int = 4
    false_friend_similarity_pct: float = 0.80
    false_friend_prediction_pct: float = 0.75
    false_friend_label_max: int = 1
    hard_ratio_bounds: Tuple[float, float] = (0.35, 0.70)
    semi_ratio_bounds: Tuple[float, float] = (0.20, 0.45)
    easy_min_ratio: float = 0.10
    max_per_author: int = 2
    max_per_topic: int = 4
    max_per_era: int = 5
    era_bucket: str = "month"
    seed: int = 13

    def __post_init__(self) -> None:
        if self.mode != "adaptive_v2":
            raise ValueError("AdaptiveNegativeMiningConfig.mode must be 'adaptive_v2'.")
        if self.mining_candidate_k < 1:
            raise ValueError("mining_candidate_k must be >= 1.")
        if self.negatives_per_positive < 1:
            raise ValueError("negatives_per_positive must be >= 1.")
        if not (0.0 <= self.false_friend_similarity_pct <= 1.0):
            raise ValueError("false_friend_similarity_pct must be in [0, 1].")
        if not (0.0 <= self.false_friend_prediction_pct <= 1.0):
            raise ValueError("false_friend_prediction_pct must be in [0, 1].")
        if self.false_friend_label_max < 0 or self.false_friend_label_max > 3:
            raise ValueError("false_friend_label_max must be in [0, 3].")
        hard_lo, hard_hi = self.hard_ratio_bounds
        semi_lo, semi_hi = self.semi_ratio_bounds
        if hard_lo < 0 or hard_hi > 1 or hard_lo > hard_hi:
            raise ValueError("hard_ratio_bounds must satisfy 0 <= min <= max <= 1.")
        if semi_lo < 0 or semi_hi > 1 or semi_lo > semi_hi:
            raise ValueError("semi_ratio_bounds must satisfy 0 <= min <= max <= 1.")
        if self.easy_min_ratio < 0 or self.easy_min_ratio > 1:
            raise ValueError("easy_min_ratio must be in [0, 1].")
        if self.max_per_author < 1 or self.max_per_topic < 1 or self.max_per_era < 1:
            raise ValueError("max_per_author/max_per_topic/max_per_era must be >= 1.")
        if self.era_bucket != "month":
            raise ValueError("era_bucket must be 'month' for this version.")


class AdaptiveNegativeMiner:
    def __init__(self, config: Optional[AdaptiveNegativeMiningConfig] = None) -> None:
        self.config = config or AdaptiveNegativeMiningConfig()

    def _select_with_caps(
        self,
        *,
        query_row: Dict[str, Any],
        pool: Sequence[Dict[str, Any]],
        count: int,
        reason_fallback: str,
        selected_keys: set[str],
        author_counts: Dict[str, int],
        topic_counts: Dict[str, int],
        era_counts: Dict[str, int],
        drop_counters: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for entry in pool:
            if len(out) >= count:
                break
            candidate_row = entry["candidate_row"]
            candidate_key = entry["candidate_key"]
            if candidate_key in selected_keys:
                drop_counters["duplicate_candidate"] = drop_counters.get("duplicate_candidate", 0) + 1
                continue
            author_key = str(candidate_row.get("author_id") or "unknown")
            topic_key = str(candidate_row.get("topic_key") or "general")
            era_key = str(entry["debias_era_bucket"])
            if author_counts.get(author_key, 0) >= self.config.max_per_author:
                drop_counters["author_cap"] = drop_counters.get("author_cap", 0) + 1
                continue
            if topic_counts.get(topic_key, 0) >= self.config.max_per_topic:
                drop_counters["topic_cap"] = drop_counters.get("topic_cap", 0) + 1
                continue
            if era_counts.get(era_key, 0) >= self.config.max_per_era:
                drop_counters["era_cap"] = drop_counters.get("era_cap", 0) + 1
                continue
            selected_keys.add(candidate_key)
            author_counts[author_key] = author_counts.get(author_key, 0) + 1
            topic_counts[topic_key] = topic_counts.get(topic_key, 0) + 1
            era_counts[era_key] = era_counts.get(era_key, 0) + 1
            entry["mining_reason"] = (
                "false_friend" if bool(entry.get("is_false_friend")) else reason_fallback
            )
            out.append(entry)
        return out

    def mine(
        self,
        *,
        objective: str,
        target_source: str,
        rows_by_id: Dict[str, Dict[str, Any]],
        train_rows: Sequence[Dict[str, Any]],
        base_pair_rows: Sequence[Dict[str, Any]],
        retriever: Any,
        baseline_ranker: Any,
        max_age_days: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.config.enabled:
            return [], {
                "enabled": False,
                "mode": self.config.mode,
                "objective": objective,
            }

        from .ranker import pair_feature_vector_array

        candidate_rows = [row for row in train_rows if str(row.get("split")) == "train"]
        candidate_by_row_id = {str(row.get("row_id")): row for row in candidate_rows}
        label_by_pair: Dict[Tuple[str, str], int] = {}
        score_by_pair: Dict[Tuple[str, str], float] = {}
        positives_by_query: Dict[str, int] = {}

        for pair in base_pair_rows:
            if str(pair.get("objective")) != objective:
                continue
            query_id = str(pair.get("query_row_id"))
            candidate_id = str(pair.get("candidate_row_id"))
            rel = int(float(pair.get("relevance_label") or 0.0))
            score = float(pair.get("objective_score") or 0.0)
            label_by_pair[(query_id, candidate_id)] = rel
            score_by_pair[(query_id, candidate_id)] = score
            query_row = rows_by_id.get(query_id)
            candidate_row = rows_by_id.get(candidate_id)
            if query_row is None or candidate_row is None:
                continue
            if str(query_row.get("split")) != "train" or str(candidate_row.get("split")) != "train":
                continue
            if rel >= 1:
                positives_by_query[query_id] = positives_by_query.get(query_id, 0) + 1

        per_query_pool: Dict[str, List[Dict[str, Any]]] = {}
        eligible_negative_count = 0
        false_friend_count = 0

        for query_row in train_rows:
            query_id = str(query_row.get("row_id"))
            if positives_by_query.get(query_id, 0) <= 0:
                continue
            retrieved = retriever.retrieve(
                query_row=query_row,
                candidate_rows=candidate_rows,
                top_k=max(1, int(self.config.mining_candidate_k)),
                index_cutoff_time=query_row.get("as_of_time"),
                objective=objective,
                retrieval_constraints={"max_age_days": max(1, int(max_age_days))},
            )
            raw_pool: List[Dict[str, Any]] = []
            pred_values: List[float] = []
            sim_values: List[float] = []

            for item in retrieved:
                candidate_row_id = str(item.get("candidate_row_id"))
                if candidate_row_id == query_id:
                    continue
                candidate_row = candidate_by_row_id.get(candidate_row_id)
                if candidate_row is None:
                    continue
                query_dt = parse_dt(query_row.get("as_of_time"))
                candidate_dt = parse_dt(candidate_row.get("as_of_time"))
                if query_dt is not None and candidate_dt is not None and candidate_dt >= query_dt:
                    continue

                pair_key = (query_id, candidate_row_id)
                payload = _objective_payload(
                    candidate_row=candidate_row,
                    objective=objective,
                    target_source=target_source,
                )
                if payload is None:
                    continue
                objective_score, objective_components, availability_mask = payload
                observed_label = label_by_pair.get(pair_key)
                if observed_label is None:
                    observed_label = _relevance_label(objective_score)
                if observed_label > self.config.false_friend_label_max:
                    continue
                retrieval_similarity = _clip(float(item.get("fused_score") or 0.0), 0.0, 1.0)
                pair_vec = pair_feature_vector_array(
                    query_row=query_row,
                    candidate_row=candidate_row,
                    similarity=retrieval_similarity,
                )
                baseline_pred = float(baseline_ranker.predict_scores(pair_vec)[0])
                raw_pool.append(
                    {
                        "query_row": query_row,
                        "candidate_row": candidate_row,
                        "candidate_row_id": candidate_row_id,
                        "candidate_key": str(
                            candidate_row.get("video_id") or candidate_row_id.split("::", 1)[0]
                        ),
                        "retrieval_similarity": retrieval_similarity,
                        "baseline_pred": baseline_pred,
                        "observed_label": int(observed_label),
                        "objective_score": float(score_by_pair.get(pair_key, objective_score)),
                        "objective_score_components": objective_components,
                        "availability_mask": availability_mask,
                    }
                )
                pred_values.append(baseline_pred)
                sim_values.append(retrieval_similarity)

            if not raw_pool:
                continue

            pred_norm_values = _minmax(pred_values)
            sim_threshold = _percentile(sim_values, self.config.false_friend_similarity_pct)
            pred_threshold = _percentile(pred_norm_values, self.config.false_friend_prediction_pct)
            for idx, entry in enumerate(raw_pool):
                pred_norm = pred_norm_values[idx]
                label = int(entry["observed_label"])
                is_false_friend = (
                    float(entry["retrieval_similarity"]) >= sim_threshold
                    and pred_norm >= pred_threshold
                    and label <= self.config.false_friend_label_max
                )
                if is_false_friend:
                    false_friend_count += 1
                eligible_negative_count += 1
                hardness_score = _clip(
                    (0.50 * float(entry["retrieval_similarity"]))
                    + (0.35 * float(pred_norm))
                    + (0.15 * (1.0 - (float(label) / 3.0))),
                    0.0,
                    1.0,
                )
                entry["prediction_norm"] = pred_norm
                entry["is_false_friend"] = is_false_friend
                entry["hardness_score"] = hardness_score
                entry["debias_era_bucket"] = _era_bucket(
                    entry["candidate_row"].get("as_of_time"),
                    self.config.era_bucket,
                )
            per_query_pool[query_id] = raw_pool

        confusion_rate = (
            float(false_friend_count) / float(eligible_negative_count)
            if eligible_negative_count > 0
            else 0.0
        )
        hard_ratio = _clip(
            0.35 + (0.40 * confusion_rate),
            self.config.hard_ratio_bounds[0],
            self.config.hard_ratio_bounds[1],
        )
        semi_ratio = _clip(
            0.45 - (0.25 * confusion_rate),
            self.config.semi_ratio_bounds[0],
            self.config.semi_ratio_bounds[1],
        )
        easy_ratio = max(float(self.config.easy_min_ratio), 1.0 - hard_ratio - semi_ratio)
        ratio_total = hard_ratio + semi_ratio + easy_ratio
        if ratio_total > 0:
            hard_ratio /= ratio_total
            semi_ratio /= ratio_total
            easy_ratio /= ratio_total

        mined_rows: List[Dict[str, Any]] = []
        dropped_by_cap: Dict[str, int] = {}
        selected_counts = {"false_friend": 0, "hard": 0, "semi_hard": 0, "easy": 0}
        selected_author: Dict[str, int] = {}
        selected_topic: Dict[str, int] = {}
        selected_era: Dict[str, int] = {}

        for query_id, pool in per_query_pool.items():
            positive_count = positives_by_query.get(query_id, 0)
            if positive_count <= 0:
                continue
            required_negatives = positive_count * self.config.negatives_per_positive
            if required_negatives <= 0:
                continue

            ordered_by_hardness = sorted(
                pool,
                key=lambda item: (
                    -int(bool(item.get("is_false_friend"))),
                    -float(item.get("hardness_score") or 0.0),
                    _stable_noise(
                        f"{self.config.seed}:{objective}:{query_id}:{item.get('candidate_row_id')}"
                    ),
                ),
            )
            n = len(ordered_by_hardness)
            hard_edge = max(1, n // 3)
            semi_edge = max(hard_edge + 1, (2 * n) // 3)
            hard_pool = list(ordered_by_hardness[:hard_edge])
            hard_ids = {str(item.get("candidate_row_id")) for item in hard_pool}
            for entry in ordered_by_hardness[hard_edge:]:
                candidate_row_id = str(entry.get("candidate_row_id"))
                if bool(entry.get("is_false_friend")) and candidate_row_id not in hard_ids:
                    hard_pool.append(entry)
                    hard_ids.add(candidate_row_id)
            semi_pool = [
                item
                for item in ordered_by_hardness[hard_edge:semi_edge]
                if str(item.get("candidate_row_id")) not in hard_ids
            ]
            easy_pool = [
                item
                for item in ordered_by_hardness[semi_edge:]
                if str(item.get("candidate_row_id")) not in hard_ids
            ]

            hard_count = int(round(required_negatives * hard_ratio))
            semi_count = int(round(required_negatives * semi_ratio))
            easy_count = max(0, required_negatives - hard_count - semi_count)

            selected_keys: set[str] = set()
            author_counts: Dict[str, int] = {}
            topic_counts: Dict[str, int] = {}
            era_counts: Dict[str, int] = {}

            selected_entries: List[Dict[str, Any]] = []
            selected_entries.extend(
                self._select_with_caps(
                    query_row=rows_by_id[query_id],
                    pool=hard_pool,
                    count=hard_count,
                    reason_fallback="hard",
                    selected_keys=selected_keys,
                    author_counts=author_counts,
                    topic_counts=topic_counts,
                    era_counts=era_counts,
                    drop_counters=dropped_by_cap,
                )
            )
            selected_entries.extend(
                self._select_with_caps(
                    query_row=rows_by_id[query_id],
                    pool=semi_pool,
                    count=semi_count,
                    reason_fallback="semi_hard",
                    selected_keys=selected_keys,
                    author_counts=author_counts,
                    topic_counts=topic_counts,
                    era_counts=era_counts,
                    drop_counters=dropped_by_cap,
                )
            )
            selected_entries.extend(
                self._select_with_caps(
                    query_row=rows_by_id[query_id],
                    pool=easy_pool,
                    count=easy_count,
                    reason_fallback="easy",
                    selected_keys=selected_keys,
                    author_counts=author_counts,
                    topic_counts=topic_counts,
                    era_counts=era_counts,
                    drop_counters=dropped_by_cap,
                )
            )

            if len(selected_entries) < required_negatives:
                selected_entries.extend(
                    self._select_with_caps(
                        query_row=rows_by_id[query_id],
                        pool=ordered_by_hardness,
                        count=required_negatives - len(selected_entries),
                        reason_fallback="hard",
                        selected_keys=selected_keys,
                        author_counts=author_counts,
                        topic_counts=topic_counts,
                        era_counts=era_counts,
                        drop_counters=dropped_by_cap,
                    )
                )

            query_row = rows_by_id[query_id]
            for entry in selected_entries:
                candidate_row = entry["candidate_row"]
                candidate_row_id = str(entry["candidate_row_id"])
                reason = str(entry.get("mining_reason") or "hard")
                selected_counts[reason] = selected_counts.get(reason, 0) + 1
                author_key = str(candidate_row.get("author_id") or "unknown")
                topic_key = str(candidate_row.get("topic_key") or "general")
                era_key = str(entry.get("debias_era_bucket") or "unknown")
                selected_author[author_key] = selected_author.get(author_key, 0) + 1
                selected_topic[topic_key] = selected_topic.get(topic_key, 0) + 1
                selected_era[era_key] = selected_era.get(era_key, 0) + 1
                mined_rows.append(
                    {
                        "pair_id": (
                            f"{query_id}::{candidate_row_id}::"
                            f"{self.config.mode}::{reason}"
                        ),
                        "query_row_id": query_id,
                        "query_video_id": str(query_row.get("video_id")),
                        "candidate_row_id": candidate_row_id,
                        "candidate_video_id": str(candidate_row.get("video_id")),
                        "query_as_of_time": query_row.get("as_of_time"),
                        "candidate_as_of_time": candidate_row.get("as_of_time"),
                        "similarity": round(float(entry["retrieval_similarity"]), 6),
                        "objective": objective,
                        "target_source": target_source,
                        "objective_score": round(float(entry["objective_score"]), 6),
                        "objective_score_components": entry.get("objective_score_components"),
                        "availability_mask": entry.get("availability_mask"),
                        "relevance_label": int(entry["observed_label"]),
                        "mined": True,
                        "mining_policy_version": self.config.mode,
                        "mining_reason": reason,
                        "hardness_score": round(float(entry["hardness_score"]), 6),
                        "debias_era_bucket": str(entry.get("debias_era_bucket") or "unknown"),
                    }
                )

        mined_unique = {
            (str(row.get("query_row_id")), str(row.get("candidate_row_id"))): row
            for row in mined_rows
        }
        mined_rows = list(mined_unique.values())

        def _top_share(counts: Dict[str, int]) -> float:
            if not counts:
                return 0.0
            total = sum(counts.values())
            if total <= 0:
                return 0.0
            return max(counts.values()) / total

        diagnostics = {
            "enabled": True,
            "mode": self.config.mode,
            "objective": objective,
            "target_source": target_source,
            "seed": self.config.seed,
            "mining_candidate_k": self.config.mining_candidate_k,
            "negatives_per_positive": self.config.negatives_per_positive,
            "eligible_negative_count": int(eligible_negative_count),
            "false_friend_count": int(false_friend_count),
            "confusion_rate": round(float(confusion_rate), 6),
            "ratios": {
                "hard": round(float(hard_ratio), 6),
                "semi_hard": round(float(semi_ratio), 6),
                "easy": round(float(easy_ratio), 6),
            },
            "query_pool_count": len(per_query_pool),
            "queries_with_positives": len(positives_by_query),
            "mined_rows_total": len(mined_rows),
            "selected_by_reason": selected_counts,
            "dropped_by_cap": dropped_by_cap,
            "concentration": {
                "top_author_share": round(_top_share(selected_author), 6),
                "top_topic_share": round(_top_share(selected_topic), 6),
                "top_era_share": round(_top_share(selected_era), 6),
            },
            "debias_caps": {
                "max_per_author": self.config.max_per_author,
                "max_per_topic": self.config.max_per_topic,
                "max_per_era": self.config.max_per_era,
                "era_bucket": self.config.era_bucket,
            },
        }
        return mined_rows, diagnostics
