from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    lgb = None


FEATURE_NAMES = [
    "similarity",
    "query_caption_word_count",
    "query_hashtag_count",
    "query_keyword_count",
    "candidate_caption_word_count",
    "candidate_hashtag_count",
    "candidate_keyword_count",
    "delta_caption_word_count",
    "delta_hashtag_count",
    "delta_keyword_count",
    "same_author",
    "same_topic",
]


def _as_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    if np.isfinite(out):
        return out
    return fallback


def _row_feature(row: Dict[str, Any], key: str) -> float:
    features = row.get("features", {})
    if not isinstance(features, dict):
        return 0.0
    return _as_float(features.get(key), 0.0)


def _pair_feature_vector(
    query_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
    similarity: float,
) -> List[float]:
    q_caption = _row_feature(query_row, "caption_word_count")
    q_hashtag = _row_feature(query_row, "hashtag_count")
    q_keyword = _row_feature(query_row, "keyword_count")

    c_caption = _row_feature(candidate_row, "caption_word_count")
    c_hashtag = _row_feature(candidate_row, "hashtag_count")
    c_keyword = _row_feature(candidate_row, "keyword_count")

    return [
        _as_float(similarity, 0.0),
        q_caption,
        q_hashtag,
        q_keyword,
        c_caption,
        c_hashtag,
        c_keyword,
        abs(q_caption - c_caption),
        abs(q_hashtag - c_hashtag),
        abs(q_keyword - c_keyword),
        1.0 if query_row.get("author_id") == candidate_row.get("author_id") else 0.0,
        1.0 if query_row.get("topic_key") == candidate_row.get("topic_key") else 0.0,
    ]


def build_ranker_training_frame(
    rows_by_id: Dict[str, Dict[str, Any]],
    pair_rows: Sequence[Dict[str, Any]],
    objective: str,
    query_split: str = "train",
    candidate_split: Optional[str] = "train",
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    grouped: Dict[str, List[Tuple[List[float], float]]] = {}
    for pair in pair_rows:
        if str(pair.get("objective")) != objective:
            continue
        query_id = str(pair.get("query_row_id"))
        candidate_id = str(pair.get("candidate_row_id"))
        query_row = rows_by_id.get(query_id)
        candidate_row = rows_by_id.get(candidate_id)
        if query_row is None or candidate_row is None:
            continue
        if str(query_row.get("split")) != query_split:
            continue
        if candidate_split and str(candidate_row.get("split")) != candidate_split:
            continue
        feature_vec = _pair_feature_vector(
            query_row=query_row,
            candidate_row=candidate_row,
            similarity=_as_float(pair.get("similarity"), 0.0),
        )
        label = _as_float(pair.get("relevance_label"), 0.0)
        grouped.setdefault(query_id, []).append((feature_vec, label))

    X: List[List[float]] = []
    y: List[float] = []
    groups: List[int] = []
    query_ids: List[str] = []
    for query_id, rows in grouped.items():
        if len(rows) < 2:
            continue
        groups.append(len(rows))
        query_ids.append(query_id)
        for feature_vec, label in rows:
            X.append(feature_vec)
            y.append(label)

    if not X:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros((0,)), [], []
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), groups, query_ids


@dataclass
class ObjectiveRankerConfig:
    objective: str
    n_estimators: int = 200
    random_state: int = 13


class ObjectiveRankerModel:
    def __init__(
        self,
        objective: str,
        backend: str,
        model: Any,
        calibrator: Optional[IsotonicRegression] = None,
    ) -> None:
        self.objective = objective
        self.backend = backend
        self.model = model
        self.calibrator = calibrator

    @classmethod
    def train(
        cls,
        config: ObjectiveRankerConfig,
        rows_by_id: Dict[str, Dict[str, Any]],
        pair_rows: Sequence[Dict[str, Any]],
    ) -> "ObjectiveRankerModel":
        X_train, y_train, groups, _ = build_ranker_training_frame(
            rows_by_id=rows_by_id,
            pair_rows=pair_rows,
            objective=config.objective,
            query_split="train",
            candidate_split="train",
        )
        if X_train.shape[0] == 0:
            raise ValueError(f"No training rows available for objective '{config.objective}'.")

        backend = "sklearn-hgbr"
        if lgb is not None:
            model = lgb.LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=config.n_estimators,
                learning_rate=0.05,
                num_leaves=31,
                random_state=config.random_state,
            )
            model.fit(X_train, y_train, group=groups)
            backend = "lightgbm-lambdarank"
        else:
            model = HistGradientBoostingRegressor(
                max_iter=config.n_estimators,
                learning_rate=0.05,
                random_state=config.random_state,
            )
            model.fit(X_train, y_train)

        calibrator = None
        X_val, y_val, _, _ = build_ranker_training_frame(
            rows_by_id=rows_by_id,
            pair_rows=pair_rows,
            objective=config.objective,
            query_split="validation",
            candidate_split="train",
        )
        if X_val.shape[0] >= 20 and len(set(y_val.tolist())) >= 2:
            val_pred = model.predict(X_val)
            calibrator = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                out_of_bounds="clip",
            )
            calibrator.fit(val_pred, np.clip(y_val / 3.0, 0.0, 1.0))

        return cls(
            objective=config.objective,
            backend=backend,
            model=model,
            calibrator=calibrator,
        )

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.model.predict(X), dtype=np.float32)
        if self.calibrator is None:
            return raw
        calibrated = np.asarray(self.calibrator.predict(raw), dtype=np.float32)
        return calibrated

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "objective": self.objective,
            "backend": self.backend,
            "feature_names": FEATURE_NAMES,
            "has_calibrator": self.calibrator is not None,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with (output_dir / "model.pkl").open("wb") as fh:
            pickle.dump(self.model, fh)
        if self.calibrator is not None:
            with (output_dir / "calibrator.pkl").open("wb") as fh:
                pickle.dump(self.calibrator, fh)

    @classmethod
    def load(cls, output_dir: Path) -> "ObjectiveRankerModel":
        meta = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        with (output_dir / "model.pkl").open("rb") as fh:
            model = pickle.load(fh)
        calibrator = None
        calibrator_path = output_dir / "calibrator.pkl"
        if calibrator_path.exists():
            with calibrator_path.open("rb") as fh:
                calibrator = pickle.load(fh)
        return cls(
            objective=str(meta["objective"]),
            backend=str(meta["backend"]),
            model=model,
            calibrator=calibrator,
        )

