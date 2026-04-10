from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


RUNTIME_OBJECTIVES = ("reach", "engagement", "conversion")
OBJECTIVE_ALIAS_MAP: Dict[str, str] = {
    "community": "engagement",
}


@dataclass(frozen=True)
class ObjectiveSpec:
    objective_id: str
    label_key: str
    primary_metric: str
    training_loss: str
    calibration: str
    minimum_training_rows: int = 50


OBJECTIVE_SPECS: Dict[str, ObjectiveSpec] = {
    "reach": ObjectiveSpec(
        objective_id="reach",
        label_key="reach",
        primary_metric="ndcg@20",
        training_loss="lambdarank",
        calibration="isotonic",
        minimum_training_rows=50,
    ),
    "engagement": ObjectiveSpec(
        objective_id="engagement",
        label_key="engagement",
        primary_metric="ndcg@20",
        training_loss="lambdarank",
        calibration="isotonic",
        minimum_training_rows=50,
    ),
    "conversion": ObjectiveSpec(
        objective_id="conversion",
        label_key="conversion",
        primary_metric="ndcg@20",
        training_loss="lambdarank",
        calibration="isotonic",
        minimum_training_rows=50,
    ),
}


def map_objective(requested_objective: str) -> Tuple[str, str]:
    requested = (requested_objective or "").strip().lower()
    if not requested:
        raise ValueError("objective is required.")
    effective = OBJECTIVE_ALIAS_MAP.get(requested, requested)
    if effective not in OBJECTIVE_SPECS:
        raise ValueError(
            f"Unknown objective '{requested_objective}'. Allowed: {', '.join(RUNTIME_OBJECTIVES)} plus alias 'community'."
        )
    return requested, effective


def objective_spec(objective_id: str) -> ObjectiveSpec:
    _, effective = map_objective(objective_id)
    return OBJECTIVE_SPECS[effective]

