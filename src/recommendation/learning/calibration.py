from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .ranker import SEGMENT_GLOBAL


CALIBRATION_VERSION = "ranker_calibration.v2"


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _to_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    if np.isfinite(out):
        return out
    return fallback


@dataclass
class SegmentCalibrator:
    segment_id: str
    method: str
    support_count: int
    x_thresholds: List[float]
    y_thresholds: List[float]
    score_mean: float
    score_std: float
    label_mean: float

    @classmethod
    def identity(cls, segment_id: str, support_count: int = 0) -> "SegmentCalibrator":
        return cls(
            segment_id=segment_id,
            method="identity",
            support_count=int(support_count),
            x_thresholds=[],
            y_thresholds=[],
            score_mean=0.0,
            score_std=0.0,
            label_mean=0.0,
        )

    @classmethod
    def fit(
        cls,
        *,
        segment_id: str,
        samples: Sequence[Tuple[float, float]],
        min_support: int = 25,
    ) -> "SegmentCalibrator":
        clean: List[Tuple[float, float]] = []
        for score, label in samples:
            s = _to_float(score, 0.0)
            y = _clamp(_to_float(label, 0.0), 0.0, 1.0)
            clean.append((s, y))
        support = len(clean)
        if support < max(2, int(min_support)):
            return cls.identity(segment_id=segment_id, support_count=support)
        x = np.asarray([row[0] for row in clean], dtype=np.float32)
        y = np.asarray([row[1] for row in clean], dtype=np.float32)
        if len(np.unique(x)) < 4 or len(np.unique(y)) < 2:
            return cls.identity(segment_id=segment_id, support_count=support)
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        try:
            model.fit(x, y)
        except Exception:
            return cls.identity(segment_id=segment_id, support_count=support)
        x_thr = list(np.asarray(model.X_thresholds_, dtype=np.float64).tolist())
        y_thr = list(np.asarray(model.y_thresholds_, dtype=np.float64).tolist())
        if not x_thr or not y_thr or len(x_thr) != len(y_thr):
            return cls.identity(segment_id=segment_id, support_count=support)
        return cls(
            segment_id=segment_id,
            method="isotonic",
            support_count=support,
            x_thresholds=[float(v) for v in x_thr],
            y_thresholds=[float(v) for v in y_thr],
            score_mean=float(np.mean(x)),
            score_std=float(np.std(x)),
            label_mean=float(np.mean(y)),
        )

    def apply(self, score: float) -> float:
        s = _to_float(score, 0.0)
        if self.method != "isotonic" or not self.x_thresholds:
            return s
        x = np.asarray(self.x_thresholds, dtype=np.float64)
        y = np.asarray(self.y_thresholds, dtype=np.float64)
        calibrated = np.interp(s, x, y, left=y[0], right=y[-1])
        return float(_clamp(float(calibrated), 0.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "method": self.method,
            "support_count": self.support_count,
            "x_thresholds": [float(v) for v in self.x_thresholds],
            "y_thresholds": [float(v) for v in self.y_thresholds],
            "score_mean": float(self.score_mean),
            "score_std": float(self.score_std),
            "label_mean": float(self.label_mean),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SegmentCalibrator":
        return cls(
            segment_id=str(payload.get("segment_id") or SEGMENT_GLOBAL),
            method=str(payload.get("method") or "identity"),
            support_count=int(payload.get("support_count") or 0),
            x_thresholds=[float(v) for v in list(payload.get("x_thresholds") or [])],
            y_thresholds=[float(v) for v in list(payload.get("y_thresholds") or [])],
            score_mean=float(payload.get("score_mean") or 0.0),
            score_std=float(payload.get("score_std") or 0.0),
            label_mean=float(payload.get("label_mean") or 0.0),
        )


def _binary_label(value: Any) -> float:
    return 1.0 if _to_float(value, 0.0) >= 2.0 else 0.0


def _compute_metrics(
    samples: Sequence[Tuple[float, float]],
    calibrator: SegmentCalibrator,
    *,
    ece_bins: int = 10,
) -> Dict[str, float]:
    if not samples:
        return {
            "support_count": 0.0,
            "brier": 0.0,
            "logloss": 0.0,
            "ece": 0.0,
            "score_min": 0.0,
            "score_max": 0.0,
            "score_mean": 0.0,
            "score_std": 0.0,
            "label_mean": 0.0,
        }
    scores = np.asarray([_to_float(row[0], 0.0) for row in samples], dtype=np.float64)
    labels = np.asarray([_clamp(_to_float(row[1], 0.0), 0.0, 1.0) for row in samples], dtype=np.float64)
    probs = np.asarray([_clamp(calibrator.apply(float(score)), 1e-6, 1.0 - 1e-6) for score in scores], dtype=np.float64)
    brier = float(np.mean((probs - labels) ** 2))
    logloss = float(-np.mean((labels * np.log(probs)) + ((1.0 - labels) * np.log(1.0 - probs))))
    bins = max(2, int(ece_bins))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        left = edges[idx]
        right = edges[idx + 1]
        mask = (probs >= left) & (probs < right if idx < bins - 1 else probs <= right)
        count = int(np.sum(mask))
        if count <= 0:
            continue
        mean_prob = float(np.mean(probs[mask]))
        mean_label = float(np.mean(labels[mask]))
        ece += (count / len(probs)) * abs(mean_prob - mean_label)
    return {
        "support_count": float(len(samples)),
        "brier": round(brier, 6),
        "logloss": round(logloss, 6),
        "ece": round(float(ece), 6),
        "score_min": round(float(np.min(scores)), 6),
        "score_max": round(float(np.max(scores)), 6),
        "score_mean": round(float(np.mean(scores)), 6),
        "score_std": round(float(np.std(scores)), 6),
        "label_mean": round(float(np.mean(labels)), 6),
    }


class ObjectiveSegmentCalibrators:
    def __init__(
        self,
        *,
        objective: str,
        calibrators: Dict[str, SegmentCalibrator],
        version: str = CALIBRATION_VERSION,
        min_support: int = 25,
        target_definition: str = "p_relevance_ge_2",
        compatibility: Optional[Dict[str, Any]] = None,
        quality: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.objective = objective
        self.calibrators = dict(calibrators)
        self.version = version
        self.min_support = int(min_support)
        self.target_definition = target_definition
        self.compatibility = dict(compatibility or {})
        self.quality = dict(quality or {})

    @classmethod
    def fit(
        cls,
        *,
        objective: str,
        samples: Iterable[Dict[str, Any]],
        known_segments: Sequence[str],
        min_support: int = 25,
        compatibility: Optional[Dict[str, Any]] = None,
    ) -> "ObjectiveSegmentCalibrators":
        by_segment: Dict[str, List[Tuple[float, float]]] = {SEGMENT_GLOBAL: []}
        for segment_id in known_segments:
            by_segment.setdefault(str(segment_id), [])
        for sample in samples:
            segment_id = str(sample.get("segment_id") or SEGMENT_GLOBAL)
            score = _to_float(sample.get("score_raw"), 0.0)
            label = _binary_label(sample.get("label"))
            by_segment.setdefault(segment_id, []).append((score, label))
            by_segment[SEGMENT_GLOBAL].append((score, label))
        calibrators: Dict[str, SegmentCalibrator] = {}
        quality_by_segment: Dict[str, Dict[str, float]] = {}
        for segment_id, segment_samples in by_segment.items():
            calibrator = SegmentCalibrator.fit(
                segment_id=segment_id,
                samples=segment_samples,
                min_support=min_support,
            )
            calibrators[segment_id] = calibrator
            quality_by_segment[segment_id] = _compute_metrics(segment_samples, calibrator)
        overall_metrics = quality_by_segment.get(SEGMENT_GLOBAL, {})
        return cls(
            objective=objective,
            calibrators=calibrators,
            version=CALIBRATION_VERSION,
            min_support=min_support,
            target_definition="p_relevance_ge_2",
            compatibility=compatibility or {},
            quality={
                "overall": overall_metrics,
                "by_segment": quality_by_segment,
            },
        )

    def calibrate(
        self,
        *,
        score_raw: float,
        segment_id: Optional[str],
        include_debug: bool = False,
    ) -> Dict[str, Any]:
        wanted = str(segment_id or SEGMENT_GLOBAL)
        calibrator = self.calibrators.get(wanted)
        fallback_used = False
        used_segment = wanted
        if calibrator is None or calibrator.support_count < self.min_support:
            calibrator = self.calibrators.get(SEGMENT_GLOBAL)
            used_segment = SEGMENT_GLOBAL
            fallback_used = True
        if calibrator is None:
            calibrator = SegmentCalibrator.identity(segment_id=SEGMENT_GLOBAL)
            used_segment = SEGMENT_GLOBAL
            fallback_used = True
        calibrated = calibrator.apply(score_raw)
        return {
            "score_raw": float(score_raw),
            "score_calibrated": float(calibrated),
            "calibrator_segment_id": used_segment,
            "requested_segment_id": wanted,
            "calibrator_method": calibrator.method,
            "calibrator_support_count": int(calibrator.support_count),
            "calibration_fallback_used": bool(fallback_used),
            "target_definition": self.target_definition,
            **(
                {
                    "objective": self.objective,
                    "version": self.version,
                }
                if include_debug
                else {}
            ),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "version": self.version,
            "min_support": self.min_support,
            "target_definition": self.target_definition,
            "compatibility": self.compatibility,
            "quality": self.quality,
            "segments": {
                segment_id: calibrator.to_dict()
                for segment_id, calibrator in sorted(self.calibrators.items())
            },
        }

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.summary(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def compatibility_errors(self, expected: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        for key, expected_value in expected.items():
            actual_value = self.compatibility.get(key)
            if expected_value is None:
                continue
            if actual_value != expected_value:
                errors.append(
                    f"{key}: expected={expected_value!r}, actual={actual_value!r}"
                )
        return errors

    @classmethod
    def load(cls, output_path: Path) -> "ObjectiveSegmentCalibrators":
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        segments_payload = payload.get("segments")
        calibrators: Dict[str, SegmentCalibrator] = {}
        if isinstance(segments_payload, dict):
            for segment_id, segment_data in segments_payload.items():
                if isinstance(segment_data, dict):
                    calibrators[str(segment_id)] = SegmentCalibrator.from_dict(segment_data)
        return cls(
            objective=str(payload.get("objective") or ""),
            calibrators=calibrators,
            version=str(payload.get("version") or CALIBRATION_VERSION),
            min_support=int(payload.get("min_support") or 25),
            target_definition=str(payload.get("target_definition") or "p_relevance_ge_2"),
            compatibility=dict(payload.get("compatibility") or {}),
            quality=dict(payload.get("quality") or {}),
        )
