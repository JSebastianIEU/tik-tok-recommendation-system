from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .temporal import parse_dt


TRAJECTORY_VERSION = "trajectory.v2"
TRAJECTORY_REGIMES = ("spike", "balanced", "durable")


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_table(frame: pd.DataFrame, output_stem: Path) -> Dict[str, str]:
    parquet_path = output_stem.with_suffix(".parquet")
    jsonl_path = output_stem.with_suffix(".jsonl")
    try:
        frame.to_parquet(parquet_path, index=False)
        return {"format": "parquet", "path": parquet_path.name}
    except Exception:
        records = frame.to_dict(orient="records")
        jsonl_path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in records),
            encoding="utf-8",
        )
        return {"format": "jsonl", "path": jsonl_path.name}


def _read_table(base_dir: Path, payload: Dict[str, Any], stem: str) -> List[Dict[str, Any]]:
    table_meta = payload.get(stem)
    if isinstance(table_meta, dict):
        fmt = str(table_meta.get("format") or "").lower()
        path_name = str(table_meta.get("path") or "")
        path = (base_dir / path_name) if path_name else None
        if path is not None and path.exists():
            if fmt == "parquet":
                return pd.read_parquet(path).to_dict(orient="records")
            rows: List[Dict[str, Any]] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    rows.append(parsed)
            return rows
    fallback_parquet = base_dir / f"{stem}.parquet"
    if fallback_parquet.exists():
        return pd.read_parquet(fallback_parquet).to_dict(orient="records")
    fallback_jsonl = base_dir / f"{stem}.jsonl"
    if fallback_jsonl.exists():
        rows: List[Dict[str, Any]] = []
        for line in fallback_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
        return rows
    return []


def _to_utc_iso(value: Any) -> Optional[str]:
    parsed = parse_dt(value)
    if parsed is None:
        return None
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isfinite(parsed):
        return parsed
    return default


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-9:
        return 0.0
    return numerator / denominator


def _normalize_vec(values: Sequence[float], dim: int) -> List[float]:
    out = np.zeros((max(1, int(dim)),), dtype=np.float32)
    if values:
        arr = np.asarray([float(item) for item in values], dtype=np.float32)
        limit = min(out.shape[0], arr.shape[0])
        out[:limit] = arr[:limit]
    norm = float(np.linalg.norm(out))
    if norm > 0.0:
        out = out / norm
    return out.astype(np.float32).tolist()


def _target_availability(
    row: Dict[str, Any],
    objective: str,
    component: Optional[str] = None,
) -> bool:
    payload = row.get("target_availability", {})
    if not isinstance(payload, dict):
        return False
    objective_payload = payload.get(objective, {})
    if not isinstance(objective_payload, dict):
        return False
    if component is None:
        return bool(objective_payload.get("objective_available", False))
    components = objective_payload.get("components", {})
    if not isinstance(components, dict):
        return False
    return bool(components.get(component, False))


def _trajectory_components(row: Dict[str, Any], objective: str) -> Dict[str, float]:
    labels = row.get("labels_trajectory", {})
    if not isinstance(labels, dict):
        return {}
    objective_payload = labels.get(objective, {})
    if not isinstance(objective_payload, dict):
        return {}
    components = objective_payload.get("components", {})
    if not isinstance(components, dict):
        return {}
    out: Dict[str, float] = {}
    for key in ("early_velocity", "core_velocity", "late_lift", "stability"):
        value = components.get(key)
        out[key] = _as_float(value, 0.0) if value is not None else 0.0
    return out


def _trajectory_series(row: Dict[str, Any], objective: str) -> Dict[str, float]:
    labels = row.get("labels_trajectory", {})
    if not isinstance(labels, dict):
        return {}
    objective_payload = labels.get(objective, {})
    if not isinstance(objective_payload, dict):
        return {}
    series = objective_payload.get("series", {})
    if not isinstance(series, dict):
        return {}
    return {
        "t0": _as_float(series.get("t0"), 0.0),
        "t6": _as_float(series.get("t6"), 0.0),
        "t24": _as_float(series.get("t24"), 0.0),
        "t96": _as_float(series.get("t96"), 0.0),
    }


def _trajectory_target_composite(row: Dict[str, Any], objective: str) -> float:
    payload = row.get("targets_trajectory_z", {})
    if not isinstance(payload, dict):
        return 0.0
    objective_payload = payload.get(objective, {})
    if not isinstance(objective_payload, dict):
        return 0.0
    return _as_float(objective_payload.get("composite_z"), 0.0)


def _regime_from_features(
    *,
    early_velocity: float,
    late_lift: float,
    durability_ratio: float,
    stability: float,
) -> Tuple[str, Dict[str, float], float]:
    spike_score = (1.4 * max(early_velocity, 0.0)) + (0.8 * max(-late_lift, 0.0)) + (
        0.7 * max(0.0, 0.35 - durability_ratio)
    )
    durable_score = (1.3 * max(late_lift, 0.0)) + (0.9 * max(durability_ratio, 0.0)) + (
        0.5 * max(stability, 0.0)
    )
    balanced_score = max(
        0.05,
        1.0
        - abs(early_velocity - late_lift)
        - abs(durability_ratio - 0.5)
        + (0.2 * max(stability, 0.0)),
    )
    logits = np.asarray(
        [spike_score, balanced_score, durable_score],
        dtype=np.float64,
    )
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    denom = float(np.sum(probs)) if float(np.sum(probs)) > 0.0 else 1.0
    probs = probs / denom
    regime_probs = {
        "spike": float(round(float(probs[0]), 6)),
        "balanced": float(round(float(probs[1]), 6)),
        "durable": float(round(float(probs[2]), 6)),
    }
    regime_pred = max(regime_probs.items(), key=lambda item: item[1])[0]
    sorted_probs = sorted(regime_probs.values(), reverse=True)
    confidence = (
        float(sorted_probs[0] - sorted_probs[1])
        if len(sorted_probs) >= 2
        else float(sorted_probs[0] if sorted_probs else 0.0)
    )
    return regime_pred, regime_probs, float(round(_clip(confidence, 0.0, 1.0), 6))


@dataclass
class TrajectoryBuildConfig:
    windows_hours: Tuple[int, int, int] = (6, 24, 96)
    embedding_dim: int = 16
    feature_version: str = "trajectory_features.v2"
    encoder_mode: str = "feature_only"

    def __post_init__(self) -> None:
        if len(self.windows_hours) != 3:
            raise ValueError("windows_hours must contain exactly three entries.")
        coerced = tuple(int(value) for value in self.windows_hours)
        if any(value < 1 for value in coerced):
            raise ValueError("windows_hours values must be >= 1.")
        if tuple(sorted(coerced)) != coerced:
            raise ValueError("windows_hours must be strictly increasing.")
        self.windows_hours = coerced
        self.embedding_dim = max(4, int(self.embedding_dim))
        self.feature_version = str(self.feature_version or "trajectory_features.v2")
        self.encoder_mode = str(self.encoder_mode or "feature_only")


@dataclass
class TrajectoryBundle:
    version: str
    trajectory_manifest_id: str
    trajectory_schema_hash: str
    config: TrajectoryBuildConfig
    created_at: str
    profiles: List[Dict[str, Any]]
    embeddings_by_video: Dict[str, List[float]]
    profile_by_video: Dict[str, Dict[str, Any]]

    def query_embedding(self, query_row: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        features = query_row.get("features")
        trajectory_features = (
            features.get("trajectory_features")
            if isinstance(features, dict)
            else None
        )
        if not isinstance(trajectory_features, dict):
            source_payload = query_row.get("_source_payload")
            hints = source_payload.get("hints") if isinstance(source_payload, dict) else None
            trajectory_features = (
                hints.get("trajectory")
                if isinstance(hints, dict) and isinstance(hints.get("trajectory"), dict)
                else {}
            )
        video_id = str(query_row.get("video_id") or query_row.get("row_id") or "").split("::", 1)[0]
        if not trajectory_features and video_id in self.profile_by_video:
            trajectory_features = dict(self.profile_by_video[video_id].get("features") or {})
        if not isinstance(trajectory_features, dict):
            trajectory_features = {}

        base_vector = _trajectory_feature_vector(trajectory_features)
        vector = np.asarray(
            _normalize_vec(base_vector, self.config.embedding_dim),
            dtype=np.float32,
        )
        return vector, {
            "video_id": video_id,
            "source": (
                "row_feature"
                if bool(trajectory_features)
                else "unavailable"
            ),
            "regime_pred": str(trajectory_features.get("regime_pred") or "balanced"),
            "regime_confidence": float(_as_float(trajectory_features.get("regime_confidence"), 0.0)),
        }

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        profile_rows = list(self.profiles)
        embedding_rows = [
            {
                "video_id": video_id,
                "embedding": json.dumps(vector, ensure_ascii=False),
            }
            for video_id, vector in sorted(self.embeddings_by_video.items())
        ]
        profile_frame = pd.DataFrame(profile_rows)
        embedding_frame = pd.DataFrame(embedding_rows)
        profile_table = _write_table(profile_frame, output_dir / "profiles")
        embedding_table = _write_table(embedding_frame, output_dir / "embeddings")
        payload = {
            "version": self.version,
            "trajectory_manifest_id": self.trajectory_manifest_id,
            "trajectory_schema_hash": self.trajectory_schema_hash,
            "created_at": self.created_at,
            "config": asdict(self.config),
            "tables": {
                "profiles": profile_table,
                "embeddings": embedding_table,
            },
            "profile_count": len(self.profiles),
            "embedding_count": len(self.embeddings_by_video),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with (output_dir / "bundle.pkl").open("wb") as fh:
            import pickle

            pickle.dump(self, fh)
        return output_dir / "manifest.json"

    @classmethod
    def load(cls, output_dir: Path) -> "TrajectoryBundle":
        pickle_path = output_dir / "bundle.pkl"
        if pickle_path.exists():
            with pickle_path.open("rb") as fh:
                import pickle

                loaded = pickle.load(fh)
            if isinstance(loaded, TrajectoryBundle):
                return loaded

        payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        tables = payload.get("tables") if isinstance(payload.get("tables"), dict) else {}
        profiles = _read_table(output_dir, dict(tables), "profiles")
        embedding_rows = _read_table(output_dir, dict(tables), "embeddings")
        embeddings_by_video: Dict[str, List[float]] = {}
        for row in embedding_rows:
            video_id = str(row.get("video_id") or "").strip()
            if not video_id:
                continue
            raw = row.get("embedding")
            values: List[float] = []
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        values = [float(item) for item in parsed]
                except json.JSONDecodeError:
                    values = []
            elif isinstance(raw, list):
                values = [float(item) for item in raw]
            embeddings_by_video[video_id] = values
        profile_by_video: Dict[str, Dict[str, Any]] = {}
        for row in profiles:
            video_id = str(row.get("video_id") or "").strip()
            if not video_id:
                continue
            profile_by_video[video_id] = row
        return cls(
            version=str(payload.get("version") or TRAJECTORY_VERSION),
            trajectory_manifest_id=str(
                payload.get("trajectory_manifest_id") or output_dir.name
            ),
            trajectory_schema_hash=str(payload.get("trajectory_schema_hash") or ""),
            config=TrajectoryBuildConfig(**dict(payload.get("config") or {})),
            created_at=str(payload.get("created_at") or _to_utc_iso(datetime.now(timezone.utc))),
            profiles=profiles,
            embeddings_by_video=embeddings_by_video,
            profile_by_video=profile_by_video,
        )


def _profile_from_row(row: Dict[str, Any], *, windows_hours: Tuple[int, int, int]) -> Dict[str, Any]:
    reach_components = _trajectory_components(row, "reach")
    engagement_components = _trajectory_components(row, "engagement")
    conversion_components = _trajectory_components(row, "conversion")
    reach_series = _trajectory_series(row, "reach")
    h6, h24, h96 = windows_hours

    def _component_mean(component_key: str) -> float:
        values = [
            _as_float(payload.get(component_key), 0.0)
            for payload in (reach_components, engagement_components, conversion_components)
        ]
        return float(round(sum(values) / max(1, len(values)), 6))

    early_velocity = _component_mean("early_velocity")
    core_velocity = _component_mean("core_velocity")
    late_lift = _component_mean("late_lift")
    stability = _component_mean("stability")
    late_velocity = late_lift / max(1.0, float(h96 - h24))
    acceleration = core_velocity - early_velocity
    curvature = late_velocity - core_velocity
    durability_ratio = _safe_ratio(
        max(0.0, late_lift),
        max(1e-6, abs(_as_float(reach_series.get("t24"), 0.0) - _as_float(reach_series.get("t0"), 0.0))),
    )

    series_points = {
        0.0: _as_float(reach_series.get("t0"), 0.0),
        float(h6): _as_float(reach_series.get("t6"), 0.0),
        float(h24): _as_float(reach_series.get("t24"), 0.0),
        float(h96): _as_float(reach_series.get("t96"), 0.0),
    }
    peak_lag_hours = max(series_points.items(), key=lambda item: item[1])[0]
    regime_pred, regime_probs, regime_confidence = _regime_from_features(
        early_velocity=early_velocity,
        late_lift=late_lift,
        durability_ratio=durability_ratio,
        stability=stability,
    )

    availability_flags = []
    for objective in ("reach", "engagement", "conversion"):
        for component in ("early_velocity", "core_velocity", "late_lift", "stability"):
            availability_flags.append(
                1.0 if _target_availability(row, objective, component) else 0.0
            )
    available_ratio = float(round(sum(availability_flags) / max(1, len(availability_flags)), 6))
    missing_count = int(len(availability_flags) - int(sum(availability_flags)))
    objective_composites = {
        objective: _trajectory_target_composite(row, objective)
        for objective in ("reach", "engagement", "conversion")
    }

    return {
        "video_id": str(row.get("video_id") or str(row.get("row_id") or "").split("::", 1)[0]),
        "as_of_time": _to_utc_iso(row.get("as_of_time")),
        "row_id": str(row.get("row_id") or ""),
        "features": {
            "early_velocity": float(round(early_velocity, 6)),
            "core_velocity": float(round(core_velocity, 6)),
            "late_lift": float(round(late_lift, 6)),
            "stability": float(round(stability, 6)),
            "late_velocity": float(round(late_velocity, 6)),
            "acceleration_proxy": float(round(acceleration, 6)),
            "curvature_proxy": float(round(curvature, 6)),
            "durability_ratio": float(round(durability_ratio, 6)),
            "peak_lag_hours": float(round(peak_lag_hours, 6)),
            "available_ratio": available_ratio,
            "missing_component_count": int(missing_count),
            "regime_pred": regime_pred,
            "regime_probabilities": regime_probs,
            "regime_confidence": regime_confidence,
            "objectives": {
                objective: {
                    "composite_z": float(round(value, 6)),
                    "objective_available": _target_availability(row, objective),
                    "components": {
                        "early_velocity": _as_float(
                            _trajectory_components(row, objective).get("early_velocity"), 0.0
                        ),
                        "core_velocity": _as_float(
                            _trajectory_components(row, objective).get("core_velocity"), 0.0
                        ),
                        "late_lift": _as_float(
                            _trajectory_components(row, objective).get("late_lift"), 0.0
                        ),
                        "stability": _as_float(
                            _trajectory_components(row, objective).get("stability"), 0.0
                        ),
                    },
                }
                for objective, value in objective_composites.items()
            },
        },
    }


def _trajectory_feature_vector(features: Dict[str, Any]) -> List[float]:
    probs = features.get("regime_probabilities")
    if not isinstance(probs, dict):
        probs = {}
    return [
        _as_float(features.get("early_velocity"), 0.0),
        _as_float(features.get("core_velocity"), 0.0),
        _as_float(features.get("late_lift"), 0.0),
        _as_float(features.get("stability"), 0.0),
        _as_float(features.get("late_velocity"), 0.0),
        _as_float(features.get("acceleration_proxy"), 0.0),
        _as_float(features.get("curvature_proxy"), 0.0),
        _as_float(features.get("durability_ratio"), 0.0),
        _as_float(features.get("peak_lag_hours"), 0.0) / 96.0,
        _as_float(probs.get("spike"), 0.0),
        _as_float(probs.get("balanced"), 0.0),
        _as_float(probs.get("durable"), 0.0),
        _as_float(features.get("regime_confidence"), 0.0),
        _as_float(features.get("available_ratio"), 0.0),
    ]


def build_trajectory_bundle(
    rows: Sequence[Dict[str, Any]],
    *,
    as_of_time: Optional[Any] = None,
    run_cutoff_time: Optional[Any] = None,
    config: Optional[TrajectoryBuildConfig] = None,
) -> TrajectoryBundle:
    cfg = config or TrajectoryBuildConfig()
    as_of = parse_dt(as_of_time)
    run_cutoff = parse_dt(run_cutoff_time) or as_of

    latest_by_video: Dict[str, Tuple[datetime, Dict[str, Any]]] = {}
    for row in rows:
        row_as_of = parse_dt(row.get("as_of_time"))
        if row_as_of is None:
            continue
        row_ingested = parse_dt(row.get("ingested_at")) or row_as_of
        if as_of is not None and row_as_of > as_of:
            continue
        if run_cutoff is not None and row_ingested > run_cutoff:
            continue
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            row_id = str(row.get("row_id") or "")
            video_id = row_id.split("::", 1)[0] if row_id else ""
        if not video_id:
            continue
        existing = latest_by_video.get(video_id)
        if existing is None or row_as_of > existing[0]:
            latest_by_video[video_id] = (row_as_of, row)

    profiles: List[Dict[str, Any]] = []
    embeddings_by_video: Dict[str, List[float]] = {}
    profile_by_video: Dict[str, Dict[str, Any]] = {}
    for video_id, (_, row) in sorted(latest_by_video.items()):
        profile = _profile_from_row(row, windows_hours=cfg.windows_hours)
        profiles.append(profile)
        profile_by_video[video_id] = profile
        base_vector = _trajectory_feature_vector(profile["features"])
        embeddings_by_video[video_id] = _normalize_vec(base_vector, cfg.embedding_dim)

    schema_hash = _stable_hash(
        {
            "version": TRAJECTORY_VERSION,
            "feature_version": cfg.feature_version,
            "encoder_mode": cfg.encoder_mode,
            "windows_hours": list(cfg.windows_hours),
            "embedding_dim": cfg.embedding_dim,
            "regimes": list(TRAJECTORY_REGIMES),
        }
    )
    manifest_id = _stable_hash(
        {
            "schema_hash": schema_hash,
            "profiles_hash": _stable_hash(
                [
                    {
                        "video_id": item.get("video_id"),
                        "as_of_time": item.get("as_of_time"),
                        "features": item.get("features"),
                    }
                    for item in profiles
                ]
            ),
            "embedding_hash": _stable_hash(
                {
                    key: value[:8]
                    for key, value in sorted(embeddings_by_video.items())
                }
            ),
        }
    )[:16]
    return TrajectoryBundle(
        version=TRAJECTORY_VERSION,
        trajectory_manifest_id=manifest_id,
        trajectory_schema_hash=schema_hash,
        config=cfg,
        created_at=_to_utc_iso(datetime.now(timezone.utc)) or "",
        profiles=profiles,
        embeddings_by_video=embeddings_by_video,
        profile_by_video=profile_by_video,
    )


def annotate_rows_with_trajectory_features(
    rows: Iterable[Dict[str, Any]],
    bundle: Optional[TrajectoryBundle],
) -> None:
    if bundle is None:
        return
    for row in rows:
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            row_id = str(row.get("row_id") or "")
            video_id = row_id.split("::", 1)[0] if row_id else ""
        if not video_id:
            continue
        profile = bundle.profile_by_video.get(video_id)
        if not isinstance(profile, dict):
            continue
        features = row.get("features")
        if not isinstance(features, dict):
            continue
        trajectory_features = profile.get("features")
        if not isinstance(trajectory_features, dict):
            continue
        features["trajectory_features"] = dict(trajectory_features)
        row["_trajectory_profile"] = {
            "video_id": video_id,
            "as_of_time": profile.get("as_of_time"),
            "trajectory_manifest_id": bundle.trajectory_manifest_id,
            "trajectory_version": bundle.version,
        }


__all__ = [
    "TRAJECTORY_VERSION",
    "TRAJECTORY_REGIMES",
    "TrajectoryBuildConfig",
    "TrajectoryBundle",
    "build_trajectory_bundle",
    "annotate_rows_with_trajectory_features",
]
