from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .artifacts import ArtifactRegistry
from .calibration import CALIBRATION_VERSION, ObjectiveSegmentCalibrators
from .explainability import (
    ExplainabilityControls,
    as_payload as explainability_as_payload,
    counterfactual_scenarios_for_items,
    feature_contribution_card,
    metadata_payload as explainability_metadata_payload,
    neighbor_evidence_card,
    temporal_confidence_band,
)
from .objectives import map_objective
from .policy import POLICY_RERANK_VERSION, PolicyReranker, PolicyRerankerConfig
from .ranker import RankerFamilyModel
from .retriever import HybridRetriever
from .trajectory import TrajectoryBundle
from .temporal import parse_dt
from ..comment_intelligence import load_comment_intelligence_snapshot_manifest
from ..fabric import FABRIC_VERSION, FeatureFabric


class RoutingContractError(ValueError):
    pass


class ArtifactCompatibilityError(ValueError):
    pass


class RecommenderStageTimeoutError(TimeoutError):
    def __init__(self, stage: str, elapsed_ms: float, budget_ms: float) -> None:
        self.stage = str(stage)
        self.elapsed_ms = float(elapsed_ms)
        self.budget_ms = float(budget_ms)
        super().__init__(
            f"{self.stage}_timeout elapsed_ms={self.elapsed_ms:.3f} budget_ms={self.budget_ms:.3f}"
        )


def _to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _token_count(text: str) -> int:
    return len([token for token in text.split() if token.strip()])


def _extract_hashtag_count(text: str) -> int:
    return len([token for token in text.split() if token.strip().startswith("#")])


def _as_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    if np.isfinite(out):
        return out
    return fallback


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _minmax_scale(value: float, low: float, high: float) -> float:
    spread = high - low
    if spread <= 1e-9:
        return 0.5
    return (value - low) / spread


def _resolve_portfolio_weights(payload: Optional[Dict[str, Any]]) -> Dict[str, float]:
    defaults = {"reach": 0.45, "conversion": 0.35, "durability": 0.20}
    weights_raw = payload.get("weights") if isinstance(payload, dict) else {}
    if not isinstance(weights_raw, dict):
        return defaults
    parsed = {
        "reach": max(0.0, _as_float(weights_raw.get("reach"), defaults["reach"])),
        "conversion": max(
            0.0, _as_float(weights_raw.get("conversion"), defaults["conversion"])
        ),
        "durability": max(
            0.0, _as_float(weights_raw.get("durability"), defaults["durability"])
        ),
    }
    total = parsed["reach"] + parsed["conversion"] + parsed["durability"]
    if total <= 1e-9:
        return defaults
    return {
        "reach": parsed["reach"] / total,
        "conversion": parsed["conversion"] / total,
        "durability": parsed["durability"] / total,
    }


def _normalize_language(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    return cleaned[:8]


def _normalize_locale(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    return cleaned[:24]


def _derive_language(language: Any, locale: Any) -> Optional[str]:
    normalized = _normalize_language(language)
    if normalized:
        return normalized
    normalized_locale = _normalize_locale(locale)
    if not normalized_locale:
        return None
    return normalized_locale.split("-", 1)[0]


def _manifest_ref_candidates(
    *,
    bundle_dir: Path,
    manifest_path: Optional[str],
    manifest_id: Optional[str],
) -> List[Path]:
    out: List[Path] = []
    if isinstance(manifest_path, str) and manifest_path.strip():
        path = Path(manifest_path.strip())
        out.append(path / "manifest.json" if path.is_dir() else path)
    if isinstance(manifest_id, str) and manifest_id.strip():
        out.extend(
            [
                bundle_dir.parent.parent
                / "comment_intelligence"
                / "features"
                / manifest_id.strip()
                / "manifest.json",
                Path("artifacts")
                / "comment_intelligence"
                / "features"
                / manifest_id.strip()
                / "manifest.json",
            ]
        )
    deduped: List[Path] = []
    seen: set[str] = set()
    for path in out:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _candidate_video_id(row_id: str) -> str:
    if "::" in row_id:
        return row_id.split("::", 1)[0]
    return row_id


class _CommentManifestIndex:
    def __init__(self, rows: Sequence[Dict[str, Any]]) -> None:
        by_video: Dict[str, List[Tuple[datetime, Dict[str, Any]]]] = {}
        for row in rows:
            video_id = str(row.get("video_id") or "").strip()
            if not video_id:
                continue
            parsed = parse_dt(row.get("as_of_time"))
            if parsed is None:
                continue
            by_video.setdefault(video_id, []).append((parsed, row))
        for values in by_video.values():
            values.sort(key=lambda item: item[0])
        self.by_video = by_video

    def lookup(self, *, video_id: str, as_of: datetime) -> Optional[Dict[str, Any]]:
        values = self.by_video.get(video_id)
        if not values:
            return None
        out: Optional[Dict[str, Any]] = None
        for ts, row in values:
            if ts <= as_of:
                out = row
            else:
                break
        return out


class _TrajectoryManifestIndex:
    def __init__(self, rows: Sequence[Dict[str, Any]]) -> None:
        by_video: Dict[str, List[Tuple[datetime, Dict[str, Any]]]] = {}
        for row in rows:
            video_id = str(row.get("video_id") or "").strip()
            if not video_id:
                continue
            parsed = parse_dt(row.get("as_of_time"))
            if parsed is None:
                continue
            by_video.setdefault(video_id, []).append((parsed, row))
        for values in by_video.values():
            values.sort(key=lambda item: item[0])
        self.by_video = by_video

    def lookup(self, *, video_id: str, as_of: datetime) -> Optional[Dict[str, Any]]:
        values = self.by_video.get(video_id)
        if not values:
            return None
        out: Optional[Dict[str, Any]] = None
        for ts, row in values:
            if ts <= as_of:
                out = row
            else:
                break
        return out


def _coerce_manifest_comment_intelligence(row: Dict[str, Any]) -> Dict[str, Any]:
    features = row.get("features")
    if not isinstance(features, dict):
        return {}
    missingness = row.get("missingness")
    missingness_flags = (
        sorted(str(key) for key in missingness.keys())
        if isinstance(missingness, dict)
        else []
    )
    comment_count = int(features.get("comment_count_total") or 0)
    dominant = features.get("dominant_intents")
    dominant_intents = (
        [str(item) for item in dominant if str(item).strip()]
        if isinstance(dominant, list)
        else []
    )
    return {
        "source": "manifest_snapshot",
        "available": comment_count > 0,
        "taxonomy_version": str(row.get("taxonomy_version") or ""),
        "dominant_intents": dominant_intents,
        "confusion_index": float(features.get("confusion_index") or 0.0),
        "help_seeking_index": float(features.get("help_seeking_index") or 0.0),
        "sentiment_volatility": float(features.get("sentiment_volatility") or 0.0),
        "sentiment_shift_early_late": float(features.get("sentiment_shift_early_late") or 0.0),
        "reply_depth_max": float(features.get("reply_depth_max") or 0.0),
        "reply_branch_factor": float(features.get("reply_branch_factor") or 0.0),
        "reply_ratio": float(features.get("reply_ratio") or 0.0),
        "root_thread_concentration": float(features.get("root_thread_concentration") or 0.0),
        "alignment_score": float(features.get("alignment_score") or 0.0),
        "value_prop_coverage": float(features.get("value_prop_coverage") or 0.0),
        "on_topic_ratio": float(features.get("on_topic_ratio") or 0.0),
        "artifact_drift_ratio": float(features.get("artifact_drift_ratio") or 0.0),
        "alignment_shift_early_late": float(features.get("alignment_shift_early_late") or 0.0),
        "alignment_confidence": float(features.get("alignment_confidence") or 0.0),
        "alignment_method_version": str(features.get("alignment_method_version") or ""),
        "confidence": float(features.get("confidence") or 0.0),
        "missingness_flags": missingness_flags,
    }


def _coerce_manifest_trajectory_features(row: Dict[str, Any]) -> Dict[str, Any]:
    features = row.get("features")
    if not isinstance(features, dict):
        return {}
    regime_probs = features.get("regime_probabilities")
    if not isinstance(regime_probs, dict):
        regime_probs = {}
    objectives = features.get("objectives")
    if not isinstance(objectives, dict):
        objectives = {}
    return {
        "source": "trajectory_manifest",
        "early_velocity": float(features.get("early_velocity") or 0.0),
        "core_velocity": float(features.get("core_velocity") or 0.0),
        "late_lift": float(features.get("late_lift") or 0.0),
        "stability": float(features.get("stability") or 0.0),
        "late_velocity": float(features.get("late_velocity") or 0.0),
        "acceleration_proxy": float(features.get("acceleration_proxy") or 0.0),
        "curvature_proxy": float(features.get("curvature_proxy") or 0.0),
        "durability_ratio": float(features.get("durability_ratio") or 0.0),
        "peak_lag_hours": float(features.get("peak_lag_hours") or 0.0),
        "available_ratio": float(features.get("available_ratio") or 0.0),
        "missing_component_count": int(features.get("missing_component_count") or 0),
        "regime_pred": str(features.get("regime_pred") or "balanced"),
        "regime_probabilities": {
            "spike": float(regime_probs.get("spike") or 0.0),
            "balanced": float(regime_probs.get("balanced") or 0.0),
            "durable": float(regime_probs.get("durable") or 0.0),
        },
        "regime_confidence": float(features.get("regime_confidence") or 0.0),
        "objectives": {
            str(key): value
            for key, value in objectives.items()
            if isinstance(value, dict)
        },
    }


def _coerce_comment_intelligence(payload: Dict[str, Any]) -> Dict[str, Any]:
    hints = payload.get("hints")
    if not isinstance(hints, dict):
        return {}
    raw = hints.get("comment_intelligence")
    if not isinstance(raw, dict):
        return {}
    dominant = raw.get("dominant_intents")
    dominant_intents = (
        [str(item) for item in dominant if str(item).strip()] if isinstance(dominant, list) else []
    )
    return {
        "source": str(raw.get("source") or "request_hint"),
        "available": bool(raw.get("available", True)),
        "taxonomy_version": str(raw.get("taxonomy_version") or ""),
        "dominant_intents": dominant_intents,
        "confusion_index": float(raw.get("confusion_index") or 0.0),
        "help_seeking_index": float(raw.get("help_seeking_index") or 0.0),
        "sentiment_volatility": float(raw.get("sentiment_volatility") or 0.0),
        "sentiment_shift_early_late": float(raw.get("sentiment_shift_early_late") or 0.0),
        "reply_depth_max": float(raw.get("reply_depth_max") or 0.0),
        "reply_branch_factor": float(raw.get("reply_branch_factor") or 0.0),
        "reply_ratio": float(raw.get("reply_ratio") or 0.0),
        "root_thread_concentration": float(raw.get("root_thread_concentration") or 0.0),
        "alignment_score": float(raw.get("alignment_score") or 0.0),
        "value_prop_coverage": float(raw.get("value_prop_coverage") or 0.0),
        "on_topic_ratio": float(raw.get("on_topic_ratio") or 0.0),
        "artifact_drift_ratio": float(raw.get("artifact_drift_ratio") or 0.0),
        "alignment_shift_early_late": float(raw.get("alignment_shift_early_late") or 0.0),
        "alignment_confidence": float(raw.get("alignment_confidence") or 0.0),
        "alignment_method_version": str(raw.get("alignment_method_version") or ""),
        "confidence": float(raw.get("confidence") or 0.0),
        "missingness_flags": [str(item) for item in list(raw.get("missingness_flags") or [])],
    }


def _coerce_trajectory_features(payload: Dict[str, Any]) -> Dict[str, Any]:
    hints = payload.get("hints")
    if not isinstance(hints, dict):
        return {}
    raw = hints.get("trajectory")
    if not isinstance(raw, dict):
        return {}
    regime_probs = raw.get("regime_probabilities")
    if not isinstance(regime_probs, dict):
        regime_probs = {}
    objectives = raw.get("objectives")
    if not isinstance(objectives, dict):
        objectives = {}
    return {
        "source": str(raw.get("source") or "request_hint"),
        "early_velocity": float(raw.get("early_velocity") or 0.0),
        "core_velocity": float(raw.get("core_velocity") or 0.0),
        "late_lift": float(raw.get("late_lift") or 0.0),
        "stability": float(raw.get("stability") or 0.0),
        "late_velocity": float(raw.get("late_velocity") or 0.0),
        "acceleration_proxy": float(raw.get("acceleration_proxy") or 0.0),
        "curvature_proxy": float(raw.get("curvature_proxy") or 0.0),
        "durability_ratio": float(raw.get("durability_ratio") or 0.0),
        "peak_lag_hours": float(raw.get("peak_lag_hours") or 0.0),
        "available_ratio": float(raw.get("available_ratio") or 0.0),
        "missing_component_count": int(raw.get("missing_component_count") or 0),
        "regime_pred": str(raw.get("regime_pred") or "balanced"),
        "regime_probabilities": {
            "spike": float(regime_probs.get("spike") or 0.0),
            "balanced": float(regime_probs.get("balanced") or 0.0),
            "durable": float(regime_probs.get("durable") or 0.0),
        },
        "regime_confidence": float(raw.get("regime_confidence") or 0.0),
        "objectives": {
            str(key): value
            for key, value in objectives.items()
            if isinstance(value, dict)
        },
    }


def _to_runtime_row(row_id: str, payload: Dict[str, Any], fallback_as_of: datetime) -> Dict[str, Any]:
    text = _safe_text(payload.get("text"))
    topic_key = _safe_text(payload.get("topic_key")) or "general"
    author_id = _safe_text(payload.get("author_id")) or "unknown"
    locale = _normalize_locale(payload.get("locale"))
    language = _derive_language(payload.get("language"), locale)
    content_type = _safe_text(payload.get("content_type")) or "general"
    as_of = parse_dt(payload.get("as_of_time")) or fallback_as_of
    return {
        "row_id": row_id,
        "video_id": row_id,
        "author_id": author_id,
        "topic_key": topic_key,
        "locale": locale,
        "content_type": content_type,
        "as_of_time": as_of,
        "features": {
            "caption_word_count": _token_count(text),
            "hashtag_count": _extract_hashtag_count(text),
            "keyword_count": len(
                [token for token in text.lower().split() if len(token) >= 4]
            ),
            "language": language,
            "comment_intelligence": _coerce_comment_intelligence(payload),
            "trajectory_features": _coerce_trajectory_features(payload),
            "missingness_flags": [],
        },
        "_runtime_text": text,
        "_source_payload": payload,
        "_comment_intelligence": _coerce_comment_intelligence(payload),
        "_trajectory_features": _coerce_trajectory_features(payload),
    }


def _to_runtime_row_with_fabric(
    row_id: str,
    payload: Dict[str, Any],
    fallback_as_of: datetime,
    fabric: FeatureFabric,
) -> Dict[str, Any]:
    as_of = parse_dt(payload.get("as_of_time")) or fallback_as_of
    text = _safe_text(payload.get("text"))
    locale = _normalize_locale(payload.get("locale"))
    language = _derive_language(payload.get("language"), locale)
    content_type = _safe_text(payload.get("content_type")) or "general"
    hashtags = [token for token in text.split() if token.strip().startswith("#")]
    extracted = fabric.extract(
        {
            "video_id": row_id,
            "as_of_time": as_of,
            "caption": text,
            "hashtags": hashtags,
            "keywords": [token for token in text.split() if len(token) >= 4][:16],
            "duration_seconds": payload.get("duration_seconds"),
            "content_type": content_type,
            "hints": payload.get("hints") if isinstance(payload.get("hints"), dict) else {},
        }
    )
    comment_intelligence = _coerce_comment_intelligence(payload)
    trajectory_features = _coerce_trajectory_features(payload)
    return {
        "row_id": row_id,
        "video_id": row_id,
        "author_id": _safe_text(payload.get("author_id")) or "unknown",
        "topic_key": _safe_text(payload.get("topic_key")) or "general",
        "locale": locale,
        "content_type": content_type,
        "as_of_time": as_of,
        "features": {
            "caption_word_count": int(extracted.text.token_count),
            "hashtag_count": int(extracted.text.hashtag_count),
            "keyword_count": int(extracted.text.keyphrase_count),
            "language": language,
            "comment_intelligence": comment_intelligence,
            "trajectory_features": trajectory_features,
            "missingness_flags": sorted(
                {
                    *extracted.text.missing.keys(),
                    *extracted.audio.missing.keys(),
                    *extracted.visual.missing.keys(),
                }
            ),
        },
        "_runtime_text": text,
        "_source_payload": payload,
        "_fabric_output": extracted.model_dump(mode="python"),
        "_comment_intelligence": comment_intelligence,
        "_trajectory_features": trajectory_features,
    }


@dataclass
class RecommenderRuntimeConfig:
    feature_schema_hash: Optional[str] = None
    contract_version: Optional[str] = None
    datamart_version: Optional[str] = None
    component: Optional[str] = "recommender-learning-v1"


class RecommenderRuntime:
    def __init__(
        self,
        bundle_dir: Path,
        config: Optional[RecommenderRuntimeConfig] = None,
    ) -> None:
        self.bundle_dir = bundle_dir
        self.config = config or RecommenderRuntimeConfig()
        registry = ArtifactRegistry(bundle_dir.parent)
        manifest = registry.load_manifest(bundle_dir)
        required_fields = (
            "component",
            "contract_version",
            "datamart_version",
            "feature_schema_hash",
            "objectives",
        )
        missing = [field for field in required_fields if field not in manifest]
        if missing:
            raise ValueError(
                f"Invalid recommender artifact manifest. Missing fields: {', '.join(missing)}"
            )

        expected: Dict[str, Any] = {}
        if self.config.component:
            expected["component"] = self.config.component
        if self.config.feature_schema_hash:
            expected["feature_schema_hash"] = self.config.feature_schema_hash
        if self.config.contract_version:
            expected["contract_version"] = self.config.contract_version
        if self.config.datamart_version:
            expected["datamart_version"] = self.config.datamart_version
        if expected:
            registry.assert_compatible(bundle_dir, expected)
        self.manifest = manifest
        self.comment_index: Optional[_CommentManifestIndex] = None
        self.comment_index_source_path: Optional[str] = None
        self.comment_index_load_error: Optional[str] = None
        self._load_comment_feature_index()
        self.trajectory_index: Optional[_TrajectoryManifestIndex] = None
        self.trajectory_index_source_path: Optional[str] = None
        self.trajectory_index_load_error: Optional[str] = None
        self._load_trajectory_feature_index()
        self.trajectory_bundle: Optional[TrajectoryBundle] = None
        trajectory_bundle_dir = bundle_dir / "trajectory"
        if trajectory_bundle_dir.exists():
            try:
                self.trajectory_bundle = TrajectoryBundle.load(trajectory_bundle_dir)
            except Exception:
                self.trajectory_bundle = None
        calibration_path = os.getenv("FABRIC_CALIBRATION_PATH", "").strip()
        if calibration_path:
            path = Path(calibration_path)
            if path.exists():
                try:
                    calibration_payload = json.loads(path.read_text(encoding="utf-8"))
                    self.fabric = FeatureFabric(calibration_artifacts=calibration_payload)
                except Exception:
                    self.fabric = FeatureFabric()
            else:
                self.fabric = FeatureFabric()
        else:
            self.fabric = FeatureFabric()
        expected_fabric_signature = manifest.get("fabric_registry_signature")
        if (
            isinstance(expected_fabric_signature, str)
            and expected_fabric_signature
            and self.fabric.registry.signature() != expected_fabric_signature
        ):
            raise ValueError(
                "Fabric registry signature mismatch between runtime and artifact manifest."
            )
        expected_fabric_schema_hashes = manifest.get("fabric_schema_hashes")
        if isinstance(expected_fabric_schema_hashes, dict) and expected_fabric_schema_hashes:
            self.fabric.registry.assert_compatible(
                {
                    str(key): str(value)
                    for key, value in expected_fabric_schema_hashes.items()
                }
            )
        self.retriever = HybridRetriever.load(bundle_dir / "retriever")
        self.rankers: Dict[str, RankerFamilyModel] = {}
        self.calibrators: Dict[str, ObjectiveSegmentCalibrators] = {}
        self.calibration_load_warnings: Dict[str, str] = {}
        for objective in manifest.get("objectives", []):
            ranker_dir = bundle_dir / "rankers" / str(objective)
            if ranker_dir.exists():
                self.rankers[str(objective)] = RankerFamilyModel.load(ranker_dir)
                calibration_path = ranker_dir / "calibration.json"
                if calibration_path.exists():
                    try:
                        loaded = ObjectiveSegmentCalibrators.load(calibration_path)
                        compatibility_errors = loaded.compatibility_errors(
                            {
                                "objective": str(objective),
                                "feature_schema_hash": self.manifest.get("feature_schema_hash"),
                                "ranker_family_schema_hash": self.manifest.get("ranker_family_schema_hash"),
                                "ranker_family_version": "ranker_family.v2",
                            }
                        )
                        if str(loaded.version) != CALIBRATION_VERSION:
                            compatibility_errors.append(
                                f"version mismatch expected={CALIBRATION_VERSION!r} actual={loaded.version!r}"
                            )
                        if compatibility_errors:
                            self.calibration_load_warnings[str(objective)] = (
                                "calibration_incompatible: " + " | ".join(compatibility_errors)
                            )
                        else:
                            self.calibrators[str(objective)] = loaded
                    except Exception as error:
                        self.calibration_load_warnings[str(objective)] = (
                            f"calibration_load_failed: {error}"
                        )
                        continue
        missing_rankers = [
            str(objective)
            for objective in manifest.get("objectives", [])
            if str(objective) not in self.rankers
        ]
        if missing_rankers:
            raise ValueError(
                "Invalid recommender artifact bundle. Missing ranker models for: "
                + ", ".join(sorted(missing_rankers))
            )
        self.policy_reranker = PolicyReranker(
            PolicyRerankerConfig.from_payload(
                self.manifest.get("policy_reranker")
                if isinstance(self.manifest.get("policy_reranker"), dict)
                else None
            )
        )
        self.bundle_id = str(
            self.manifest.get("bundle_id")
            or self.manifest.get("run_name")
            or self.bundle_dir.name
        )

    def compatibility_payload(
        self,
        required_compat: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        required = {
            str(key): str(value)
            for key, value in (required_compat or {}).items()
            if str(key).strip()
        }
        fingerprints = {
            "bundle_id": self.bundle_id,
            "component": str(self.manifest.get("component") or ""),
            "contract_version": str(self.manifest.get("contract_version") or ""),
            "datamart_version": str(self.manifest.get("datamart_version") or ""),
            "feature_schema_hash": str(self.manifest.get("feature_schema_hash") or ""),
            "ranker_family_schema_hash": str(self.manifest.get("ranker_family_schema_hash") or ""),
            "ranker_family_version": "ranker_family.v2",
            "graph_bundle_id": str(
                (
                    (self.manifest.get("graph") or {}).get("graph_bundle_id")
                    if isinstance(self.manifest.get("graph"), dict)
                    else self.manifest.get("graph_bundle_id")
                )
                or ""
            ),
            "trajectory_manifest_id": str(
                (
                    (self.manifest.get("trajectory") or {}).get("trajectory_manifest_id")
                    if isinstance(self.manifest.get("trajectory"), dict)
                    else self.manifest.get("trajectory_manifest_id")
                )
                or ""
            ),
            "trajectory_version": str(
                (
                    (self.manifest.get("trajectory") or {}).get("trajectory_version")
                    if isinstance(self.manifest.get("trajectory"), dict)
                    else self.manifest.get("trajectory_version")
                )
                or ""
            ),
        }
        mismatches: List[Dict[str, str]] = []
        for key, expected_value in required.items():
            actual = str(fingerprints.get(key, ""))
            if actual != str(expected_value):
                mismatches.append(
                    {
                        "key": key,
                        "expected": str(expected_value),
                        "actual": actual,
                    }
                )
        return {
            "ok": len(mismatches) == 0,
            "bundle_dir": str(self.bundle_dir),
            "bundle_id": self.bundle_id,
            "fingerprints": fingerprints,
            "required_compat": required,
            "mismatches": mismatches,
        }

    def _validate_routing_contract(
        self,
        *,
        routing: Dict[str, Any],
        requested_objective: str,
        effective_objective: str,
    ) -> Dict[str, Any]:
        objective_requested = str(
            routing.get("objective_requested") or requested_objective
        ).strip()
        objective_effective = str(
            routing.get("objective_effective") or effective_objective
        ).strip()
        if objective_effective != effective_objective:
            raise RoutingContractError(
                f"routing.objective_effective={objective_effective!r} does not match runtime effective objective={effective_objective!r}"
            )
        if objective_requested and objective_requested != requested_objective:
            raise RoutingContractError(
                f"routing.objective_requested={objective_requested!r} does not match request objective={requested_objective!r}"
            )
        track = str(routing.get("track") or "post_publication").strip().lower()
        if track not in {"pre_publication", "post_publication"}:
            raise RoutingContractError(
                "routing.track must be one of: pre_publication, post_publication."
            )
        allow_fallback = bool(routing.get("allow_fallback", True))
        required_compat_raw = (
            routing.get("required_compat")
            if isinstance(routing.get("required_compat"), dict)
            else {}
        )
        required_compat = {
            str(key): str(value)
            for key, value in required_compat_raw.items()
            if str(key).strip()
        }
        request_id = str(routing.get("request_id") or "").strip()
        if request_id:
            try:
                uuid.UUID(request_id)
            except ValueError as error:
                raise RoutingContractError(
                    "routing.request_id must be a valid UUID string."
                ) from error
        experiment_payload = (
            routing.get("experiment")
            if isinstance(routing.get("experiment"), dict)
            else {}
        )
        experiment_id = str(experiment_payload.get("id") or "").strip() or None
        experiment_variant = str(experiment_payload.get("variant") or "").strip() or None
        experiment_unit_hash = str(experiment_payload.get("unit_hash") or "").strip() or None
        if experiment_variant and experiment_variant not in {"control", "treatment"}:
            raise RoutingContractError(
                "routing.experiment.variant must be one of: control, treatment."
            )
        compatibility = self.compatibility_payload(required_compat=required_compat)
        if not bool(compatibility.get("ok", False)):
            raise ArtifactCompatibilityError(
                "incompatible_artifact: "
                + " | ".join(
                    f"{item['key']} expected={item['expected']!r} actual={item['actual']!r}"
                    for item in compatibility.get("mismatches", [])
                    if isinstance(item, dict)
                )
            )
        return {
            "objective_requested": requested_objective,
            "objective_effective": effective_objective,
            "track": track,
            "allow_fallback": allow_fallback,
            "required_compat": required_compat,
            "selected_bundle_id": self.bundle_id,
            "route_reason": "objective_route_match",
            "request_id": request_id,
            "experiment": {
                "id": experiment_id,
                "variant": experiment_variant,
                "unit_hash": experiment_unit_hash,
            },
        }

    def _load_comment_feature_index(self) -> None:
        manifest_path = self.manifest.get("comment_feature_manifest_path")
        manifest_id = self.manifest.get("comment_feature_manifest_id")
        refs = _manifest_ref_candidates(
            bundle_dir=self.bundle_dir,
            manifest_path=str(manifest_path) if manifest_path is not None else None,
            manifest_id=str(manifest_id) if manifest_id is not None else None,
        )
        for ref in refs:
            if not ref.exists():
                continue
            try:
                _, rows = load_comment_intelligence_snapshot_manifest(ref)
                self.comment_index = _CommentManifestIndex(rows)
                self.comment_index_source_path = str(ref)
                self.comment_index_load_error = None
                return
            except Exception as error:
                self.comment_index = None
                self.comment_index_source_path = None
                self.comment_index_load_error = str(error)

    def _load_trajectory_feature_index(self) -> None:
        refs: List[Path] = []
        manifest_path = self.manifest.get("trajectory_manifest_path")
        manifest_id = self.manifest.get("trajectory_manifest_id")
        if isinstance(manifest_path, str) and manifest_path.strip():
            path = Path(manifest_path.strip())
            refs.append(path / "manifest.json" if path.is_dir() else path)
        if isinstance(manifest_id, str) and manifest_id.strip():
            refs.extend(
                [
                    self.bundle_dir.parent.parent
                    / "trajectory"
                    / manifest_id.strip()
                    / "manifest.json",
                    Path("artifacts")
                    / "trajectory"
                    / manifest_id.strip()
                    / "manifest.json",
                    self.bundle_dir
                    / "trajectory"
                    / "manifest.json",
                ]
            )
        refs.append(self.bundle_dir / "trajectory" / "manifest.json")
        deduped: List[Path] = []
        seen: set[str] = set()
        for ref in refs:
            key = str(ref)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ref)
        for ref in deduped:
            if not ref.exists():
                continue
            try:
                payload = json.loads(ref.read_text(encoding="utf-8"))
                tables = payload.get("tables") if isinstance(payload.get("tables"), dict) else {}
                profiles_meta = tables.get("profiles") if isinstance(tables, dict) else {}
                rows: List[Dict[str, Any]] = []
                if isinstance(profiles_meta, dict):
                    table_path = ref.parent / str(profiles_meta.get("path") or "")
                    fmt = str(profiles_meta.get("format") or "").lower()
                    if table_path.exists():
                        if fmt == "parquet":
                            import pandas as pd  # type: ignore

                            rows = pd.read_parquet(table_path).to_dict(orient="records")
                        else:
                            for line in table_path.read_text(encoding="utf-8").splitlines():
                                if not line.strip():
                                    continue
                                try:
                                    parsed = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                if isinstance(parsed, dict):
                                    rows.append(parsed)
                if not rows:
                    profiles_path = ref.parent / "profiles.parquet"
                    if profiles_path.exists():
                        import pandas as pd  # type: ignore

                        rows = pd.read_parquet(profiles_path).to_dict(orient="records")
                if not rows:
                    continue
                self.trajectory_index = _TrajectoryManifestIndex(rows)
                self.trajectory_index_source_path = str(ref)
                self.trajectory_index_load_error = None
                return
            except Exception as error:
                self.trajectory_index = None
                self.trajectory_index_source_path = None
                self.trajectory_index_load_error = str(error)

    def _manifest_comment_for_row_id(
        self,
        *,
        row_id: str,
        as_of: datetime,
    ) -> Optional[Dict[str, Any]]:
        if self.comment_index is None:
            return None
        candidate_video_ids = [row_id]
        row_video_id = _candidate_video_id(row_id)
        if row_video_id not in candidate_video_ids:
            candidate_video_ids.append(row_video_id)
        for video_id in candidate_video_ids:
            found = self.comment_index.lookup(video_id=video_id, as_of=as_of)
            if isinstance(found, dict):
                return _coerce_manifest_comment_intelligence(found)
        return None

    def _manifest_trajectory_for_row_id(
        self,
        *,
        row_id: str,
        as_of: datetime,
    ) -> Optional[Dict[str, Any]]:
        if self.trajectory_index is None:
            return None
        candidate_video_ids = [row_id]
        row_video_id = _candidate_video_id(row_id)
        if row_video_id not in candidate_video_ids:
            candidate_video_ids.append(row_video_id)
        for video_id in candidate_video_ids:
            found = self.trajectory_index.lookup(video_id=video_id, as_of=as_of)
            if isinstance(found, dict):
                return _coerce_manifest_trajectory_features(found)
        return None

    def _resolve_comment_intelligence(
        self,
        *,
        row_id: str,
        as_of: datetime,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        from_manifest = self._manifest_comment_for_row_id(row_id=row_id, as_of=as_of)
        if from_manifest:
            return from_manifest
        return _coerce_comment_intelligence(payload)

    def _resolve_trajectory_features(
        self,
        *,
        row_id: str,
        as_of: datetime,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        from_manifest = self._manifest_trajectory_for_row_id(row_id=row_id, as_of=as_of)
        if from_manifest:
            return from_manifest
        hinted = _coerce_trajectory_features(payload)
        if hinted:
            return hinted
        return {}

    def recommend(
        self,
        objective: str,
        as_of_time: Any,
        query: Dict[str, Any],
        candidates: Sequence[Dict[str, Any]],
        top_k: int = 20,
        retrieve_k: int = 200,
        language: Optional[str] = None,
        locale: Optional[str] = None,
        content_type: Optional[str] = None,
        candidate_ids: Optional[Sequence[str]] = None,
        policy_overrides: Optional[Dict[str, Any]] = None,
        portfolio: Optional[Dict[str, Any]] = None,
        graph_controls: Optional[Dict[str, Any]] = None,
        trajectory_controls: Optional[Dict[str, Any]] = None,
        explainability: Optional[Dict[str, Any]] = None,
        routing: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        started_total = datetime.now(timezone.utc).timestamp()
        requested_objective, effective_objective = map_objective(objective)
        configured_objectives = {
            str(item) for item in self.manifest.get("objectives", [])
        }
        if effective_objective not in configured_objectives:
            raise ValueError(
                f"Objective '{effective_objective}' is not available in loaded artifact bundle."
            )
        as_of = parse_dt(as_of_time)
        if as_of is None:
            raise ValueError("as_of_time must be a valid ISO-8601 timestamp.")
        explainability_controls = ExplainabilityControls.from_payload(explainability)
        portfolio_payload = portfolio if isinstance(portfolio, dict) else {}
        portfolio_requested = bool(portfolio_payload.get("enabled", False))
        portfolio_precheck_reason: Optional[str] = None
        portfolio_weights = _resolve_portfolio_weights(portfolio_payload)
        if portfolio_requested:
            for objective_id in ("reach", "conversion"):
                if objective_id not in self.rankers:
                    portfolio_precheck_reason = f"missing_ranker_{objective_id}"
                    break
            if portfolio_precheck_reason is None:
                for objective_id in ("reach", "conversion"):
                    if objective_id not in self.calibrators:
                        portfolio_precheck_reason = f"missing_calibrator_{objective_id}"
                        break
        graph_controls_payload = graph_controls if isinstance(graph_controls, dict) else {}
        graph_branch_enabled = bool(graph_controls_payload.get("enable_graph_branch", True))
        trajectory_controls_payload = (
            trajectory_controls if isinstance(trajectory_controls, dict) else {}
        )
        trajectory_branch_enabled = bool(
            trajectory_controls_payload.get("enabled", True)
        )
        routing_payload = routing if isinstance(routing, dict) else {}
        routing_decision = self._validate_routing_contract(
            routing=routing_payload,
            requested_objective=requested_objective,
            effective_objective=effective_objective,
        )
        stage_budgets_raw = (
            routing_payload.get("stage_budgets_ms")
            if isinstance(routing_payload.get("stage_budgets_ms"), dict)
            else {}
        )
        retrieval_budget_ms = float(stage_budgets_raw.get("retrieval") or 260.0)
        ranking_budget_ms = float(stage_budgets_raw.get("ranking") or 260.0)
        explainability_budget_ms = float(stage_budgets_raw.get("explainability") or 300.0)
        latency_breakdown_ms: Dict[str, float] = {}
        query_row = _to_runtime_row(
            row_id=str(query.get("query_id") or "query"),
            payload={
                "text": query.get("text")
                or " ".join(
                    [
                        _safe_text(query.get("description")),
                        " ".join(query.get("hashtags") or []),
                        " ".join(query.get("mentions") or []),
                    ]
                ),
                "topic_key": query.get("topic_key") or query.get("objective") or "general",
                "author_id": query.get("author_id"),
                "as_of_time": query.get("as_of_time") or as_of,
                "locale": query.get("locale") or locale,
                "language": query.get("language") or language,
                "content_type": query.get("content_type") or content_type,
            },
            fallback_as_of=as_of,
        )
        query_row = _to_runtime_row_with_fabric(
            row_id=query_row["row_id"],
            payload=query_row["_source_payload"],
            fallback_as_of=as_of,
            fabric=self.fabric,
        )
        query_comment = self._resolve_comment_intelligence(
            row_id=str(query_row["row_id"]),
            as_of=as_of,
            payload=query_row["_source_payload"],
        )
        if query_comment:
            query_row["features"]["comment_intelligence"] = query_comment
            query_row["_comment_intelligence"] = query_comment
        query_trajectory = self._resolve_trajectory_features(
            row_id=str(query_row["row_id"]),
            as_of=as_of,
            payload=query_row["_source_payload"],
        )
        if query_trajectory:
            query_row["features"]["trajectory_features"] = query_trajectory
            query_row["_trajectory_features"] = query_trajectory

        candidate_rows = []
        for item in candidates:
            candidate_id = str(item.get("candidate_id") or item.get("video_id") or item.get("row_id") or "")
            if not candidate_id:
                continue
            row = _to_runtime_row_with_fabric(
                row_id=candidate_id,
                payload={
                    "text": item.get("text")
                    or " ".join(
                        [
                            _safe_text(item.get("caption")),
                            " ".join(item.get("hashtags") or []),
                            " ".join(item.get("keywords") or []),
                        ]
                    ),
                    "topic_key": item.get("topic_key") or "general",
                    "author_id": item.get("author_id"),
                    "as_of_time": item.get("as_of_time") or item.get("posted_at") or as_of,
                    "locale": item.get("locale") or locale,
                    "language": item.get("language") or language,
                    "content_type": item.get("content_type") or content_type,
                    "hints": item.get("signal_hints") if isinstance(item.get("signal_hints"), dict) else {},
                },
                fallback_as_of=as_of,
                fabric=self.fabric,
            )
            comment_payload = self._resolve_comment_intelligence(
                row_id=candidate_id,
                as_of=parse_dt(item.get("as_of_time")) or parse_dt(item.get("posted_at")) or as_of,
                payload=row["_source_payload"],
            )
            if comment_payload:
                row["features"]["comment_intelligence"] = comment_payload
                row["_comment_intelligence"] = comment_payload
            trajectory_payload = self._resolve_trajectory_features(
                row_id=candidate_id,
                as_of=parse_dt(item.get("as_of_time")) or parse_dt(item.get("posted_at")) or as_of,
                payload=row["_source_payload"],
            )
            if trajectory_payload:
                row["features"]["trajectory_features"] = trajectory_payload
                row["_trajectory_features"] = trajectory_payload
            candidate_rows.append(row)

        retrieval_constraints = {
            "max_age_days": int(self.manifest.get("max_age_days") or 180),
            "language": _derive_language(language or query.get("language"), locale or query.get("locale")),
            "locale": _normalize_locale(locale or query.get("locale")),
            "content_type": (_safe_text(content_type or query.get("content_type")) or None),
        }
        candidate_scope = [
            str(item).strip()
            for item in list(candidate_ids or [])
            if isinstance(item, str) and str(item).strip()
        ]
        weight_override: Optional[Dict[str, float]] = None
        graph_fallback_mode = "active"
        trajectory_mode = "active"
        override = self.retriever.branch_weights(effective_objective)
        if not graph_branch_enabled:
            override["graph_dense"] = 0.0
            graph_fallback_mode = "graph_branch_disabled"
        elif not self.retriever.graph_bundle_id:
            graph_fallback_mode = "graph_bundle_unavailable"
            override["graph_dense"] = 0.0
        if not trajectory_branch_enabled:
            override["trajectory_dense"] = 0.0
            trajectory_mode = "trajectory_branch_disabled"
        elif not self.retriever.trajectory_manifest_id:
            override["trajectory_dense"] = 0.0
            trajectory_mode = "trajectory_artifact_unavailable"
        else:
            trajectory_mode = "active"
        weight_override = override
        started_retrieval = datetime.now(timezone.utc).timestamp()
        retrieved_raw = self.retriever.retrieve(
            query_row=query_row,
            candidate_rows=candidate_rows if candidate_scope else None,
            top_k=max(top_k, retrieve_k),
            index_cutoff_time=as_of,
            objective=effective_objective,
            candidate_ids=candidate_scope,
            retrieval_constraints=retrieval_constraints,
            weight_override=weight_override,
            return_metadata=True,
        )
        retrieved, retrieval_metadata = retrieved_raw
        retrieval_elapsed_ms = (datetime.now(timezone.utc).timestamp() - started_retrieval) * 1000.0
        latency_breakdown_ms["retrieval"] = round(retrieval_elapsed_ms, 6)
        if retrieval_elapsed_ms > retrieval_budget_ms:
            raise RecommenderStageTimeoutError(
                stage="retrieval",
                elapsed_ms=retrieval_elapsed_ms,
                budget_ms=retrieval_budget_ms,
            )

        ranker = self.rankers.get(effective_objective)
        fallback_mode = ranker is None
        ranked_items = []
        candidate_by_id: Dict[str, Dict[str, Any]] = {}
        for row in candidate_rows:
            candidate_by_id[str(row.get("row_id"))] = row
            candidate_by_id[str(row.get("video_id") or row.get("row_id"))] = row

        def _index_candidate_row(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            feature_row_id = str(item.get("feature_row_id") or item.get("candidate_row_id") or "")
            if not feature_row_id:
                return None
            metadata = self.retriever.row_metadata.get(feature_row_id, {})
            text = str(metadata.get("text") or metadata.get("topic_key") or "").strip()
            if not text:
                text = str(metadata.get("topic_key") or "general")
            as_of_hint = metadata.get("as_of_time") or as_of
            payload = {
                "text": text,
                "topic_key": metadata.get("topic_key") or "general",
                "author_id": metadata.get("author_id"),
                "as_of_time": as_of_hint,
                "language": metadata.get("language"),
                "locale": metadata.get("locale"),
                "content_type": metadata.get("content_type"),
            }
            candidate_key = str(item.get("candidate_id") or metadata.get("video_id") or feature_row_id)
            synthesized = _to_runtime_row_with_fabric(
                row_id=candidate_key,
                payload=payload,
                fallback_as_of=as_of,
                fabric=self.fabric,
            )
            comment_payload = self._resolve_comment_intelligence(
                row_id=candidate_key,
                as_of=parse_dt(as_of_hint) or as_of,
                payload=synthesized.get("_source_payload", {}),
            )
            if comment_payload:
                synthesized["features"]["comment_intelligence"] = comment_payload
                synthesized["_comment_intelligence"] = comment_payload
            trajectory_payload = self._resolve_trajectory_features(
                row_id=candidate_key,
                as_of=parse_dt(as_of_hint) or as_of,
                payload=synthesized.get("_source_payload", {}),
            )
            if trajectory_payload:
                synthesized["features"]["trajectory_features"] = trajectory_payload
                synthesized["_trajectory_features"] = trajectory_payload
            return synthesized

        def _score_candidate_for_objective(
            *,
            objective_id: str,
            candidate_row: Dict[str, Any],
            similarity: float,
        ) -> Tuple[Optional[float], Optional[float], Optional[str], float]:
            ranker_obj = self.rankers.get(objective_id)
            if ranker_obj is None:
                return None, None, f"missing_ranker_{objective_id}", 0.0
            ranker_out = ranker_obj.score_pair(
                query_row=query_row,
                candidate_row=candidate_row,
                similarity=similarity,
            )
            raw_score = float(ranker_out.get("final_score") or 0.0)
            score_std = float(ranker_out.get("score_std") or 0.0)
            calibrator_obj = self.calibrators.get(objective_id)
            if calibrator_obj is None:
                return None, raw_score, f"missing_calibrator_{objective_id}", score_std
            calibration = calibrator_obj.calibrate(
                score_raw=raw_score,
                segment_id=str(ranker_out.get("selected_ranker_id") or "global"),
                include_debug=False,
            )
            return (
                float(calibration.get("score_calibrated") or raw_score),
                raw_score,
                None,
                score_std,
            )

        started_ranking = datetime.now(timezone.utc).timestamp()
        for item in retrieved:
            candidate_key = str(item.get("candidate_id") or "")
            candidate_row = candidate_by_id.get(candidate_key) or candidate_by_id.get(
                str(item.get("candidate_row_id") or "")
            )
            if candidate_row is None:
                candidate_row = _index_candidate_row(item)
                if candidate_row is not None:
                    candidate_by_id[str(candidate_row.get("row_id"))] = candidate_row
                    candidate_by_id[str(candidate_row.get("video_id") or candidate_row.get("row_id"))] = candidate_row
            if candidate_row is None:
                continue
            candidate_features = candidate_row.get("features")
            if isinstance(candidate_features, dict):
                candidate_features["graph_similarity_hint"] = float(
                    ((item.get("retrieval_branch_scores") or {}).get("graph_dense") or 0.0)
                )
                candidate_features["trajectory_similarity_hint"] = float(
                    (
                        (item.get("retrieval_branch_scores") or {}).get(
                            "trajectory_dense"
                        )
                        or 0.0
                    )
                )
            candidate_row["_graph_similarity"] = float(
                ((item.get("retrieval_branch_scores") or {}).get("graph_dense") or 0.0)
            )
            candidate_row["_trajectory_similarity"] = float(
                (
                    (item.get("retrieval_branch_scores") or {}).get("trajectory_dense")
                    or 0.0
                )
            )
            if ranker is None:
                final_score = float(item["fused_score"])
                ranker_score = {
                    "final_score": final_score,
                    "score_mean": final_score,
                    "score_std": 0.0,
                    "confidence": 0.0,
                    "selected_ranker_id": "global",
                    "global_score_mean": final_score,
                    "segment_blend_weight": 0.0,
                }
                ranker_backend = None
            else:
                ranker_score = ranker.score_pair(
                    query_row=query_row,
                    candidate_row=candidate_row,
                    similarity=float(item.get("fused_score") or 0.0),
                )
                final_score = float(ranker_score["final_score"])
                ranker_backend = (
                    ranker.global_ensemble.models[0].backend
                    if ranker.global_ensemble.models
                    else None
                )
            score_raw = float(ranker_score.get("final_score") or final_score)
            calibrator = self.calibrators.get(effective_objective)
            if calibrator is not None:
                calibration = calibrator.calibrate(
                    score_raw=score_raw,
                    segment_id=str(ranker_score.get("selected_ranker_id") or "global"),
                    include_debug=True,
                )
                calibration_warning = None
            else:
                calibration_warning = self.calibration_load_warnings.get(effective_objective)
                if not calibration_warning:
                    calibration_warning = "calibrator_unavailable_identity_fallback"
                calibration = {
                    "score_raw": score_raw,
                    "score_calibrated": score_raw,
                    "calibrator_segment_id": "global",
                    "requested_segment_id": str(ranker_score.get("selected_ranker_id") or "global"),
                    "calibrator_method": "identity",
                    "calibrator_support_count": 0,
                    "calibration_fallback_used": True,
                    "target_definition": "p_relevance_ge_2",
                    "objective": effective_objective,
                    "version": CALIBRATION_VERSION,
                    "warning": calibration_warning,
                }
            comment_trace = candidate_row.get("_comment_intelligence")
            portfolio_components: Optional[Dict[str, Any]] = None
            if portfolio_requested and portfolio_precheck_reason is None:
                similarity_fused = float(item.get("fused_score") or 0.0)
                reach_score: Optional[float]
                conversion_score: Optional[float]
                uncertainty_raw = float(ranker_score.get("score_std") or 0.0)
                portfolio_component_reason: Optional[str] = None
                if effective_objective == "reach":
                    reach_score = float(calibration["score_calibrated"])
                else:
                    reach_score, _, reason, reach_std = _score_candidate_for_objective(
                        objective_id="reach",
                        candidate_row=candidate_row,
                        similarity=similarity_fused,
                    )
                    uncertainty_raw = max(uncertainty_raw, float(reach_std))
                    if reason:
                        portfolio_component_reason = reason
                if effective_objective == "conversion":
                    conversion_score = float(calibration["score_calibrated"])
                else:
                    (
                        conversion_score,
                        _,
                        reason,
                        conversion_std,
                    ) = _score_candidate_for_objective(
                        objective_id="conversion",
                        candidate_row=candidate_row,
                        similarity=similarity_fused,
                    )
                    uncertainty_raw = max(uncertainty_raw, float(conversion_std))
                    if portfolio_component_reason is None and reason:
                        portfolio_component_reason = reason
                traj_payload = (
                    (candidate_row.get("features") or {}).get("trajectory_features")
                    if isinstance(candidate_row.get("features"), dict)
                    else {}
                )
                if not isinstance(traj_payload, dict):
                    traj_payload = {}
                regime_probs = (
                    traj_payload.get("regime_probabilities")
                    if isinstance(traj_payload.get("regime_probabilities"), dict)
                    else {}
                )
                durability_ratio = _as_float(traj_payload.get("durability_ratio"), 0.0)
                stability = _as_float(traj_payload.get("stability"), 0.0)
                durable_prob = _as_float(regime_probs.get("durable"), 0.0)
                durability_raw = _clamp(
                    (0.60 * durability_ratio)
                    + (0.25 * max(0.0, stability))
                    + (0.15 * durable_prob),
                    0.0,
                    2.0,
                )
                if reach_score is None or conversion_score is None:
                    if portfolio_component_reason is None:
                        portfolio_component_reason = "missing_portfolio_components"
                else:
                    portfolio_components = {
                        "reach_score": float(reach_score),
                        "conversion_score": float(conversion_score),
                        "durability_score_raw": float(durability_raw),
                        "uncertainty_penalty_raw": float(max(0.0, uncertainty_raw)),
                        "base_portfolio_utility": 0.0,
                    }
                if portfolio_component_reason:
                    portfolio_components = None
            ranked_items.append(
                {
                    "candidate_id": candidate_key or str(candidate_row["row_id"]),
                    "candidate_row_id": candidate_row["row_id"],
                    "score": float(calibration["score_calibrated"]),
                    "score_raw": float(calibration["score_raw"]),
                    "score_calibrated": float(calibration["score_calibrated"]),
                    "score_mean": float(ranker_score.get("score_mean") or final_score),
                    "score_std": float(ranker_score.get("score_std") or 0.0),
                    "confidence": float(ranker_score.get("confidence") or 0.0),
                    "selected_ranker_id": str(ranker_score.get("selected_ranker_id") or "global"),
                    "global_score_mean": float(ranker_score.get("global_score_mean") or final_score),
                    "segment_blend_weight": float(ranker_score.get("segment_blend_weight") or 0.0),
                    "trajectory_score": float(ranker_score.get("trajectory_score") or 0.0),
                    "trajectory_similarity": float(
                        ranker_score.get("trajectory_similarity") or 0.0
                    ),
                    "trajectory_regime_pred": str(
                        ranker_score.get("trajectory_regime_pred") or "balanced"
                    ),
                    "trajectory_regime_probabilities": dict(
                        ranker_score.get("trajectory_regime_probabilities")
                        if isinstance(
                            ranker_score.get("trajectory_regime_probabilities"),
                            dict,
                        )
                        else {}
                    ),
                    "trajectory_regime_confidence": float(
                        ranker_score.get("trajectory_regime_confidence") or 0.0
                    ),
                    "trajectory_available": bool(
                        ranker_score.get("trajectory_available", False)
                    ),
                    "similarity": {
                        "sparse": float(item.get("sparse_score") or 0.0),
                        "dense": float(item.get("dense_score") or 0.0),
                        "fused": float(item.get("fused_score") or 0.0),
                    },
                    "retrieval_branch_scores": dict(
                        item.get("retrieval_branch_scores")
                        if isinstance(item.get("retrieval_branch_scores"), dict)
                        else {
                            "lexical": float(item.get("lexical_score") or 0.0),
                            "dense_text": float(item.get("dense_text_score") or 0.0),
                            "multimodal": float(item.get("multimodal_score") or 0.0),
                            "graph_dense": float(item.get("graph_dense_score") or 0.0),
                            "trajectory_dense": float(
                                item.get("trajectory_dense_score") or 0.0
                            ),
                            "fused": float(item.get("fused_score") or 0.0),
                        }
                    ),
                    "trace": {
                        "objective_model": effective_objective if ranker else None,
                        "ranker_backend": ranker_backend,
                        "calibration": calibration,
                    },
                    "calibration_trace": calibration,
                    "comment_trace": comment_trace if isinstance(comment_trace, dict) and comment_trace else None,
                    "portfolio_components": portfolio_components,
                    "portfolio_trace": None,
                    "graph_trace": (
                        item.get("graph_trace")
                        if isinstance(item.get("graph_trace"), dict)
                        else None
                    ),
                    "trajectory_trace": {
                        "trajectory_manifest_id": self.retriever.trajectory_manifest_id,
                        "trajectory_version": self.retriever.trajectory_version,
                        "trajectory_mode": trajectory_mode,
                        "similarity": float(
                            (
                                (item.get("retrieval_branch_scores") or {}).get(
                                    "trajectory_dense"
                                )
                                or 0.0
                            )
                        ),
                        "regime_pred": str(
                            ranker_score.get("trajectory_regime_pred") or "balanced"
                        ),
                        "regime_probabilities": dict(
                            ranker_score.get("trajectory_regime_probabilities")
                            if isinstance(
                                ranker_score.get("trajectory_regime_probabilities"),
                                dict,
                            )
                            else {}
                        ),
                        "regime_confidence": float(
                            ranker_score.get("trajectory_regime_confidence") or 0.0
                        ),
                        "available": bool(ranker_score.get("trajectory_available", False)),
                    },
                    "_author_id": str(candidate_row.get("author_id") or "unknown"),
                    "_topic_key": str(candidate_row.get("topic_key") or "unknown"),
                    "_language": str((candidate_row.get("features") or {}).get("language") or ""),
                    "_locale": str(candidate_row.get("locale") or ""),
                    "_candidate_as_of_time": candidate_row.get("as_of_time"),
                }
            )
        ranked_items.sort(key=lambda item: float(item.get("score_calibrated") or 0.0), reverse=True)
        if portfolio_requested and portfolio_precheck_reason is None:
            valid_portfolio_items = [
                item
                for item in ranked_items
                if isinstance(item.get("portfolio_components"), dict)
            ]
            if not valid_portfolio_items:
                portfolio_precheck_reason = "portfolio_components_unavailable"
            else:
                durability_values = [
                    _as_float(
                        (item.get("portfolio_components") or {}).get("durability_score_raw"), 0.0
                    )
                    for item in valid_portfolio_items
                ]
                uncertainty_values = [
                    _as_float(
                        (item.get("portfolio_components") or {}).get(
                            "uncertainty_penalty_raw"
                        ),
                        0.0,
                    )
                    for item in valid_portfolio_items
                ]
                dur_min = min(durability_values) if durability_values else 0.0
                dur_max = max(durability_values) if durability_values else 1.0
                unc_min = min(uncertainty_values) if uncertainty_values else 0.0
                unc_max = max(uncertainty_values) if uncertainty_values else 1.0
                for item in valid_portfolio_items:
                    payload = (
                        item.get("portfolio_components")
                        if isinstance(item.get("portfolio_components"), dict)
                        else {}
                    )
                    reach_score = _clamp(_as_float(payload.get("reach_score"), 0.0), 0.0, 1.0)
                    conversion_score = _clamp(
                        _as_float(payload.get("conversion_score"), 0.0), 0.0, 1.0
                    )
                    durability_score = _clamp(
                        _minmax_scale(
                            _as_float(payload.get("durability_score_raw"), 0.0),
                            dur_min,
                            dur_max,
                        ),
                        0.0,
                        1.0,
                    )
                    uncertainty_penalty = _clamp(
                        _minmax_scale(
                            _as_float(payload.get("uncertainty_penalty_raw"), 0.0),
                            unc_min,
                            unc_max,
                        ),
                        0.0,
                        1.0,
                    )
                    base_utility = (
                        portfolio_weights["reach"] * reach_score
                        + portfolio_weights["conversion"] * conversion_score
                        + portfolio_weights["durability"] * durability_score
                        - (
                            _as_float(portfolio_payload.get("risk_aversion"), 0.10)
                            * uncertainty_penalty
                        )
                    )
                    item["portfolio_components"] = {
                        "reach_score": round(reach_score, 6),
                        "conversion_score": round(conversion_score, 6),
                        "durability_score": round(durability_score, 6),
                        "uncertainty_penalty": round(uncertainty_penalty, 6),
                        "base_portfolio_utility": round(float(base_utility), 6),
                    }
        portfolio_controls_for_policy: Dict[str, Any] = {
            **(portfolio_payload if isinstance(portfolio_payload, dict) else {}),
            "enabled": bool(portfolio_requested and portfolio_precheck_reason is None),
        }
        if portfolio_precheck_reason:
            portfolio_controls_for_policy["fallback_reason"] = portfolio_precheck_reason
        top_items, policy_metadata = self.policy_reranker.rerank(
            ranked_items=ranked_items,
            query_context={
                "as_of_time": as_of,
                "language": retrieval_constraints.get("language"),
                "locale": retrieval_constraints.get("locale"),
            },
            top_k=max(1, top_k),
            overrides=policy_overrides,
            portfolio=portfolio_controls_for_policy,
        )
        portfolio_metadata = (
            policy_metadata.get("portfolio_metadata")
            if isinstance(policy_metadata.get("portfolio_metadata"), dict)
            else {}
        )
        if portfolio_requested and portfolio_precheck_reason:
            portfolio_metadata = {
                **portfolio_metadata,
                "enabled_requested": True,
                "enabled": False,
                "fallback_reason": portfolio_precheck_reason,
            }
            policy_metadata["portfolio_mode"] = False
            policy_metadata["portfolio_metadata"] = portfolio_metadata
        for item in top_items:
            adjusted = float(item.get("policy_adjusted_score") or item.get("score_calibrated") or item.get("score") or 0.0)
            item["score"] = adjusted
        ranking_elapsed_ms = (datetime.now(timezone.utc).timestamp() - started_ranking) * 1000.0
        latency_breakdown_ms["ranking"] = round(ranking_elapsed_ms, 6)
        if ranking_elapsed_ms > ranking_budget_ms:
            raise RecommenderStageTimeoutError(
                stage="ranking",
                elapsed_ms=ranking_elapsed_ms,
                budget_ms=ranking_budget_ms,
            )

        score_values = np.asarray(
            [float(item.get("score_calibrated") or item.get("score") or 0.0) for item in ranked_items],
            dtype=np.float64,
        )
        score_spread = float(np.ptp(score_values)) if score_values.size > 0 else 1.0
        if score_spread <= 1e-6:
            score_spread = 1.0

        explainability_cards: Dict[str, Dict[str, Any]] = {}
        counterfactual_map: Dict[str, List[Dict[str, Any]]] = {}
        explainability_methods: List[str] = []
        explainability_fallback_counts: Dict[str, int] = {
            "contribution_fallback": 0,
            "missing_candidate_row": 0,
            "counterfactual_not_applicable": 0,
        }
        explainability_timeout = False
        if explainability_controls.enabled:
            started_explainability = datetime.now(timezone.utc).timestamp()
            neighbor_card = neighbor_evidence_card(
                query_row=query_row,
                ranked_items=ranked_items,
                candidate_by_id=candidate_by_id,
                query_as_of=as_of,
                neighbor_k=explainability_controls.neighbor_k,
                top_features=explainability_controls.top_features,
            )
            explainability_methods.append("neighbor_quartile")
            for item in top_items:
                candidate_key = str(item.get("candidate_id") or item.get("candidate_row_id") or "")
                candidate_row = candidate_by_id.get(candidate_key) or candidate_by_id.get(
                    str(item.get("candidate_row_id") or "")
                )
                if not isinstance(candidate_row, dict) or ranker is None:
                    explainability_fallback_counts["missing_candidate_row"] = (
                        explainability_fallback_counts.get("missing_candidate_row", 0) + 1
                    )
                    band = temporal_confidence_band(
                        score_std=float(item.get("score_std") or 0.0),
                        score_raw=float(item.get("score_raw") or item.get("score") or 0.0),
                        score_calibrated=float(item.get("score_calibrated") or item.get("score") or 0.0),
                        rank_count=max(1, len(ranked_items)),
                        score_spread=score_spread,
                    )
                    explainability_cards[candidate_key] = {
                        "feature_contributions": {
                            "method": "unavailable",
                            "fallback_used": True,
                            "input_trace_id": f"trace::{candidate_key}",
                            "top_positive": [],
                            "top_negative": [],
                        },
                        "neighbor_evidence": explainability_as_payload(neighbor_card),
                        "temporal_confidence_band": explainability_as_payload(band),
                        "graph_evidence": (
                            item.get("graph_trace")
                            if isinstance(item.get("graph_trace"), dict)
                            else None
                        ),
                    }
                    continue
                contribution = feature_contribution_card(
                    ranker=ranker,
                    query_row=query_row,
                    candidate_row=candidate_row,
                    similarity=float((item.get("similarity") or {}).get("fused") or 0.0),
                    selected_ranker_id=str(item.get("selected_ranker_id") or "global"),
                    segment_blend_weight=float(item.get("segment_blend_weight") or 0.0),
                    top_features=explainability_controls.top_features,
                    input_trace_id=f"trace::{candidate_key}",
                )
                if contribution.fallback_used:
                    explainability_fallback_counts["contribution_fallback"] = (
                        explainability_fallback_counts.get("contribution_fallback", 0) + 1
                    )
                explainability_methods.append(contribution.method)
                band = temporal_confidence_band(
                    score_std=float(item.get("score_std") or 0.0),
                    score_raw=float(item.get("score_raw") or item.get("score") or 0.0),
                    score_calibrated=float(item.get("score_calibrated") or item.get("score") or 0.0),
                    rank_count=max(1, len(ranked_items)),
                    score_spread=score_spread,
                )
                explainability_cards[candidate_key] = {
                    "feature_contributions": explainability_as_payload(contribution),
                    "neighbor_evidence": explainability_as_payload(neighbor_card),
                    "temporal_confidence_band": explainability_as_payload(band),
                    "graph_evidence": (
                        item.get("graph_trace")
                        if isinstance(item.get("graph_trace"), dict)
                        else None
                    ),
                }

            if explainability_controls.run_counterfactuals and ranker is not None:
                calibrator = self.calibrators.get(effective_objective)
                scenario_map = counterfactual_scenarios_for_items(
                    objective=effective_objective,
                    ranker=ranker,
                    calibrator=calibrator,
                    query_row=query_row,
                    top_items=top_items,
                    ranked_items=ranked_items,
                    candidate_by_id=candidate_by_id,
                    score_spread=score_spread,
                )
                explainability_methods.append("counterfactual_library")
                for candidate_key, values in scenario_map.items():
                    serialized = explainability_as_payload(values)
                    counterfactual_map[candidate_key] = serialized if isinstance(serialized, list) else []
                    explainability_fallback_counts["counterfactual_not_applicable"] = (
                        explainability_fallback_counts.get("counterfactual_not_applicable", 0)
                        + sum(
                            1
                            for item in counterfactual_map[candidate_key]
                            if str(item.get("feasibility")) == "not_applicable"
                        )
                    )
            explainability_elapsed_ms = (
                datetime.now(timezone.utc).timestamp() - started_explainability
            ) * 1000.0
            latency_breakdown_ms["explainability"] = round(explainability_elapsed_ms, 6)
            if explainability_elapsed_ms > explainability_budget_ms:
                explainability_timeout = True
                explainability_cards = {}
                counterfactual_map = {}
                explainability_methods = ["explainability_timeout"]
                explainability_fallback_counts["timeout"] = 1

        calibration_fallback_total = sum(
            1 for item in top_items if bool((item.get("calibration_trace") or {}).get("calibration_fallback_used"))
        )
        calibration_metadata = {
            "objective": effective_objective,
            "target_definition": "p_relevance_ge_2",
            "fallback_count": calibration_fallback_total,
            "fallback_rate": round(calibration_fallback_total / max(1, len(top_items)), 6),
            "load_warning": self.calibration_load_warnings.get(effective_objective),
            "calibrator_available": effective_objective in self.calibrators,
        }
        query_trajectory_payload = (
            (query_row.get("features") or {}).get("trajectory_features")
            if isinstance(query_row.get("features"), dict)
            else {}
        )
        if not isinstance(query_trajectory_payload, dict):
            query_trajectory_payload = {}
        query_regime_probs = query_trajectory_payload.get("regime_probabilities")
        if not isinstance(query_regime_probs, dict):
            query_regime_probs = {}
        query_objectives = (
            query_trajectory_payload.get("objectives")
            if isinstance(query_trajectory_payload.get("objectives"), dict)
            else {}
        )
        objective_payload = (
            query_objectives.get(effective_objective)
            if isinstance(query_objectives, dict)
            else {}
        )
        objective_score = (
            float(objective_payload.get("composite_z") or 0.0)
            if isinstance(objective_payload, dict)
            else 0.0
        )
        query_trajectory_prediction = {
            "trajectory_score": round(float(objective_score), 6),
            "regime_pred": str(query_trajectory_payload.get("regime_pred") or "balanced"),
            "regime_probabilities": {
                "spike": round(float(query_regime_probs.get("spike") or 0.0), 6),
                "balanced": round(float(query_regime_probs.get("balanced") or 0.0), 6),
                "durable": round(float(query_regime_probs.get("durable") or 0.0), 6),
            },
            "regime_confidence": round(
                float(query_trajectory_payload.get("regime_confidence") or 0.0),
                6,
            ),
            "available": bool(query_trajectory_payload),
        }

        compatibility_status = self.compatibility_payload(
            required_compat=routing_decision.get("required_compat")
            if isinstance(routing_decision.get("required_compat"), dict)
            else None
        )
        fallback_reason: Optional[str] = None
        if fallback_mode:
            fallback_reason = "ranker_unavailable"
        if explainability_timeout:
            fallback_reason = "explainability_timeout"
        latency_breakdown_ms["total"] = round(
            (datetime.now(timezone.utc).timestamp() - started_total) * 1000.0, 6
        )
        response: Dict[str, Any] = {
            "request_id": routing_decision.get("request_id"),
            "experiment_id": (
                (routing_decision.get("experiment") or {}).get("id")
                if isinstance(routing_decision.get("experiment"), dict)
                else None
            ),
            "variant": (
                (routing_decision.get("experiment") or {}).get("variant")
                if isinstance(routing_decision.get("experiment"), dict)
                else None
            ),
            "objective": requested_objective,
            "objective_effective": effective_objective,
            "generated_at": _to_utc_iso(datetime.now(timezone.utc)),
            "fallback_mode": bool(fallback_mode or explainability_timeout),
            "fallback_reason": fallback_reason,
            "calibration_version": CALIBRATION_VERSION,
            "policy_version": POLICY_RERANK_VERSION,
            "fabric_version": FABRIC_VERSION,
            "feature_manifest_id": self.manifest.get("feature_manifest_id"),
            "comment_feature_manifest_id": self.manifest.get("comment_feature_manifest_id"),
            "comment_intelligence_version": self.manifest.get(
                "comment_intelligence_version", "comment_intelligence.v2"
            ),
            "retrieval_mode": retrieval_metadata.get("retrieval_mode"),
            "constraint_tier_used": retrieval_metadata.get("constraint_tier_used"),
            "retriever_artifact_version": retrieval_metadata.get("retriever_artifact_version"),
            "graph_bundle_id": retrieval_metadata.get("graph_bundle_id") or self.retriever.graph_bundle_id,
            "graph_version": retrieval_metadata.get("graph_version") or self.retriever.graph_version,
            "graph_coverage": round(
                (
                    float((retrieval_metadata.get("branch_coverage") or {}).get("graph_dense") or 0.0)
                    / max(1, len(retrieved))
                ),
                6,
            ),
            "graph_fallback_mode": graph_fallback_mode,
            "trajectory_manifest_id": retrieval_metadata.get("trajectory_manifest_id")
            or self.retriever.trajectory_manifest_id,
            "trajectory_version": retrieval_metadata.get("trajectory_version")
            or self.retriever.trajectory_version,
            "trajectory_mode": trajectory_mode,
            "trajectory_prediction": query_trajectory_prediction,
            "portfolio_mode": bool(policy_metadata.get("portfolio_mode", False)),
            "portfolio_metadata": (
                policy_metadata.get("portfolio_metadata")
                if isinstance(policy_metadata.get("portfolio_metadata"), dict)
                else {
                    "enabled_requested": bool(portfolio_requested),
                    "enabled": False,
                    "fallback_reason": portfolio_precheck_reason,
                }
            ),
            "policy_metadata": policy_metadata,
            "calibration_metadata": calibration_metadata,
            "routing_decision": routing_decision,
            "compatibility_status": compatibility_status,
            "latency_breakdown_ms": latency_breakdown_ms,
            "extraction_trace_ids": [
                *(
                    (query_row.get("_fabric_output") or {}).get("trace_ids")
                    if isinstance(query_row.get("_fabric_output"), dict)
                    else []
                ),
                *[
                    trace_id
                    for row in candidate_rows
                    for trace_id in (
                        (row.get("_fabric_output") or {}).get("trace_ids")
                        if isinstance(row.get("_fabric_output"), dict)
                        else []
                    )
                ][:10],
            ],
            "items": [],
        }
        if explainability_controls.enabled:
            response["explainability_metadata"] = explainability_metadata_payload(
                objective=effective_objective,
                methods=explainability_methods,
                fallback_counts=explainability_fallback_counts,
            )
        for idx, item in enumerate(top_items):
            candidate_key = str(item.get("candidate_id") or item.get("candidate_row_id") or "")
            item_payload: Dict[str, Any] = {
                "candidate_id": item["candidate_id"],
                "rank": idx + 1,
                "score": round(item["score"], 6),
                "score_raw": round(float(item.get("score_raw") or item["score"]), 6),
                "score_calibrated": round(
                    float(item.get("score_calibrated") or item["score"]),
                    6,
                ),
                "score_mean": round(float(item.get("score_mean") or item["score"]), 6),
                "score_std": round(float(item.get("score_std") or 0.0), 6),
                "confidence": round(float(item.get("confidence") or 0.0), 6),
                "selected_ranker_id": str(item.get("selected_ranker_id") or "global"),
                "global_score_mean": round(
                    float(item.get("global_score_mean") or item["score"]),
                    6,
                ),
                "segment_blend_weight": round(
                    float(item.get("segment_blend_weight") or 0.0),
                    6,
                ),
                "policy_penalty": round(float(item.get("policy_penalty") or 0.0), 6),
                "policy_bonus": round(float(item.get("policy_bonus") or 0.0), 6),
                "policy_adjusted_score": round(
                    float(item.get("policy_adjusted_score") or item["score"]),
                    6,
                ),
                "calibration_trace": item.get("calibration_trace"),
                "policy_trace": item.get("policy_trace"),
                "portfolio_trace": item.get("portfolio_trace"),
                "similarity": {
                    "sparse": round(item["similarity"]["sparse"], 6),
                    "dense": round(item["similarity"]["dense"], 6),
                    "fused": round(item["similarity"]["fused"], 6),
                },
                "retrieval_branch_scores": {
                    "lexical": round(
                        float(
                            (item.get("retrieval_branch_scores") or {}).get("lexical")
                            or item["similarity"]["sparse"]
                        ),
                        6,
                    ),
                    "dense_text": round(
                        float(
                            (item.get("retrieval_branch_scores") or {}).get("dense_text")
                            or item["similarity"]["dense"]
                        ),
                        6,
                    ),
                    "multimodal": round(
                        float(
                            (item.get("retrieval_branch_scores") or {}).get("multimodal")
                            or 0.0
                        ),
                        6,
                    ),
                    "graph_dense": round(
                        float(
                            (item.get("retrieval_branch_scores") or {}).get("graph_dense")
                            or 0.0
                        ),
                        6,
                    ),
                    "trajectory_dense": round(
                        float(
                            (item.get("retrieval_branch_scores") or {}).get(
                                "trajectory_dense"
                            )
                            or 0.0
                        ),
                        6,
                    ),
                    "fused": round(
                        float(
                            (item.get("retrieval_branch_scores") or {}).get("fused")
                            or item["similarity"]["fused"]
                        ),
                        6,
                    ),
                },
                "trace": item["trace"],
                "comment_trace": item.get("comment_trace"),
                "graph_trace": item.get("graph_trace"),
                "trajectory_trace": item.get("trajectory_trace"),
                "trajectory_score": round(float(item.get("trajectory_score") or 0.0), 6),
                "trajectory_similarity": round(
                    float(item.get("trajectory_similarity") or 0.0),
                    6,
                ),
                "trajectory_regime_pred": str(
                    item.get("trajectory_regime_pred") or "balanced"
                ),
                "trajectory_regime_probabilities": dict(
                    item.get("trajectory_regime_probabilities")
                    if isinstance(item.get("trajectory_regime_probabilities"), dict)
                    else {}
                ),
                "trajectory_regime_confidence": round(
                    float(item.get("trajectory_regime_confidence") or 0.0),
                    6,
                ),
            }
            if explainability_controls.enabled:
                card = explainability_cards.get(candidate_key)
                if card is None:
                    band = temporal_confidence_band(
                        score_std=float(item.get("score_std") or 0.0),
                        score_raw=float(item.get("score_raw") or item.get("score") or 0.0),
                        score_calibrated=float(item.get("score_calibrated") or item.get("score") or 0.0),
                        rank_count=max(1, len(ranked_items)),
                        score_spread=score_spread,
                    )
                    card = {
                        "feature_contributions": {
                            "method": "unavailable",
                            "fallback_used": True,
                            "input_trace_id": f"trace::{candidate_key}",
                            "top_positive": [],
                            "top_negative": [],
                        },
                        "neighbor_evidence": {
                            "method": "quartile_similarity",
                            "temporal_safe": True,
                            "winner_count": 0,
                            "loser_count": 0,
                            "winners": [],
                            "losers": [],
                        },
                        "temporal_confidence_band": explainability_as_payload(band),
                        "graph_evidence": (
                            item.get("graph_trace")
                            if isinstance(item.get("graph_trace"), dict)
                            else None
                        ),
                    }
                item_payload["evidence_cards"] = card
                item_payload["temporal_confidence_band"] = card.get("temporal_confidence_band")
                item_payload["counterfactual_scenarios"] = counterfactual_map.get(candidate_key, [])
            response["items"].append(item_payload)
        if debug:
            selected_map: Dict[str, Dict[str, Any]] = {}
            for selected_rank, item in enumerate(top_items):
                candidate_key = str(item.get("candidate_id") or "")
                if not candidate_key:
                    continue
                selected_map[candidate_key] = {"item": item, "rank": selected_rank + 1}
            ranking_universe: List[Dict[str, Any]] = []
            for rank_idx, item in enumerate(ranked_items):
                candidate_id = str(item.get("candidate_id") or "")
                if not candidate_id:
                    continue
                selected_entry = selected_map.get(candidate_id)
                selected_item = (
                    selected_entry.get("item")
                    if isinstance(selected_entry, dict)
                    and isinstance(selected_entry.get("item"), dict)
                    else None
                )
                ranking_universe.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate_row_id": item.get("candidate_row_id"),
                        "retrieved_rank": rank_idx + 1,
                        "final_rank": (
                            int(selected_entry.get("rank", 0))
                            if isinstance(selected_entry, dict)
                            and selected_entry.get("rank") is not None
                            else None
                        ),
                        "selected": bool(selected_item),
                        "score_raw": float(item.get("score_raw") or 0.0),
                        "score_calibrated": float(item.get("score_calibrated") or 0.0),
                        "policy_adjusted_score": (
                            float(
                                selected_item.get("policy_adjusted_score")
                                or selected_item.get("score")
                                or 0.0
                            )
                            if isinstance(selected_item, dict)
                            else None
                        ),
                        "retrieval_branch_scores": dict(
                            item.get("retrieval_branch_scores")
                            if isinstance(item.get("retrieval_branch_scores"), dict)
                            else {}
                        ),
                        "similarity": dict(
                            item.get("similarity")
                            if isinstance(item.get("similarity"), dict)
                            else {}
                        ),
                        "calibration_trace": item.get("calibration_trace"),
                        "policy_trace": selected_item.get("policy_trace")
                        if isinstance(selected_item, dict)
                        else None,
                        "portfolio_trace": selected_item.get("portfolio_trace")
                        if isinstance(selected_item, dict)
                        else None,
                        "explainability_available": bool(explainability_cards.get(candidate_id)),
                    }
                )
            retrieved_universe = [
                {
                    "candidate_id": str(item.get("candidate_id") or ""),
                    "candidate_row_id": str(item.get("candidate_row_id") or ""),
                    "retrieved_rank": idx + 1,
                    "retrieval_branch_scores": dict(
                        item.get("retrieval_branch_scores")
                        if isinstance(item.get("retrieval_branch_scores"), dict)
                        else {}
                    ),
                    "similarity": {
                        "sparse": float(item.get("sparse_score") or 0.0),
                        "dense": float(item.get("dense_score") or 0.0),
                        "fused": float(item.get("fused_score") or 0.0),
                    },
                }
                for idx, item in enumerate(retrieved)
                if str(item.get("candidate_id") or "")
            ]
            response["debug"] = {
                "candidate_pool_size": len(candidate_rows),
                "retrieved_count": len(retrieved),
                "ranked_count": len(ranked_items),
                "bundle_dir": str(self.bundle_dir),
                "retrieval_metadata": retrieval_metadata,
                "comment_index_loaded": self.comment_index is not None,
                "comment_index_source_path": self.comment_index_source_path,
                "comment_index_load_error": self.comment_index_load_error,
                "retrieved_universe": retrieved_universe,
                "ranking_universe": ranking_universe,
                "portfolio_mode": bool(policy_metadata.get("portfolio_mode", False)),
                "portfolio_metadata": (
                    policy_metadata.get("portfolio_metadata")
                    if isinstance(policy_metadata.get("portfolio_metadata"), dict)
                    else {}
                ),
            }
        return response
