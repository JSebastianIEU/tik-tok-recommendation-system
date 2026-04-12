from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field


FABRIC_VERSION = "fabric.v2"

MissingReason = Literal[
    "not_available",
    "low_quality",
    "extraction_failed",
    "not_applicable",
]


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _safe_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for item in values:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                out.append(stripped)
    return out


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9#@ ]+", " ", text.lower())).strip()


def _tokens(text: str) -> List[str]:
    return [token for token in _normalize_text(text).split(" ") if token]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class MissingFeature(BaseModel):
    reason: MissingReason
    source_age_hours: Optional[float] = Field(default=None, ge=0.0)
    detail: Optional[str] = None


class ConfidenceScore(BaseModel):
    raw: float = Field(ge=0.0, le=1.0)
    calibrated: float = Field(ge=0.0, le=1.0)
    calibration_version: str


class ExtractorTrace(BaseModel):
    extractor: str
    extractor_version: str
    model_version: str
    output_schema_hash: str
    input_digest: str
    generated_at: str


class TextFeatures(BaseModel):
    token_count: int = Field(ge=0)
    unique_token_count: int = Field(ge=0)
    hashtag_count: int = Field(ge=0)
    keyphrase_count: int = Field(ge=0)
    cta_keyword_count: int = Field(ge=0)
    clarity_score: float = Field(ge=0.0, le=1.0)
    confidence: ConfidenceScore
    missing: Dict[str, MissingFeature] = Field(default_factory=dict)


class StructureFeatures(BaseModel):
    hook_window_start_sec: float = Field(ge=0.0)
    hook_window_end_sec: float = Field(ge=0.0)
    mid_window_start_sec: float = Field(ge=0.0)
    mid_window_end_sec: float = Field(ge=0.0)
    payoff_window_start_sec: float = Field(ge=0.0)
    payoff_window_end_sec: float = Field(ge=0.0)
    hook_timing_seconds: float = Field(ge=0.0)
    payoff_timing_seconds: float = Field(ge=0.0)
    step_density: float = Field(ge=0.0)
    pacing_score: float = Field(ge=0.0, le=1.0)
    confidence: ConfidenceScore
    missing: Dict[str, MissingFeature] = Field(default_factory=dict)


class AudioFeatures(BaseModel):
    speech_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    tempo_bpm: Optional[float] = Field(default=None, ge=0.0)
    energy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    music_presence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence: ConfidenceScore
    missing: Dict[str, MissingFeature] = Field(default_factory=dict)


class VisualFeatures(BaseModel):
    shot_change_rate: Optional[float] = Field(default=None, ge=0.0)
    visual_motion_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    style_tags: List[str] = Field(default_factory=list)
    confidence: ConfidenceScore
    missing: Dict[str, MissingFeature] = Field(default_factory=dict)


class FeatureFabricInput(BaseModel):
    video_id: str
    as_of_time: datetime
    caption: str = ""
    hashtags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    transcript_text: Optional[str] = None
    ocr_text: Optional[str] = None
    duration_seconds: Optional[int] = Field(default=None, ge=0)
    content_type: Optional[str] = None
    source_updated_at: Optional[datetime] = None
    hints: Dict[str, Any] = Field(default_factory=dict)


class FeatureFabricOutput(BaseModel):
    fabric_version: Literal["fabric.v2"]
    generated_at: str
    video_id: str
    as_of_time: str
    registry_signature: str
    extractor_traces: List[ExtractorTrace]
    text: TextFeatures
    structure: StructureFeatures
    audio: AudioFeatures
    visual: VisualFeatures
    trace_ids: List[str]


@dataclass
class _ExtractorSpec:
    name: str
    version: str
    model_version: str
    output_schema_hash: str
    fn: Callable[[FeatureFabricInput], Dict[str, Any]]


class FeatureFabricRegistry:
    def __init__(self) -> None:
        self._extractors: Dict[str, _ExtractorSpec] = {}

    def register(
        self,
        *,
        name: str,
        version: str,
        model_version: str,
        output_schema_hash: str,
        fn: Callable[[FeatureFabricInput], Dict[str, Any]],
    ) -> None:
        if not re.match(r"^\d+\.\d+\.\d+$", version):
            raise ValueError(f"Extractor version must be semver (x.y.z). Got '{version}'.")
        self._extractors[name] = _ExtractorSpec(
            name=name,
            version=version,
            model_version=model_version,
            output_schema_hash=output_schema_hash,
            fn=fn,
        )

    def get(self, name: str) -> _ExtractorSpec:
        if name not in self._extractors:
            raise KeyError(f"Unknown extractor '{name}'.")
        return self._extractors[name]

    def signature(self) -> str:
        payload = [
            {
                "name": spec.name,
                "version": spec.version,
                "model_version": spec.model_version,
                "schema": spec.output_schema_hash,
            }
            for _, spec in sorted(self._extractors.items(), key=lambda item: item[0])
        ]
        return _sha256_text(_canonical_json(payload))

    def assert_compatible(self, expected_schema_hashes: Dict[str, str]) -> None:
        mismatches: List[str] = []
        for name, expected_hash in expected_schema_hashes.items():
            spec = self._extractors.get(name)
            if spec is None:
                mismatches.append(f"{name}: extractor not registered")
                continue
            if spec.output_schema_hash != expected_hash:
                mismatches.append(
                    f"{name}: expected schema {expected_hash}, got {spec.output_schema_hash}"
                )
        if mismatches:
            raise ValueError("Feature registry compatibility mismatch: " + " | ".join(mismatches))


class _Calibrator:
    def __init__(
        self,
        slope: float,
        bias: float,
        version: str,
        bins: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        self.slope = slope
        self.bias = bias
        self.version = version
        self.bins = bins or []

    def apply(self, raw: float) -> ConfidenceScore:
        calibrated = _clamp(raw * self.slope + self.bias, 0.0, 1.0)
        if len(self.bins) >= 2:
            ordered = sorted(self.bins, key=lambda item: item["raw"])
            if raw <= ordered[0]["raw"]:
                calibrated = ordered[0]["target"]
            elif raw >= ordered[-1]["raw"]:
                calibrated = ordered[-1]["target"]
            else:
                for left, right in zip(ordered[:-1], ordered[1:]):
                    if left["raw"] <= raw <= right["raw"]:
                        span = max(right["raw"] - left["raw"], 1e-9)
                        ratio = (raw - left["raw"]) / span
                        calibrated = left["target"] + (right["target"] - left["target"]) * ratio
                        break
        calibrated = _clamp(calibrated, 0.0, 1.0)
        return ConfidenceScore(
            raw=round(raw, 6),
            calibrated=round(calibrated, 6),
            calibration_version=self.version,
        )

    @classmethod
    def from_observations(
        cls,
        observations: Sequence[tuple[float, float]],
        *,
        version: str,
        bins: int = 8,
    ) -> "_Calibrator":
        if not observations:
            return cls(1.0, 0.0, version, [])
        clipped = [(_clamp(raw, 0.0, 1.0), _clamp(target, 0.0, 1.0)) for raw, target in observations]
        clipped.sort(key=lambda item: item[0])
        step = max(1, len(clipped) // max(1, bins))
        points: List[Dict[str, float]] = []
        for idx in range(0, len(clipped), step):
            chunk = clipped[idx : idx + step]
            if not chunk:
                continue
            avg_raw = sum(item[0] for item in chunk) / len(chunk)
            avg_target = sum(item[1] for item in chunk) / len(chunk)
            points.append({"raw": round(avg_raw, 6), "target": round(avg_target, 6)})
        if points and points[-1]["raw"] < clipped[-1][0]:
            points.append(
                {
                    "raw": round(clipped[-1][0], 6),
                    "target": round(clipped[-1][1], 6),
                }
            )
        slope = 1.0
        bias = 0.0
        if len(clipped) >= 2:
            x_mean = sum(item[0] for item in clipped) / len(clipped)
            y_mean = sum(item[1] for item in clipped) / len(clipped)
            num = sum((x - x_mean) * (y - y_mean) for x, y in clipped)
            den = sum((x - x_mean) ** 2 for x, _ in clipped)
            if den > 0:
                slope = num / den
                bias = y_mean - slope * x_mean
        return cls(slope=slope, bias=bias, version=version, bins=points)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slope": self.slope,
            "bias": self.bias,
            "version": self.version,
            "bins": self.bins,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], default_version: str) -> "_Calibrator":
        return cls(
            slope=float(payload.get("slope", 1.0)),
            bias=float(payload.get("bias", 0.0)),
            version=str(payload.get("version") or default_version),
            bins=[
                {"raw": float(item["raw"]), "target": float(item["target"])}
                for item in payload.get("bins", [])
                if isinstance(item, dict) and "raw" in item and "target" in item
            ],
        )


def _source_age_hours(as_of_time: datetime, source_updated_at: Optional[datetime]) -> Optional[float]:
    if source_updated_at is None:
        return None
    delta = as_of_time.astimezone(timezone.utc) - source_updated_at.astimezone(timezone.utc)
    return round(max(delta.total_seconds(), 0.0) / 3600.0, 4)


CTA_TERMS = {
    "follow",
    "comment",
    "save",
    "share",
    "subscribe",
    "join",
    "shop",
    "link",
    "bio",
    "sigue",
    "guarda",
    "comenta",
}

CONTENT_TYPE_WINDOWS = {
    "tutorial": {"hook": 0.16, "payoff": 0.78},
    "story": {"hook": 0.14, "payoff": 0.83},
    "reaction": {"hook": 0.12, "payoff": 0.74},
    "showcase": {"hook": 0.15, "payoff": 0.8},
    "opinion": {"hook": 0.18, "payoff": 0.82},
}


class FeatureFabric:
    def __init__(self, calibration_artifacts: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.registry = FeatureFabricRegistry()
        default_calibrators: Dict[str, _Calibrator] = {
            "text": _Calibrator(0.94, 0.03, "text-cal.v1"),
            "structure": _Calibrator(0.92, 0.04, "structure-cal.v1"),
            "audio": _Calibrator(0.9, 0.05, "audio-cal.v1"),
            "visual": _Calibrator(0.9, 0.05, "visual-cal.v1"),
        }
        artifacts = calibration_artifacts or {}
        self.calibrators = {
            name: _Calibrator.from_dict(
                artifacts.get(name, {}),
                default_version=default.version,
            )
            if name in artifacts
            else default
            for name, default in default_calibrators.items()
        }
        self.registry.register(
            name="text",
            version="2.0.0",
            model_version="rules.v2",
            output_schema_hash=_sha256_text(_canonical_json(TextFeatures.model_json_schema())),
            fn=self._extract_text,
        )
        self.registry.register(
            name="structure",
            version="2.0.0",
            model_version="rules.v2",
            output_schema_hash=_sha256_text(_canonical_json(StructureFeatures.model_json_schema())),
            fn=self._extract_structure,
        )
        self.registry.register(
            name="audio",
            version="2.0.0",
            model_version="rules.v2",
            output_schema_hash=_sha256_text(_canonical_json(AudioFeatures.model_json_schema())),
            fn=self._extract_audio,
        )
        self.registry.register(
            name="visual",
            version="2.0.0",
            model_version="rules.v2",
            output_schema_hash=_sha256_text(_canonical_json(VisualFeatures.model_json_schema())),
            fn=self._extract_visual,
        )

    def calibration_artifacts(self) -> Dict[str, Dict[str, Any]]:
        return {name: calibrator.to_dict() for name, calibrator in self.calibrators.items()}

    def fit_calibrators(
        self,
        observations: Dict[str, Sequence[tuple[float, float]]],
    ) -> Dict[str, Dict[str, Any]]:
        fitted: Dict[str, Dict[str, Any]] = {}
        for name, samples in observations.items():
            if name not in self.calibrators:
                continue
            version = f"{name}-cal.v2"
            self.calibrators[name] = _Calibrator.from_observations(
                samples,
                version=version,
            )
            fitted[name] = self.calibrators[name].to_dict()
        return fitted

    def _build_trace(self, name: str, spec: _ExtractorSpec, request: FeatureFabricInput) -> ExtractorTrace:
        digest_payload = {
            "video_id": request.video_id,
            "as_of_time": _to_iso(request.as_of_time),
            "caption": request.caption,
            "hashtags": sorted(request.hashtags),
            "keywords": sorted(request.keywords),
            "transcript_text": request.transcript_text or "",
            "ocr_text": request.ocr_text or "",
            "duration_seconds": request.duration_seconds,
            "content_type": request.content_type or "",
            "hints": request.hints,
        }
        return ExtractorTrace(
            extractor=name,
            extractor_version=spec.version,
            model_version=spec.model_version,
            output_schema_hash=spec.output_schema_hash,
            input_digest=_sha256_text(_canonical_json(digest_payload)),
            generated_at=_to_iso(request.as_of_time),
        )

    def _extract_text(self, request: FeatureFabricInput) -> Dict[str, Any]:
        caption = _safe_text(request.caption)
        transcript = _safe_text(request.transcript_text)
        ocr_text = _safe_text(request.ocr_text)
        video_caption = _safe_text((request.hints or {}).get("video_caption", ""))
        hashtags = _safe_list(request.hashtags)
        combined = " ".join([caption, transcript, ocr_text, video_caption, " ".join(hashtags), " ".join(_safe_list(request.keywords))]).strip()
        tokens = _tokens(combined)
        unique_tokens = set(tokens)
        cta_count = sum(1 for token in tokens if token in CTA_TERMS)
        lexical_diversity = len(unique_tokens) / max(1, len(tokens))
        clarity_raw = _clamp(0.42 + lexical_diversity * 0.5 + (0.04 if cta_count > 0 else 0.0), 0.0, 1.0)
        missing: Dict[str, Any] = {}
        source_age = _source_age_hours(request.as_of_time, request.source_updated_at)
        if not transcript:
            missing["transcript_text"] = MissingFeature(
                reason="not_available",
                source_age_hours=source_age,
                detail="No transcript supplied.",
            ).model_dump(mode="python")
        if not ocr_text:
            missing["ocr_text"] = MissingFeature(
                reason="not_available",
                source_age_hours=source_age,
                detail="No OCR supplied.",
            ).model_dump(mode="python")
        if not caption:
            missing["caption"] = MissingFeature(
                reason="low_quality",
                source_age_hours=source_age,
                detail="Caption is empty.",
            ).model_dump(mode="python")
        if not _safe_list(request.keywords):
            missing["keyphrase_count"] = MissingFeature(
                reason="not_available",
                source_age_hours=source_age,
                detail="No keyphrases provided.",
            ).model_dump(mode="python")
        if len(tokens) <= 2:
            missing["token_count"] = MissingFeature(
                reason="low_quality",
                source_age_hours=source_age,
                detail="Text payload too short for robust lexical analysis.",
            ).model_dump(mode="python")
        return {
            "token_count": len(tokens),
            "unique_token_count": len(unique_tokens),
            "hashtag_count": len(hashtags),
            "keyphrase_count": min(len(_safe_list(request.keywords)), 32),
            "cta_keyword_count": cta_count,
            "clarity_score": round(clarity_raw, 6),
            "confidence": self.calibrators["text"].apply(raw=_clamp(0.62 + 0.08 * (1 if transcript else 0) + 0.06 * (1 if ocr_text else 0) + 0.08 * (1 if video_caption else 0), 0.0, 1.0)).model_dump(mode="python"),
            "missing": missing,
        }

    def _extract_structure(self, request: FeatureFabricInput) -> Dict[str, Any]:
        duration = int(request.duration_seconds or max(10, min(180, len(_tokens(request.caption)) * 2)))
        ctype = _safe_text(request.content_type).lower() or "tutorial"
        profile = CONTENT_TYPE_WINDOWS.get(ctype, CONTENT_TYPE_WINDOWS["tutorial"])
        hook_end = round(max(1.0, duration * profile["hook"]), 4)
        payoff_start = round(max(hook_end + 2.0, duration * profile["payoff"]), 4)
        mid_start = hook_end
        mid_end = max(mid_start, payoff_start - 0.5)
        step_density = round(len(_safe_list(request.keywords)) / max(duration / 10.0, 1.0), 6)
        pacing_raw = _clamp(0.5 + min(duration, 90) / 180 + min(step_density, 2.0) / 6, 0.0, 1.0)
        missing: Dict[str, Any] = {}
        source_age = _source_age_hours(request.as_of_time, request.source_updated_at)
        if request.duration_seconds is None:
            missing["duration_seconds"] = MissingFeature(
                reason="not_available",
                source_age_hours=source_age,
                detail="Duration missing; inferred from caption length.",
            ).model_dump(mode="python")
        if not request.content_type:
            missing["content_type"] = MissingFeature(
                reason="not_available",
                source_age_hours=source_age,
                detail="Content type missing; using default segment profile.",
            ).model_dump(mode="python")
        return {
            "hook_window_start_sec": 0.0,
            "hook_window_end_sec": hook_end,
            "mid_window_start_sec": round(mid_start, 4),
            "mid_window_end_sec": round(mid_end, 4),
            "payoff_window_start_sec": round(payoff_start, 4),
            "payoff_window_end_sec": round(float(duration), 4),
            "hook_timing_seconds": round(hook_end, 4),
            "payoff_timing_seconds": round(payoff_start, 4),
            "step_density": step_density,
            "pacing_score": round(pacing_raw, 6),
            "confidence": self.calibrators["structure"].apply(
                raw=_clamp(0.66 + (0.08 if request.duration_seconds else 0.0), 0.0, 1.0)
            ).model_dump(mode="python"),
            "missing": missing,
        }

    def _extract_audio(self, request: FeatureFabricInput) -> Dict[str, Any]:
        source_age = _source_age_hours(request.as_of_time, request.source_updated_at)
        hints = request.hints or {}
        has_audio_hints = any(
            key in hints
            for key in ("speech_seconds", "music_seconds", "tempo_bpm", "audio_energy")
        )
        if not has_audio_hints:
            missing = {
                "speech_ratio": MissingFeature(
                    reason="not_available",
                    source_age_hours=source_age,
                    detail="Audio extractors are offline-only in v1.",
                ).model_dump(mode="python"),
                "tempo_bpm": MissingFeature(
                    reason="not_available",
                    source_age_hours=source_age,
                    detail="Audio extractors are offline-only in v1.",
                ).model_dump(mode="python"),
                "energy": MissingFeature(
                    reason="not_available",
                    source_age_hours=source_age,
                    detail="Audio extractors are offline-only in v1.",
                ).model_dump(mode="python"),
                "music_presence_score": MissingFeature(
                    reason="not_available",
                    source_age_hours=source_age,
                    detail="Audio extractors are offline-only in v1.",
                ).model_dump(mode="python"),
            }
            return {
                "speech_ratio": None,
                "tempo_bpm": None,
                "energy": None,
                "music_presence_score": None,
                "confidence": self.calibrators["audio"].apply(raw=0.3).model_dump(mode="python"),
                "missing": missing,
            }
        duration = float(request.duration_seconds or 30.0)
        speech_seconds = float(hints.get("speech_seconds") or duration * 0.55)
        music_seconds = float(hints.get("music_seconds") or max(0.0, duration - speech_seconds))
        speech_ratio = _clamp(speech_seconds / max(speech_seconds + music_seconds, 1.0), 0.0, 1.0)
        tempo = float(hints.get("tempo_bpm") or 108.0)
        energy = _clamp(float(hints.get("audio_energy") or ((tempo - 70.0) / 120.0)), 0.0, 1.0)
        music_presence = _clamp(1.0 - speech_ratio + 0.2, 0.0, 1.0)
        return {
            "speech_ratio": round(speech_ratio, 6),
            "tempo_bpm": round(tempo, 6),
            "energy": round(energy, 6),
            "music_presence_score": round(music_presence, 6),
            "confidence": self.calibrators["audio"].apply(raw=0.76).model_dump(mode="python"),
            "missing": {},
        }

    def _extract_visual(self, request: FeatureFabricInput) -> Dict[str, Any]:
        source_age = _source_age_hours(request.as_of_time, request.source_updated_at)
        hints = request.hints or {}
        has_visual_hints = any(
            key in hints for key in ("estimated_scene_cuts", "visual_motion_score", "fps")
        )
        if not has_visual_hints:
            missing = {
                "shot_change_rate": MissingFeature(
                    reason="not_available",
                    source_age_hours=source_age,
                    detail="Visual extractors are offline-only in v1.",
                ).model_dump(mode="python"),
                "visual_motion_score": MissingFeature(
                    reason="not_available",
                    source_age_hours=source_age,
                    detail="Visual extractors are offline-only in v1.",
                ).model_dump(mode="python"),
            }
            return {
                "shot_change_rate": None,
                "visual_motion_score": None,
                "style_tags": [],
                "confidence": self.calibrators["visual"].apply(raw=0.3).model_dump(mode="python"),
                "missing": missing,
            }
        duration = float(request.duration_seconds or 30.0)
        scene_cuts = max(float(hints.get("estimated_scene_cuts") or 12.0), 1.0)
        shot_rate = scene_cuts / max(duration, 1.0)
        motion = _clamp(float(hints.get("visual_motion_score") or shot_rate / 1.8), 0.0, 1.0)
        style_tags: List[str] = []
        if shot_rate >= 0.5:
            style_tags.append("rapid_cut")
        elif shot_rate <= 0.22:
            style_tags.append("slow_cut")
        else:
            style_tags.append("balanced_cut")
        if motion >= 0.7:
            style_tags.append("high_motion")
        elif motion <= 0.35:
            style_tags.append("static_framing")
        return {
            "shot_change_rate": round(shot_rate, 6),
            "visual_motion_score": round(motion, 6),
            "style_tags": style_tags,
            "confidence": self.calibrators["visual"].apply(raw=0.78).model_dump(mode="python"),
            "missing": {},
        }

    def extract(
        self,
        payload: FeatureFabricInput | Dict[str, Any],
    ) -> FeatureFabricOutput:
        request = payload if isinstance(payload, FeatureFabricInput) else FeatureFabricInput.model_validate(payload)
        traces: List[ExtractorTrace] = []

        out_blocks: Dict[str, Any] = {}
        for name in ("text", "structure", "audio", "visual"):
            spec = self.registry.get(name)
            traces.append(self._build_trace(name, spec, request))
            raw = spec.fn(request)
            if name == "text":
                out_blocks["text"] = TextFeatures.model_validate(raw)
            elif name == "structure":
                out_blocks["structure"] = StructureFeatures.model_validate(raw)
            elif name == "audio":
                out_blocks["audio"] = AudioFeatures.model_validate(raw)
            elif name == "visual":
                out_blocks["visual"] = VisualFeatures.model_validate(raw)

        trace_ids = [
            _sha256_text(
                _canonical_json(
                    {
                        "extractor": trace.extractor,
                        "version": trace.extractor_version,
                        "input_digest": trace.input_digest,
                    }
                )
            )[:16]
            for trace in traces
        ]

        return FeatureFabricOutput(
            fabric_version=FABRIC_VERSION,
            generated_at=_to_iso(request.as_of_time),
            video_id=request.video_id,
            as_of_time=_to_iso(request.as_of_time),
            registry_signature=self.registry.signature(),
            extractor_traces=traces,
            text=out_blocks["text"],
            structure=out_blocks["structure"],
            audio=out_blocks["audio"],
            visual=out_blocks["visual"],
            trace_ids=trace_ids,
        )
