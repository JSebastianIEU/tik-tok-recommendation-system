from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

from ..contracts import CanonicalDatasetBundle
from .core import FABRIC_VERSION, FeatureFabric


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class FeatureSnapshotStats(BaseModel):
    rows_total: int = Field(ge=0)
    mandatory_coverage_rate: float = Field(ge=0.0, le=1.0)
    optional_coverage_rate: float = Field(ge=0.0, le=1.0)
    missingness_by_reason: Dict[str, int] = Field(default_factory=dict)


class FeatureSnapshotManifest(BaseModel):
    feature_manifest_id: str
    generated_at: str
    mode: str
    as_of_time: str
    fabric_version: str
    source_contract_version: str
    source_manifest_id: Optional[str] = None
    source_manifest_path: Optional[str] = None
    registry_signature: str
    feature_file: str
    format: str
    stats: FeatureSnapshotStats


def _write_feature_rows(rows: List[Dict[str, Any]], output_file_base: Path) -> tuple[str, str]:
    output_file_base.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd  # type: ignore

        output_file = output_file_base.with_suffix(".parquet")
        pd.DataFrame(rows).to_parquet(output_file, index=False)
        return str(output_file), "parquet"
    except Exception:
        output_file = output_file_base.with_suffix(".jsonl")
        with output_file.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        return str(output_file), "jsonl"


def _iter_latest_video_rows(bundle: CanonicalDatasetBundle) -> Iterable[Dict[str, Any]]:
    snapshots_by_video: Dict[str, List[Any]] = {}
    for snapshot in bundle.video_snapshots:
        snapshots_by_video.setdefault(snapshot.video_id, []).append(snapshot)
    for snapshots in snapshots_by_video.values():
        snapshots.sort(key=lambda item: item.scraped_at.astimezone(timezone.utc))
    for video in bundle.videos:
        latest = snapshots_by_video.get(video.video_id, [])
        latest_snapshot = latest[-1] if latest else None
        yield {
            "video": video,
            "latest_snapshot": latest_snapshot,
        }


def build_feature_snapshot_manifest(
    *,
    bundle: CanonicalDatasetBundle,
    output_root: Path | str,
    mode: str = "full",
    as_of_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    fabric = FeatureFabric()
    as_of = (as_of_time or bundle.generated_at).astimezone(timezone.utc)

    rows: List[Dict[str, Any]] = []
    missingness_by_reason: Dict[str, int] = {}
    mandatory_available = 0
    optional_available = 0
    for item in _iter_latest_video_rows(bundle):
        video = item["video"]
        snapshot = item["latest_snapshot"]
        profile = fabric.extract(
            {
                "video_id": video.video_id,
                "as_of_time": as_of,
                "caption": video.caption,
                "hashtags": list(video.hashtags),
                "keywords": list(video.keywords),
                "duration_seconds": video.duration_seconds,
                "content_type": "tutorial",
                "source_updated_at": snapshot.scraped_at if snapshot is not None else None,
            }
        )
        row = {
            "video_id": video.video_id,
            "author_id": video.author_id,
            "as_of_time": profile.as_of_time,
            "fabric_version": profile.fabric_version,
            "registry_signature": profile.registry_signature,
            "text_token_count": profile.text.token_count,
            "text_clarity_score": profile.text.clarity_score,
            "structure_hook_timing_seconds": profile.structure.hook_timing_seconds,
            "structure_payoff_timing_seconds": profile.structure.payoff_timing_seconds,
            "audio_speech_ratio": profile.audio.speech_ratio,
            "visual_shot_change_rate": profile.visual.shot_change_rate,
            "trace_ids": profile.trace_ids,
        }
        rows.append(row)
        if profile.text.token_count > 0 and profile.structure.pacing_score >= 0:
            mandatory_available += 1
        if profile.audio.speech_ratio is not None and profile.visual.shot_change_rate is not None:
            optional_available += 1

        for missing in (profile.text.missing, profile.audio.missing, profile.visual.missing):
            for _, meta in missing.items():
                reason = meta.reason
                missingness_by_reason[reason] = missingness_by_reason.get(reason, 0) + 1

    stats = FeatureSnapshotStats(
        rows_total=len(rows),
        mandatory_coverage_rate=0.0 if not rows else round(mandatory_available / len(rows), 6),
        optional_coverage_rate=0.0 if not rows else round(optional_available / len(rows), 6),
        missingness_by_reason=missingness_by_reason,
    )

    manifest_seed = {
        "mode": mode,
        "as_of_time": _to_iso(as_of),
        "contract_version": bundle.version,
        "source_manifest_id": bundle.manifest_id,
        "fabric_version": FABRIC_VERSION,
        "registry_signature": fabric.registry.signature(),
        "row_hash": _sha256_text(_canonical_json(rows)),
    }
    manifest_id = _sha256_text(_canonical_json(manifest_seed))
    manifest_dir = output_dir / manifest_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    feature_file, file_format = _write_feature_rows(rows, manifest_dir / "features")

    manifest = FeatureSnapshotManifest(
        feature_manifest_id=manifest_id,
        generated_at=_to_iso(datetime.now(timezone.utc)),
        mode=mode,
        as_of_time=_to_iso(as_of),
        fabric_version=FABRIC_VERSION,
        source_contract_version=str(bundle.version),
        source_manifest_id=bundle.manifest_id,
        source_manifest_path=None,
        registry_signature=fabric.registry.signature(),
        feature_file=feature_file,
        format=file_format,
        stats=stats,
    )
    payload = manifest.model_dump(mode="python")
    (manifest_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload
