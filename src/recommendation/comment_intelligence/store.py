from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from ..contracts import CanonicalDatasetBundle
from .core import (
    COMMENT_INTELLIGENCE_VERSION,
    CommentIntelligenceConfig,
    CommentIntelligenceSnapshot,
    CommentTransferPriors,
    build_transfer_priors_from_snapshots,
    extract_comment_intelligence_snapshots,
)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return _to_iso(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        rows.append(json.loads(raw))
    return rows


def _write_rows(rows: List[Dict[str, Any]], output_base: Path) -> Tuple[str, str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd  # type: ignore

        parquet_path = output_base.with_suffix(".parquet")
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)
        return str(parquet_path), "parquet"
    except Exception:
        jsonl_path = output_base.with_suffix(".jsonl")
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        return str(jsonl_path), "jsonl"


def _load_rows(file_path: Path, file_format: str) -> List[Dict[str, Any]]:
    if file_format == "jsonl":
        return _read_jsonl(file_path)
    if file_format == "parquet":
        try:
            import pandas as pd  # type: ignore

            frame = pd.read_parquet(file_path)
            return frame.to_dict(orient="records")
        except Exception:
            fallback_jsonl = file_path.with_suffix(".jsonl")
            if fallback_jsonl.exists():
                return _read_jsonl(fallback_jsonl)
            raise
    raise ValueError(f"Unsupported file format '{file_format}'.")


class CommentIntelligenceStats(BaseModel):
    rows_total: int = Field(ge=0)
    coverage_rate: float = Field(ge=0.0, le=1.0)
    alignment_coverage_rate: float = Field(ge=0.0, le=1.0)
    low_alignment_rate: float = Field(ge=0.0, le=1.0)
    alignment_confidence_mean: float = Field(ge=0.0, le=1.0)
    alignment_shift_early_late_mean: float
    missingness_by_reason: Dict[str, int] = Field(default_factory=dict)
    intent_distribution: Dict[str, float] = Field(default_factory=dict)
    sentiment_volatility_mean: float = Field(ge=0.0)
    late_out_of_watermark_share: float = Field(ge=0.0, le=1.0)
    extraction_latency_ms_p50: float = Field(ge=0.0)
    extraction_latency_ms_p95: float = Field(ge=0.0)


class CommentIntelligenceSnapshotManifest(BaseModel):
    comment_feature_manifest_id: str
    generated_at: str
    mode: str
    as_of_time: str
    comment_intelligence_version: str = COMMENT_INTELLIGENCE_VERSION
    taxonomy_version: str
    source_contract_version: str
    source_manifest_id: Optional[str] = None
    source_manifest_path: Optional[str] = None
    feature_file: str
    format: str
    stats: CommentIntelligenceStats


class CommentTransferPriorsManifest(BaseModel):
    comment_priors_manifest_id: str
    generated_at: str
    as_of_time: str
    comment_intelligence_version: str = COMMENT_INTELLIGENCE_VERSION
    taxonomy_version: str
    source_comment_feature_manifest_id: str
    priors_file: str
    format: str
    rows_total: int = Field(ge=0)


def _latency_percentiles(values_ms: Sequence[float]) -> Tuple[float, float]:
    if not values_ms:
        return 0.0, 0.0
    ordered = sorted(float(item) for item in values_ms)
    idx50 = int(round((len(ordered) - 1) * 0.5))
    idx95 = int(round((len(ordered) - 1) * 0.95))
    return float(ordered[idx50]), float(ordered[idx95])


def build_comment_intelligence_snapshot_manifest(
    *,
    bundle: CanonicalDatasetBundle,
    as_of_time: Optional[datetime],
    output_root: Path | str,
    mode: str = "full",
    config: Optional[CommentIntelligenceConfig] = None,
) -> Dict[str, Any]:
    cfg = config or CommentIntelligenceConfig()
    as_of_utc = (as_of_time or bundle.generated_at).astimezone(timezone.utc)

    started = time.perf_counter()
    snapshots = extract_comment_intelligence_snapshots(
        bundle=bundle,
        as_of_time=as_of_utc,
        config=cfg,
    )
    elapsed_ms_total = (time.perf_counter() - started) * 1000.0
    rows = [item.model_dump(mode="python") for item in snapshots]
    row_hash = _sha256_text(_canonical_json(rows))
    manifest_seed = {
        "as_of_time": _to_iso(as_of_utc),
        "mode": mode,
        "row_hash": row_hash,
        "taxonomy_version": cfg.taxonomy_version,
        "source_contract_version": str(bundle.version),
        "source_manifest_id": bundle.manifest_id,
    }
    manifest_id = _sha256_text(_canonical_json(manifest_seed))

    output_dir = Path(output_root) / manifest_id
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_file, file_format = _write_rows(rows, output_dir / "comment_features")

    missingness_by_reason: Dict[str, int] = {}
    intent_distribution: Dict[str, float] = {}
    volatility_values: List[float] = []
    alignment_values: List[float] = []
    alignment_conf_values: List[float] = []
    alignment_shift_values: List[float] = []
    per_row_latency = (
        0.0 if not snapshots else elapsed_ms_total / max(1, len(snapshots))
    )
    latency_values: List[float] = []
    for item in snapshots:
        volatility_values.append(float(item.features.sentiment_volatility))
        alignment_values.append(float(item.features.alignment_score))
        alignment_conf_values.append(float(item.features.alignment_confidence))
        alignment_shift_values.append(float(item.features.alignment_shift_early_late))
        for reason in item.missingness.values():
            missingness_by_reason[reason.reason] = missingness_by_reason.get(reason.reason, 0) + 1
        for label, value in item.features.intent_rates.items():
            intent_distribution[label] = intent_distribution.get(label, 0.0) + float(value)
        latency_values.append(per_row_latency)

    rows_total = len(snapshots)
    if rows_total > 0:
        for key in list(intent_distribution.keys()):
            intent_distribution[key] = round(intent_distribution[key] / rows_total, 6)
    coverage = _to_coverage_rate(snapshots)
    alignment_coverage = (
        _safe_div(
            float(sum(1 for value in alignment_conf_values if value > 0.0)),
            float(max(1, rows_total)),
        )
        if rows_total > 0
        else 0.0
    )
    low_alignment_rate = (
        _safe_div(
            float(sum(1 for value in alignment_values if value < 0.35)),
            float(max(1, rows_total)),
        )
        if rows_total > 0
        else 0.0
    )
    p50_ms, p95_ms = _latency_percentiles(latency_values)

    late_out = sum(
        1
        for item in bundle.comment_snapshots
        if getattr(item, "lateness_class", "on_time") == "late_out_of_watermark"
    )
    late_share = 0.0 if not bundle.comment_snapshots else late_out / len(bundle.comment_snapshots)
    stats = CommentIntelligenceStats(
        rows_total=rows_total,
        coverage_rate=round(coverage, 6),
        alignment_coverage_rate=round(alignment_coverage, 6),
        low_alignment_rate=round(low_alignment_rate, 6),
        alignment_confidence_mean=round(
            _safe_div(sum(alignment_conf_values), float(max(1, len(alignment_conf_values)))),
            6,
        ),
        alignment_shift_early_late_mean=round(
            _safe_div(sum(alignment_shift_values), float(max(1, len(alignment_shift_values)))),
            6,
        ),
        missingness_by_reason=missingness_by_reason,
        intent_distribution=intent_distribution,
        sentiment_volatility_mean=round(
            0.0 if not volatility_values else sum(volatility_values) / len(volatility_values),
            6,
        ),
        late_out_of_watermark_share=round(late_share, 6),
        extraction_latency_ms_p50=round(p50_ms, 4),
        extraction_latency_ms_p95=round(p95_ms, 4),
    )
    manifest = CommentIntelligenceSnapshotManifest(
        comment_feature_manifest_id=manifest_id,
        generated_at=_to_iso(datetime.now(timezone.utc)),
        mode=mode,
        as_of_time=_to_iso(as_of_utc),
        taxonomy_version=cfg.taxonomy_version,
        source_contract_version=str(bundle.version),
        source_manifest_id=bundle.manifest_id,
        source_manifest_path=None,
        feature_file=feature_file,
        format=file_format,
        stats=stats,
    )
    payload = manifest.model_dump(mode="python")
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return payload


def _to_coverage_rate(snapshots: Sequence[CommentIntelligenceSnapshot]) -> float:
    if not snapshots:
        return 0.0
    available = sum(1 for item in snapshots if item.features.comment_count_total > 0)
    return available / len(snapshots)


def _resolve_manifest_path(manifest_ref: Path | str) -> Path:
    path = Path(manifest_ref)
    if path.is_dir():
        manifest_path = path / "manifest.json"
    else:
        manifest_path = path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Comment intelligence manifest not found: {manifest_path}")
    return manifest_path


def load_comment_intelligence_snapshot_manifest(
    manifest_ref: Path | str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    manifest_path = _resolve_manifest_path(manifest_ref)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = CommentIntelligenceSnapshotManifest.model_validate(payload).model_dump(mode="python")
    rows = _load_rows(Path(manifest["feature_file"]), str(manifest["format"]))
    return manifest, rows


def build_comment_transfer_priors(
    *,
    snapshot_manifest: Path | str | Dict[str, Any],
    output_root: Path | str,
    min_support: int = 3,
    shrinkage_alpha: float = 8.0,
) -> Dict[str, Any]:
    if isinstance(snapshot_manifest, dict):
        manifest = CommentIntelligenceSnapshotManifest.model_validate(snapshot_manifest)
        rows = _load_rows(Path(manifest.feature_file), str(manifest.format))
    else:
        payload, rows = load_comment_intelligence_snapshot_manifest(snapshot_manifest)
        manifest = CommentIntelligenceSnapshotManifest.model_validate(payload)

    snapshots = [CommentIntelligenceSnapshot.model_validate(item) for item in rows]
    as_of = datetime.fromisoformat(manifest.as_of_time.replace("Z", "+00:00")).astimezone(timezone.utc)
    priors = build_transfer_priors_from_snapshots(
        snapshots=snapshots,
        as_of_time=as_of,
        taxonomy_version=manifest.taxonomy_version,
        min_support=min_support,
        shrinkage_alpha=shrinkage_alpha,
    )
    prior_rows = [item.model_dump(mode="python") for item in priors.entries]
    row_hash = _sha256_text(_canonical_json(prior_rows))
    manifest_seed = {
        "source_comment_feature_manifest_id": manifest.comment_feature_manifest_id,
        "row_hash": row_hash,
        "as_of_time": manifest.as_of_time,
        "taxonomy_version": manifest.taxonomy_version,
        "min_support": int(min_support),
        "shrinkage_alpha": float(shrinkage_alpha),
    }
    priors_manifest_id = _sha256_text(_canonical_json(manifest_seed))
    output_dir = Path(output_root) / priors_manifest_id
    output_dir.mkdir(parents=True, exist_ok=True)
    priors_file, file_format = _write_rows(prior_rows, output_dir / "comment_priors")

    payload = CommentTransferPriorsManifest(
        comment_priors_manifest_id=priors_manifest_id,
        generated_at=_to_iso(datetime.now(timezone.utc)),
        as_of_time=manifest.as_of_time,
        taxonomy_version=manifest.taxonomy_version,
        source_comment_feature_manifest_id=manifest.comment_feature_manifest_id,
        priors_file=priors_file,
        format=file_format,
        rows_total=len(prior_rows),
    ).model_dump(mode="python")
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def load_comment_transfer_priors_manifest(
    manifest_ref: Path | str,
) -> Tuple[Dict[str, Any], CommentTransferPriors]:
    manifest_path = _resolve_manifest_path(manifest_ref)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = CommentTransferPriorsManifest.model_validate(payload).model_dump(mode="python")
    rows = _load_rows(Path(manifest["priors_file"]), str(manifest["format"]))
    priors = CommentTransferPriors(
        as_of_time=str(manifest["as_of_time"]),
        taxonomy_version=str(manifest["taxonomy_version"]),
        entries=[row for row in rows],
    )
    return manifest, priors


__all__ = [
    "CommentIntelligenceSnapshotManifest",
    "CommentIntelligenceStats",
    "CommentTransferPriorsManifest",
    "build_comment_intelligence_snapshot_manifest",
    "build_comment_transfer_priors",
    "load_comment_intelligence_snapshot_manifest",
    "load_comment_transfer_priors_manifest",
]
