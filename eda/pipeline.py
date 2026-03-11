from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from eda.src.contracts import ContractValidationResult, validate_dataset_rows, validate_silver_rows
from eda.src.lineage import build_lineage
from eda.src.loaders import LoadRequest, fetch_dataset, write_extract
from eda.src.metadata_store import append_run_registry
from eda.src.quality import build_quality_scorecard
from eda.src.silver import build_silver_dataset
from scraper.db.client import get_database_url


@dataclass(frozen=True)
class DatasetRequest:
    name: str
    limit: int = 1000
    since: str | None = None
    all_pages: bool = False
    fmt: str = "jsonl"


def build_run_id(now: datetime | None = None) -> str:
    ts = now or datetime.now(tz=timezone.utc)
    return ts.strftime("%Y%m%d_%H%M%S")


def build_run_dir(base_dir: Path | str, run_id: str | None = None) -> Path:
    root = Path(base_dir).resolve()
    rid = run_id or build_run_id()
    out = root / rid
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_plan(path: Path | str) -> list[DatasetRequest]:
    plan_path = Path(path).resolve()
    if not plan_path.exists():
        raise FileNotFoundError(f"EDA plan not found: {plan_path}")

    data = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}
    raw_datasets = data.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError("EDA plan must contain non-empty 'datasets' list")

    requests: list[DatasetRequest] = []
    for idx, item in enumerate(raw_datasets):
        if not isinstance(item, dict):
            raise ValueError(f"datasets[{idx}] must be an object")
        name = str(item.get("name", "")).strip()
        if name not in {"full", "videos", "comments", "authors"}:
            raise ValueError(
                f"datasets[{idx}].name must be one of: authors, comments, full, videos"
            )

        req = DatasetRequest(
            name=name,
            limit=int(item.get("limit", 1000)),
            since=item.get("since"),
            all_pages=bool(item.get("all_pages", False)),
            fmt=str(item.get("format", "jsonl")).strip().lower(),
        )
        if req.limit <= 0:
            raise ValueError(f"datasets[{idx}].limit must be > 0")
        if req.fmt not in {"jsonl", "csv"}:
            raise ValueError(f"datasets[{idx}].format must be jsonl or csv")
        requests.append(req)

    return requests


def load_profiles(path: Path | str = "eda/config/profiles.yaml") -> dict[str, dict[str, Any]]:
    profiles_path = Path(path).resolve()
    if not profiles_path.exists():
        return {}
    data = yaml.safe_load(profiles_path.read_text(encoding="utf-8")) or {}
    raw_profiles = data.get("profiles", {})
    if not isinstance(raw_profiles, dict):
        raise ValueError("profiles.yaml must contain an object at key 'profiles'")
    return {str(k): dict(v) for k, v in raw_profiles.items() if isinstance(v, dict)}


def resolve_db_url(*, db_url: str | None, profile: str | None, profiles_path: Path | str) -> str:
    if db_url:
        return get_database_url(db_url)
    if profile:
        profiles = load_profiles(profiles_path)
        if profile not in profiles:
            raise ValueError(f"Unknown EDA profile: {profile}")
        env_var = str(profiles[profile].get("db_env_var", "DATABASE_URL"))
        resolved = os.getenv(env_var)
        if not resolved:
            raise RuntimeError(f"Environment variable '{env_var}' is not set for EDA profile '{profile}'.")
        return get_database_url(override=resolved)
    return get_database_url(None)


def _build_summary_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# EDA Run Summary",
        "",
        f"- run_id: `{manifest['run_id']}`",
        f"- generated_at: `{manifest['generated_at']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        "",
        "## Datasets",
    ]
    for ds in manifest["datasets"]:
        silver_note = ""
        if ds.get("silver"):
            silver_note = (
                f" silver_rows={ds['silver']['rows']} "
                f"silver_valid_contract={ds['silver']['contract']['valid']}"
            )
        lines.extend(
            [
                f"- `{ds['name']}`: rows={ds['rows']} format={ds['format']} valid_contract={ds['contract']['valid']}{silver_note}",
                f"  - path: `{ds['path']}`",
                f"  - duplicate_rate: {ds['quality']['duplicate_rate']:.4f}",
            ]
        )
    return "\n".join(lines)


def run_plan(
    *,
    plan_path: Path | str,
    output_root: Path | str = "eda/extracts/bronze",
    db_url: str | None = None,
    run_id: str | None = None,
    profile: str | None = None,
    profiles_path: Path | str = "eda/config/profiles.yaml",
    registry_path: Path | str = "eda/metadata/runs.jsonl",
    build_silver: bool = False,
    silver_output_root: Path | str = "eda/extracts/silver",
) -> dict[str, Any]:
    requests = load_plan(plan_path)
    resolved_db = resolve_db_url(db_url=db_url, profile=profile, profiles_path=profiles_path)
    out_dir = build_run_dir(output_root, run_id=run_id)
    silver_dir: Path | None = None
    if build_silver:
        silver_dir = build_run_dir(silver_output_root, run_id=out_dir.name)

    datasets_manifest: list[dict[str, Any]] = []
    quality_summary: dict[str, dict[str, Any]] = {}
    silver_quality_summary: dict[str, dict[str, Any]] = {}

    for req in requests:
        rows, next_cursor = fetch_dataset(
            request=LoadRequest(
                dataset=req.name,  # type: ignore[arg-type]
                limit=req.limit,
                since=req.since,
                all_pages=req.all_pages,
                fmt=req.fmt,  # type: ignore[arg-type]
            ),
            db_url=resolved_db,
        )

        out_path = out_dir / f"{req.name}.{req.fmt}"
        write_extract(rows=rows, path=out_path, fmt=req.fmt)  # type: ignore[arg-type]
        contract: ContractValidationResult = validate_dataset_rows(req.name, rows)  # type: ignore[arg-type]
        quality = build_quality_scorecard(req.name, rows)
        quality_summary[req.name] = quality
        silver_payload = None
        if build_silver and silver_dir is not None:
            silver_rows = build_silver_dataset(req.name, rows)
            silver_path = silver_dir / f"{req.name}.{req.fmt}"
            write_extract(rows=silver_rows, path=silver_path, fmt=req.fmt)  # type: ignore[arg-type]
            silver_contract: ContractValidationResult = validate_silver_rows(req.name, silver_rows)  # type: ignore[arg-type]
            silver_quality = build_quality_scorecard(req.name, silver_rows)
            silver_quality_summary[req.name] = silver_quality
            silver_payload = {
                "rows": len(silver_rows),
                "path": str(silver_path),
                "contract": {
                    "valid": silver_contract.valid,
                    "rows_checked": silver_contract.rows_checked,
                    "missing_required_columns": silver_contract.missing_required_columns,
                    "row_missing_required_count": silver_contract.row_missing_required_count,
                },
                "quality": silver_quality,
            }

        datasets_manifest.append(
            {
                "name": req.name,
                "rows": len(rows),
                "format": req.fmt,
                "path": str(out_path),
                "next_cursor": next_cursor,
                "all_pages": req.all_pages,
                "since": req.since,
                "contract": {
                    "valid": contract.valid,
                    "rows_checked": contract.rows_checked,
                    "missing_required_columns": contract.missing_required_columns,
                    "row_missing_required_count": contract.row_missing_required_count,
                },
                "quality": quality,
                "silver": silver_payload,
            }
        )

    manifest = {
        "run_id": out_dir.name,
        "plan_path": str(Path(plan_path).resolve()),
        "output_dir": str(out_dir),
        "silver_output_dir": str(silver_dir) if silver_dir else None,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "lineage": build_lineage(db_url=resolved_db, plan_path=str(Path(plan_path).resolve())),
        "datasets": datasets_manifest,
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "quality_report.json").write_text(json.dumps(quality_summary, indent=2), encoding="utf-8")
    if build_silver and silver_dir is not None:
        (silver_dir / "quality_report.json").write_text(
            json.dumps(silver_quality_summary, indent=2), encoding="utf-8"
        )

    report_dir = Path("eda/reports/latest").resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.md").write_text(_build_summary_markdown(manifest), encoding="utf-8")

    append_run_registry(manifest, registry_path=Path(registry_path).resolve())
    return manifest
