#!/usr/bin/env python3
"""Fast-refresh serving artifacts from Supabase without full model retraining.

This script is intended to run after scraping/comment enrichment jobs complete.
It refreshes the serving-facing corpus and retriever layer while keeping the
existing ranker artifacts in place.

Flow:
  1. Export the latest canonical contract bundle from Supabase
  2. Rebuild the serving datamart from that manifest
  3. Rebuild the retriever artifact from the refreshed datamart
  4. Clone the currently served recommender bundle and replace only the retriever
  5. Optionally promote the refreshed bundle to the `latest` symlink

The Node gateway picks up the refreshed contract bundle automatically because the
artifact-backed corpus loader watches bundle file mtimes. The Python service can
pick up the promoted bundle without a manual restart because
`src/recommendation/service.py` now auto-reloads when the bundle target changes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._utils import parse_iso_datetime, to_jsonable
from scripts.export_db_contract_bundle import export_bundle_from_db
from scripts.build_training_datamart import _stream_write_datamart
from src.recommendation import (
    BuildTrainingDataMartConfig,
    build_contract_manifest,
    build_training_data_mart_from_manifest,
)


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_bundle_dir(bundle_ref: Path) -> Path:
    if bundle_ref.is_file() and not bundle_ref.is_dir():
        raw = bundle_ref.read_text(encoding="utf-8").strip()
        if raw:
            return Path(raw).resolve()
    return bundle_ref.resolve()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _promote_symlink(link_path: Path, target_dir: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() and not link_path.is_symlink():
        raise RuntimeError(
            f"Refusing to replace non-symlink path during promotion: {link_path}"
        )
    temp_link = link_path.parent / f".{link_path.name}.tmp"
    if temp_link.exists() or temp_link.is_symlink():
        temp_link.unlink()
    temp_link.symlink_to(target_dir.resolve())
    os.replace(temp_link, link_path)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _refresh_manifest(
    manifest_path: Path,
    *,
    retriever_manifest: Dict[str, Any],
    contract_manifest: Dict[str, Any],
    datamart_json: Path,
    base_bundle_dir: Path,
    contract_bundle_json: Path,
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["retriever"] = {
        key: value
        for key, value in retriever_manifest.items()
        if key
        not in {
            "row_ids",
            "row_times",
            "row_metadata",
            "row_payload",
        }
    }
    graph_bundle_id = retriever_manifest.get("graph_bundle_id")
    graph_version = retriever_manifest.get("graph_version")
    if graph_bundle_id or graph_version:
        payload["graph"] = {
            **(payload.get("graph") if isinstance(payload.get("graph"), dict) else {}),
            **({"graph_bundle_id": graph_bundle_id} if graph_bundle_id else {}),
            **({"graph_version": graph_version} if graph_version else {}),
        }
    trajectory_manifest_id = retriever_manifest.get("trajectory_manifest_id")
    trajectory_version = retriever_manifest.get("trajectory_version")
    if trajectory_manifest_id or trajectory_version:
        payload["trajectory"] = {
            **(
                payload.get("trajectory")
                if isinstance(payload.get("trajectory"), dict)
                else {}
            ),
            **(
                {"trajectory_manifest_id": trajectory_manifest_id}
                if trajectory_manifest_id
                else {}
            ),
            **({"trajectory_version": trajectory_version} if trajectory_version else {}),
        }
    payload["serving_refresh"] = {
        "refresh_type": "retriever_fast_refresh.v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "base_bundle_dir": str(base_bundle_dir),
        "contract_manifest_id": contract_manifest.get("manifest_id"),
        "contract_manifest_dir": contract_manifest.get("manifest_dir"),
        "contract_bundle_json": str(contract_bundle_json),
        "datamart_json": str(datamart_json),
    }
    _write_json(manifest_path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-url",
        default=os.getenv("DATABASE_URL", ""),
        help="Postgres connection string. Defaults to DATABASE_URL.",
    )
    parser.add_argument(
        "--as-of-time",
        default=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        help="As-of timestamp in ISO-8601.",
    )
    parser.add_argument(
        "--base-bundle-dir",
        type=Path,
        default=Path("artifacts/recommender_real/latest"),
        help="Currently served bundle dir or symlink.",
    )
    parser.add_argument(
        "--bundle-output-root",
        type=Path,
        default=Path("artifacts/recommender_real"),
        help="Where refreshed bundle directories are written.",
    )
    parser.add_argument(
        "--contract-manifest-root",
        type=Path,
        default=Path("artifacts/contracts"),
        help="Root for refreshed contract manifests.",
    )
    parser.add_argument(
        "--contract-bundle-json",
        type=Path,
        default=Path("artifacts/contracts/latest_supabase_bundle.json"),
        help="Path to write the refreshed canonical contract bundle JSON.",
    )
    parser.add_argument(
        "--datamart-json",
        type=Path,
        default=Path("artifacts/datamart/supabase_latest_datamart.json"),
        help="Path to write the refreshed datamart JSON.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used to run helper scripts.",
    )
    parser.add_argument(
        "--promote-latest",
        dest="promote_latest",
        action="store_true",
        help="Update the latest bundle symlink after refresh completes.",
    )
    parser.add_argument(
        "--no-promote-latest",
        dest="promote_latest",
        action="store_false",
        help="Build the refreshed bundle but do not swap the latest symlink.",
    )
    parser.set_defaults(promote_latest=True)
    args = parser.parse_args()

    if not str(args.db_url).strip():
        raise SystemExit("DATABASE_URL or --db-url is required.")

    as_of_time = parse_iso_datetime(args.as_of_time)
    run_slug = _timestamp_slug()
    base_bundle_dir = _resolve_bundle_dir(args.base_bundle_dir)
    if not base_bundle_dir.exists():
        raise SystemExit(f"Base bundle directory not found: {base_bundle_dir}")

    bundle = export_bundle_from_db(db_url=str(args.db_url).strip(), as_of_time=as_of_time)
    contract_manifest = build_contract_manifest(
        bundle=bundle,
        manifest_root=args.contract_manifest_root,
        source_file_hashes={"source": "refresh_serving_bundle"},
        as_of_time=as_of_time,
    )
    args.contract_bundle_json.parent.mkdir(parents=True, exist_ok=True)
    args.contract_bundle_json.write_text(
        json.dumps(bundle.model_dump(mode="python"), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    datamart = build_training_data_mart_from_manifest(
        manifest_ref=Path(str(contract_manifest["manifest_dir"])),
        config=BuildTrainingDataMartConfig(
            track="post_publication",
            min_history_hours=24,
            label_window_hours=72,
            pair_objective="engagement",
            pair_target_source="scalar_v1",
            enable_trajectory_labels=True,
            trajectory_windows_hours=(6, 24, 96),
        ),
    )
    args.datamart_json.parent.mkdir(parents=True, exist_ok=True)
    _stream_write_datamart(datamart, args.datamart_json)

    refreshed_bundle_dir = args.bundle_output_root / f"{run_slug}-serving-refresh"
    if refreshed_bundle_dir.exists():
        shutil.rmtree(refreshed_bundle_dir)
    shutil.copytree(base_bundle_dir, refreshed_bundle_dir)

    retriever_dir = refreshed_bundle_dir / "retriever"
    if retriever_dir.exists():
        shutil.rmtree(retriever_dir)
    _run(
        [
            args.python_bin,
            "scripts/build_retriever_index.py",
            str(args.datamart_json),
            "--output-dir",
            str(retriever_dir),
        ]
    )

    retriever_manifest_path = retriever_dir / "manifest.json"
    retriever_manifest = json.loads(retriever_manifest_path.read_text(encoding="utf-8"))
    _refresh_manifest(
        refreshed_bundle_dir / "manifest.json",
        retriever_manifest=retriever_manifest,
        contract_manifest=contract_manifest,
        datamart_json=args.datamart_json,
        base_bundle_dir=base_bundle_dir,
        contract_bundle_json=args.contract_bundle_json,
    )

    if args.promote_latest:
        _promote_symlink(args.base_bundle_dir, refreshed_bundle_dir)

    print(
        json.dumps(
            {
                "status": "ok",
                "refresh_type": "retriever_fast_refresh.v1",
                "contract_manifest_id": contract_manifest["manifest_id"],
                "contract_manifest_dir": contract_manifest["manifest_dir"],
                "contract_bundle_json": str(args.contract_bundle_json),
                "datamart_json": str(args.datamart_json),
                "base_bundle_dir": str(base_bundle_dir),
                "refreshed_bundle_dir": str(refreshed_bundle_dir),
                "promoted_latest": bool(args.promote_latest),
                "entity_counts": {
                    "authors": len(bundle.authors),
                    "videos": len(bundle.videos),
                    "video_snapshots": len(bundle.video_snapshots),
                    "comments": len(bundle.comments),
                    "comment_snapshots": len(bundle.comment_snapshots),
                },
                "datamart_stats": datamart.get("stats", {}),
                "retriever_artifact_version": retriever_manifest.get("artifact_version"),
                "retriever_index_cutoff_time": retriever_manifest.get("index_cutoff_time"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
