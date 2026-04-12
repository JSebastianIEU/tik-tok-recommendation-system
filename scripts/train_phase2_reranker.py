#!/usr/bin/env python3
"""Train Phase 2 learned reranker artifacts on top of a Phase 1 bundle.

Default production recipe (recommended):
  1) **Bootstrap** pairwise rows from the training datamart (weak supervision from metrics; cold-start).
  2) **Augment** with **implicit feedback** from Postgres (rec_request_events, rec_served_outputs,
     rec_ui_feedback_events) when ``--db-url`` is set — saves, relevance taps, etc.
  3) **Optional** labeling-session JSON files for extra explicit pairs.

Rows are concatenated in that order (bootstrap → feedback → labeling). All sources are optional
except at least one must yield supervision (see CLI validation).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scraper.db.client import get_connection
from src.recommendation.learning.artifacts import ArtifactRegistry
from src.recommendation.learning.feedback_pairwise import (
    group_rows_by_objective,
    materialize_pairwise_rows,
    summarize_feedback_training_support,
)
from src.recommendation.learning.labeling_pairwise import (
    materialize_labeling_session_rows,
)
from src.recommendation.learning.bootstrap_pairwise import (
    materialize_datamart_bootstrap_rows,
)
from src.recommendation.learning.learned_reranker import (
    LEARNED_RERANKER_ID,
    LEARNED_RERANKER_LABEL_POLICY_VERSION,
    LEARNED_RERANKER_VERSION,
    LearnedPairwiseReranker,
    rows_to_json_ready,
    summarize_pairwise_rows,
)

logger = logging.getLogger(__name__)

PHASE2_ROW_ORDER = ("bootstrap", "feedback", "labeling")


def _resolve_bundle_dir(bundle_ref: Path) -> Path:
    if bundle_ref.is_file() and not bundle_ref.name.endswith(".json"):
        resolved = bundle_ref.read_text(encoding="utf-8").strip()
        if resolved:
            return Path(resolved)
    return bundle_ref.resolve()


def _link_latest(artifact_root: Path, bundle_dir: Path) -> None:
    latest_link = artifact_root / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        if latest_link.is_symlink() or latest_link.is_file():
            latest_link.unlink()
        else:
            shutil.rmtree(latest_link)
    try:
        latest_link.symlink_to(bundle_dir.resolve())
    except OSError:
        latest_link.write_text(str(bundle_dir.resolve()), encoding="utf-8")


def _read_manifest(bundle_dir: Path) -> Dict[str, Any]:
    return json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(bundle_dir: Path, manifest: Dict[str, Any]) -> None:
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _fetch_requests(
    conn,
    *,
    objectives: Sequence[str],
    limit_requests: int,
    include_synthetic: bool,
    include_injected_failures: bool,
) -> List[Dict[str, Any]]:
    scoped_objectives = list(objectives)
    with conn.cursor() as cursor:
        cursor.execute(
            """
SELECT
  request_id::text,
  objective_effective,
  served_at,
  fallback_mode,
  traffic_class,
  is_synthetic,
  injected_failure
FROM rec_request_events
WHERE objective_effective = ANY(%s::text[])
  AND fallback_mode = FALSE
  AND (%s OR COALESCE(is_synthetic, FALSE) = FALSE)
  AND (%s OR COALESCE(injected_failure, FALSE) = FALSE)
ORDER BY served_at DESC
LIMIT %s
""",
            (
                scoped_objectives,
                bool(include_synthetic),
                bool(include_injected_failures),
                int(limit_requests),
            ),
        )
        rows = cursor.fetchall()
    return [
        {
            "request_id": str(row[0]),
            "objective_effective": str(row[1]),
            "served_at": row[2],
            "fallback_mode": bool(row[3]),
            "traffic_class": str(row[4]),
            "is_synthetic": bool(row[5]),
            "injected_failure": bool(row[6]),
        }
        for row in rows
    ]


def _fetch_served_outputs(conn, request_ids: Sequence[str]) -> List[Dict[str, Any]]:
    scoped_request_ids = list(request_ids)
    if not scoped_request_ids:
        return []
    with conn.cursor() as cursor:
        cursor.execute(
            """
SELECT request_id::text, candidate_id, rank, score, metadata
FROM rec_served_outputs
WHERE request_id = ANY(%s::uuid[])
ORDER BY request_id ASC, rank ASC
""",
            (scoped_request_ids,),
        )
        rows = cursor.fetchall()
    return [
        {
            "request_id": str(row[0]),
            "candidate_id": str(row[1]),
            "rank": int(row[2]),
            "score": float(row[3]),
            "metadata": dict(row[4] or {}),
        }
        for row in rows
    ]


def _fetch_feedback_events(conn, request_ids: Sequence[str]) -> List[Dict[str, Any]]:
    scoped_request_ids = list(request_ids)
    if not scoped_request_ids:
        return []
    with conn.cursor() as cursor:
        cursor.execute(
            """
SELECT
  request_id::text,
  event_name,
  entity_type,
  entity_id,
  section,
  rank,
  objective_effective,
  signal_strength,
  label_direction,
  metadata,
  created_at
FROM rec_ui_feedback_events
WHERE request_id = ANY(%s::uuid[])
ORDER BY created_at ASC
""",
            (scoped_request_ids,),
        )
        rows = cursor.fetchall()
    return [
        {
            "request_id": str(row[0]),
            "event_name": str(row[1]),
            "entity_type": str(row[2]),
            "entity_id": str(row[3]) if row[3] is not None else None,
            "section": str(row[4]),
            "rank": int(row[5]) if row[5] is not None else None,
            "objective_effective": str(row[6]),
            "signal_strength": str(row[7]),
            "label_direction": str(row[8]),
            "metadata": dict(row[9] or {}),
            "created_at": row[10],
        }
        for row in rows
    ]


def train_phase2_reranker(
    *,
    db_url: str,
    base_bundle_dir: Path,
    artifact_root: Path,
    objectives: Sequence[str],
    run_name: str,
    limit_requests: int,
    min_pairs_per_objective: int,
    include_synthetic: bool,
    include_injected_failures: bool,
    labeling_session_jsons: Optional[Sequence[Path]],
    bootstrap_datamart_json: Optional[Path],
    bootstrap_target_source: Optional[str],
    bootstrap_include_neutral_pairs: bool,
    feedback_max_served_rank: Optional[int],
    update_latest: bool,
) -> Dict[str, Any]:
    registry = ArtifactRegistry(artifact_root)
    output_bundle_dir = registry.create_bundle_dir(run_name=run_name)
    shutil.copytree(base_bundle_dir, output_bundle_dir, dirs_exist_ok=True)

    feedback_requests: List[Dict[str, Any]] = []
    feedback_served_outputs: List[Dict[str, Any]] = []
    feedback_events: List[Dict[str, Any]] = []
    feedback_db_error: Optional[str] = None
    if str(db_url).strip():
        try:
            with get_connection(db_url) as conn:
                feedback_requests = _fetch_requests(
                    conn,
                    objectives=objectives,
                    limit_requests=limit_requests,
                    include_synthetic=include_synthetic,
                    include_injected_failures=include_injected_failures,
                )
                request_ids = [str(row["request_id"]) for row in feedback_requests]
                feedback_served_outputs = _fetch_served_outputs(conn, request_ids)
                feedback_events = _fetch_feedback_events(conn, request_ids)
        except Exception as exc:
            feedback_db_error = str(exc)
            logger.warning(
                "Phase 2: could not load implicit feedback from DB (%s). Continuing with bootstrap/labeling only.",
                feedback_db_error,
            )

    feedback_support_summary = summarize_feedback_training_support(
        requests=feedback_requests,
        served_outputs=feedback_served_outputs,
        feedback_events=feedback_events,
        objectives=objectives,
        max_served_rank=feedback_max_served_rank,
    )
    feedback_pairwise_rows = materialize_pairwise_rows(
        requests=feedback_requests,
        served_outputs=feedback_served_outputs,
        feedback_events=feedback_events,
        objectives=objectives,
        max_served_rank=feedback_max_served_rank,
    )
    labeling_pairwise_rows = materialize_labeling_session_rows(
        session_json_paths=list(labeling_session_jsons or []),
        bundle_dir=base_bundle_dir,
        objectives=objectives,
    )
    bootstrap_pairwise_rows = []
    if bootstrap_datamart_json is not None:
        with open(bootstrap_datamart_json, "r", encoding="utf-8") as f:
            datamart_payload = json.load(f)
        bootstrap_pairwise_rows = materialize_datamart_bootstrap_rows(
            datamart=datamart_payload,
            bundle_dir=base_bundle_dir,
            objectives=objectives,
            target_source=bootstrap_target_source,
            include_neutral_pairs=bootstrap_include_neutral_pairs,
        )

    # Bootstrap first (dense cold-start signal), then implicit DB feedback, then optional labeling JSON.
    pairwise_rows = bootstrap_pairwise_rows + feedback_pairwise_rows + labeling_pairwise_rows
    rows_by_objective = group_rows_by_objective(pairwise_rows)

    diagnostics_dir = output_bundle_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    feedback_summary = summarize_pairwise_rows(feedback_pairwise_rows)
    labeling_summary = summarize_pairwise_rows(labeling_pairwise_rows)
    bootstrap_summary = summarize_pairwise_rows(bootstrap_pairwise_rows)
    objective_reports: Dict[str, Any] = {}
    trained_objectives: List[str] = []
    feedback_by_objective = dict(feedback_summary.get("by_objective") or {})
    labeling_by_objective = dict(labeling_summary.get("by_objective") or {})
    bootstrap_by_objective = dict(bootstrap_summary.get("by_objective") or {})
    for objective in objectives:
        objective_rows = rows_by_objective.get(objective, [])
        objective_summary = summarize_pairwise_rows(objective_rows)
        training_source_mix = {
            "bootstrap_pair_count": int(bootstrap_by_objective.get(objective, 0)),
            "feedback_pair_count": int(feedback_by_objective.get(objective, 0)),
            "labeling_pair_count": int(labeling_by_objective.get(objective, 0)),
        }
        objective_report: Dict[str, Any] = {
            "objective": objective,
            "training_pairs": objective_summary,
            "training_source_mix": training_source_mix,
            "feedback_support": feedback_support_summary.get(objective, {}),
            "status": "skipped",
            "reason": None,
        }
        if len(objective_rows) < max(1, int(min_pairs_per_objective)):
            objective_report["reason"] = "insufficient_pair_rows"
            objective_reports[objective] = objective_report
            continue

        reranker = LearnedPairwiseReranker.train(objective=objective, rows=objective_rows)
        output_dir = output_bundle_dir / "rankers" / objective / "learned_reranker"
        reranker.save(output_dir)

        dataset_path = diagnostics_dir / f"phase2_pairwise_{objective}.json"
        dataset_path.write_text(
            json.dumps(
                {
                    "summary": objective_summary,
                    "rows": rows_to_json_ready(objective_rows),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        objective_report["status"] = "trained"
        objective_report["artifact_path"] = str(output_dir.relative_to(output_bundle_dir))
        objective_report["dataset_path"] = str(dataset_path.relative_to(output_bundle_dir))
        objective_report["train_summary"] = reranker.train_summary
        objective_reports[objective] = objective_report
        trained_objectives.append(objective)

    manifest = _read_manifest(output_bundle_dir)
    manifest["phase2_learned_reranker"] = {
        "version": LEARNED_RERANKER_VERSION,
        "ranker_id": LEARNED_RERANKER_ID,
        "label_policy_version": LEARNED_RERANKER_LABEL_POLICY_VERSION,
        "base_bundle_dir": str(base_bundle_dir),
        "training_composition_policy": {
            "row_concat_order": list(PHASE2_ROW_ORDER),
            "description": (
                "Pairwise training rows are built as: datamart bootstrap pairs, then implicit "
                "UI feedback pairs from Postgres (if available), then optional labeling-session JSON."
            ),
        },
        "feedback_db_error": feedback_db_error,
        "feedback_pairwise_summary": feedback_summary,
        "feedback_training_support": feedback_support_summary,
        "labeling_pairwise_summary": labeling_summary,
        "bootstrap_pairwise_summary": bootstrap_summary,
        "trained_objectives": trained_objectives,
        "objective_reports": objective_reports,
        "feedback_max_served_rank": (
            None if feedback_max_served_rank is None else int(feedback_max_served_rank)
        ),
    }
    _write_manifest(output_bundle_dir, manifest)

    if update_latest:
        _link_latest(artifact_root, output_bundle_dir)

    return {
        "bundle_dir": str(output_bundle_dir),
        "base_bundle_dir": str(base_bundle_dir),
        "trained_objectives": trained_objectives,
        "objective_reports": objective_reports,
        "feedback_db_error": feedback_db_error,
        "feedback_pairwise_summary": feedback_summary,
        "feedback_training_support": feedback_support_summary,
        "labeling_pairwise_summary": labeling_summary,
        "bootstrap_pairwise_summary": bootstrap_summary,
        "pairwise_summary": summarize_pairwise_rows(pairwise_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-url",
        default=os.getenv("DATABASE_URL", ""),
        help="Postgres connection string. Defaults to DATABASE_URL.",
    )
    parser.add_argument(
        "--base-bundle-dir",
        type=Path,
        default=Path("artifacts/recommender/latest"),
        help="Existing recommender bundle to clone and extend.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/recommender"),
        help="Artifact root for the new bundle.",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["reach", "engagement", "conversion"],
        help="Objectives to train.",
    )
    parser.add_argument(
        "--run-name",
        default="phase2-feedback-reranker",
        help="Suffix for the new artifact bundle directory.",
    )
    parser.add_argument(
        "--limit-requests",
        type=int,
        default=2000,
        help="Maximum recent requests to inspect when building training pairs.",
    )
    parser.add_argument(
        "--min-pairs-per-objective",
        type=int,
        default=12,
        help="Minimum pair rows required before training an objective-specific model.",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Include synthetic validation traffic in the training pull.",
    )
    parser.add_argument(
        "--include-injected-failures",
        action="store_true",
        help="Include injected-failure traffic in the training pull.",
    )
    parser.add_argument(
        "--labeling-session-json",
        type=Path,
        action="append",
        default=None,
        help="Optional local labeling-session JSON path(s) to convert into pairwise supervision.",
    )
    parser.add_argument(
        "--bootstrap-datamart-json",
        type=Path,
        default=None,
        help=(
            "Training datamart JSON for bootstrap pairwise supervision (primary volume; weak labels). "
            "Combined with --db-url feedback and optional --labeling-session-json."
        ),
    )
    parser.add_argument(
        "--bootstrap-target-source",
        type=str,
        default=None,
        choices=["scalar_v1", "trajectory_v2_composite"],
        help="Optional datamart pair target source to require for bootstrap supervision. Defaults to the datamart config.",
    )
    parser.add_argument(
        "--bootstrap-include-neutral-pairs",
        action="store_true",
        help="Allow weaker bootstrap pairs against relevance label 1 rows. Off by default for conservative bootstrap training.",
    )
    parser.add_argument(
        "--feedback-max-served-rank",
        type=int,
        default=10,
        help="Only use feedback from served outputs up to this rank when building feedback-derived pair rows. Use <=0 to disable the rank band.",
    )
    parser.add_argument(
        "--update-latest",
        action="store_true",
        help="Update artifacts/recommender/latest after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        not str(args.db_url).strip()
        and args.bootstrap_datamart_json is None
        and not list(args.labeling_session_json or [])
    ):
        raise SystemExit(
            "At least one supervision source is required: DATABASE_URL/--db-url, "
            "--labeling-session-json, or --bootstrap-datamart-json."
        )
    base_bundle_dir = _resolve_bundle_dir(args.base_bundle_dir)
    if not base_bundle_dir.exists():
        raise SystemExit(f"Base bundle directory not found: {base_bundle_dir}")

    result = train_phase2_reranker(
        db_url=str(args.db_url).strip(),
        base_bundle_dir=base_bundle_dir,
        artifact_root=args.artifact_root,
        objectives=[str(item) for item in list(args.objectives or []) if str(item).strip()],
        run_name=str(args.run_name).strip() or "phase2-feedback-reranker",
        limit_requests=max(1, int(args.limit_requests)),
        min_pairs_per_objective=max(1, int(args.min_pairs_per_objective)),
        include_synthetic=bool(args.include_synthetic),
        include_injected_failures=bool(args.include_injected_failures),
        labeling_session_jsons=list(args.labeling_session_json or []),
        bootstrap_datamart_json=args.bootstrap_datamart_json,
        bootstrap_target_source=(
            str(args.bootstrap_target_source).strip() if args.bootstrap_target_source else None
        ),
        bootstrap_include_neutral_pairs=bool(args.bootstrap_include_neutral_pairs),
        feedback_max_served_rank=(
            None if int(args.feedback_max_served_rank) <= 0 else int(args.feedback_max_served_rank)
        ),
        update_latest=bool(args.update_latest),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
