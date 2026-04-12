#!/usr/bin/env python3
"""Full training pipeline: Supabase DB -> Contract Bundle -> DataMart -> all components.

Orchestrates:
  1. Backfill video_hashtags bridge table (if needed)
  2. Export canonical contract bundle from Supabase
  3. Build training data mart (features, labels, pairs)
  4. Build comment intelligence snapshots
  5. Train Phase 1 recommender (dense retriever + ranker + graph + trajectory)
  6. Train Phase 2 learned reranker (datamart bootstrap + optional implicit DB feedback)
  7. Fit fabric score calibration from eval split
  8. Validate all artifacts

Usage:
  python scripts/train_full_pipeline.py --db-url "$DATABASE_URL"
  python scripts/train_full_pipeline.py  # uses DATABASE_URL env var
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._utils import to_jsonable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_full_pipeline")

# --- Pipeline defaults ---
DEFAULT_RETRIEVE_K = 200
DEFAULT_MAX_AGE_DAYS = 180
DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GRAPH_EMBEDDING_DIM = 32
DEFAULT_TRAJECTORY_EMBEDDING_DIM = 16


def _elapsed(start: float) -> str:
    s = time.time() - start
    if s < 60:
        return f"{s:.1f}s"
    return f"{s/60:.1f}min"


# ---------------------------------------------------------------------------
# Step 1: Backfill hashtag bridge table
# ---------------------------------------------------------------------------
def step_backfill_hashtags(db_url: str) -> Dict[str, Any]:
    logger.info("Step 1/7: Backfill video_hashtags bridge table")
    from scripts.backfill_video_hashtags import backfill, print_bridge_stats

    print_bridge_stats(db_url)
    t0 = time.time()
    stats = backfill(db_url, batch_size=5000)
    logger.info("Backfill completed in %s — %s links created", _elapsed(t0), f"{stats['links_created']:,}")
    print_bridge_stats(db_url)
    return stats


# ---------------------------------------------------------------------------
# Step 2: Export canonical contract bundle from Supabase
# ---------------------------------------------------------------------------
def step_export_bundle(
    db_url: str,
    as_of_time: datetime,
    manifest_root: Path,
    bundle_json_path: Path,
) -> Dict[str, Any]:
    logger.info("Step 2/7: Export canonical contract bundle from Supabase")
    from scripts.export_db_contract_bundle import export_bundle_from_db
    from src.recommendation import build_contract_manifest

    t0 = time.time()
    bundle = export_bundle_from_db(db_url=db_url, as_of_time=as_of_time)
    manifest = build_contract_manifest(
        bundle=bundle,
        manifest_root=manifest_root,
        source_file_hashes={"source": "train_full_pipeline"},
        as_of_time=as_of_time,
    )

    bundle_json_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_json_path.write_text(
        json.dumps(bundle.model_dump(mode="python"), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    counts = {
        "authors": len(bundle.authors),
        "videos": len(bundle.videos),
        "video_snapshots": len(bundle.video_snapshots),
        "comments": len(bundle.comments),
        "comment_snapshots": len(bundle.comment_snapshots),
    }
    logger.info("Exported in %s", _elapsed(t0))
    for k, v in counts.items():
        logger.info("  %s: %s", k, f"{v:,}")
    logger.info("Manifest: %s", manifest['manifest_dir'])
    logger.info("Bundle JSON: %s", bundle_json_path)

    return {
        "manifest": manifest,
        "bundle_json": str(bundle_json_path),
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# Step 3: Build comment intelligence snapshots
# ---------------------------------------------------------------------------
def step_build_comment_intelligence(
    bundle_json_path: Path,
    as_of_time: datetime,
    contract_manifest_root: Path,
    output_root: Path,
) -> Dict[str, Any]:
    logger.info("Step 3/7: Build comment intelligence snapshots")
    from pydantic import ValidationError

    from src.recommendation import (
        CanonicalDatasetBundle,
        CommentIntelligenceConfig,
        build_comment_intelligence_snapshot_manifest,
        build_contract_manifest,
    )

    t0 = time.time()

    # Same payload as Step 2 export — parse as the contract model directly. Do not
    # round-trip through JSONL: raw JSONL validation expects per-line shapes (e.g.
    # video_id) that author rows do not satisfy.
    bundle_data = json.loads(bundle_json_path.read_text(encoding="utf-8"))
    try:
        bundle = CanonicalDatasetBundle.model_validate(bundle_data)
    except ValidationError as exc:
        errs = exc.errors()[:10]
        logger.warning("Bundle parse failed for comment intelligence: %s", errs)
        return {"status": "validation_failed", "errors": errs}
    contract_manifest = build_contract_manifest(
        bundle=bundle,
        manifest_root=contract_manifest_root,
        source_file_hashes={"source": "train_full_pipeline"},
        as_of_time=as_of_time,
    )

    config = CommentIntelligenceConfig(
        early_window_hours=24,
        late_window_hours=96,
        min_comments_for_stable=3,
    )

    payload = build_comment_intelligence_snapshot_manifest(
        bundle=bundle,
        as_of_time=as_of_time,
        output_root=output_root,
        mode="full",
        config=config,
    )

    logger.info("Comment intelligence built in %s", _elapsed(t0))
    logger.info("Snapshots: %s", payload.get('stats', {}).get('total_snapshots', 'N/A'))
    logger.info("Output: %s", output_root)

    return {
        "status": "ok",
        "payload": payload,
        "contract_manifest": contract_manifest,
    }


# ---------------------------------------------------------------------------
# Step 4: Build training data mart
# ---------------------------------------------------------------------------
def step_build_datamart(
    manifest_path: Path,
    as_of_time: datetime,
    output_json: Path,
    comment_feature_manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    logger.info("Step 4/7: Build training data mart")
    from src.recommendation import (
        BuildTrainingDataMartConfig,
        build_training_data_mart_from_manifest,
    )

    t0 = time.time()

    config_kwargs: Dict[str, Any] = {}
    if comment_feature_manifest_path:
        config_kwargs["comment_feature_manifest_path"] = comment_feature_manifest_path

    config = BuildTrainingDataMartConfig(
        track="post_publication",
        min_history_hours=24,
        label_window_hours=72,
        pair_objective="engagement",
        pair_target_source="scalar_v1",
        enable_trajectory_labels=True,
        trajectory_windows_hours=(6, 24, 96),
        **config_kwargs,
    )

    mart = build_training_data_mart_from_manifest(
        manifest_ref=manifest_path,
        config=config,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(to_jsonable(mart), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    stats = mart.get("stats", {})
    logger.info("DataMart built in %s", _elapsed(t0))
    logger.info("Rows: %s, Pair rows: %s", stats.get('rows_total', 'N/A'), stats.get('pair_rows_total', 'N/A'))
    logger.info("Output: %s", output_json)

    return {"status": "ok", "stats": stats, "output": str(output_json)}


# ---------------------------------------------------------------------------
# Step 5: Train Phase 1 recommender
# ---------------------------------------------------------------------------
def step_train_recommender(
    datamart_json: Path,
    artifact_root: Path,
) -> Dict[str, Any]:
    logger.info("Step 5/7: Train Phase 1 recommender")
    from src.recommendation.learning import (
        RecommenderTrainingConfig,
        train_recommender_from_datamart,
    )

    t0 = time.time()
    with open(datamart_json, "r", encoding="utf-8") as f:
        datamart = json.load(f)

    result = train_recommender_from_datamart(
        datamart=datamart,
        artifact_root=artifact_root,
        config=RecommenderTrainingConfig(
            objectives=["reach", "engagement", "conversion"],
            retrieve_k=DEFAULT_RETRIEVE_K,
            max_age_days=DEFAULT_MAX_AGE_DAYS,
            dense_model_name=DEFAULT_DENSE_MODEL,
            run_name="pipeline-v1",
            pair_target_source="scalar_v1",
            graph_enabled=True,
            graph_embedding_dim=DEFAULT_GRAPH_EMBEDDING_DIM,
            graph_walk_params={
                "walk_length": 12,
                "num_walks": 20,
                "context_size": 4,
                "seed": 13,
            },
            graph_weighting_params={
                "recency_half_life_days": 45.0,
                "include_creator_similarity": True,
                "creator_similarity_top_k": 5,
                "creator_similarity_min_jaccard": 0.15,
                "branch_weight": 0.10,
            },
            trajectory_enabled=True,
            trajectory_embedding_dim=DEFAULT_TRAJECTORY_EMBEDDING_DIM,
            trajectory_feature_version="trajectory_features.v2",
            trajectory_branch_weight=0.08,
            trajectory_encoder_mode="feature_only",
            contract_version=str(datamart.get("source_contract_version", "contract.v2")),
            datamart_version=str(datamart.get("version", "datamart.v1")),
        ),
    )

    # Set up latest link
    bundle_dir = Path(result["bundle_dir"])
    latest_link = artifact_root / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        if latest_link.is_symlink() or latest_link.is_file():
            latest_link.unlink()
    if not latest_link.exists():
        try:
            latest_link.symlink_to(bundle_dir.resolve())
        except OSError:
            latest_link.write_text(str(bundle_dir.resolve()), encoding="utf-8")

    logger.info("Recommender trained in %s", _elapsed(t0))
    logger.info("Bundle: %s", result['bundle_dir'])
    return result


# ---------------------------------------------------------------------------
# Step 6: Train Phase 2 learned reranker (datamart bootstrap + optional DB feedback)
# ---------------------------------------------------------------------------
def step_train_reranker(
    datamart_json: Path,
    base_bundle_dir: Path,
    artifact_root: Path,
    *,
    db_url: str,
    phase2_feedback_limit: int,
    phase2_skip_feedback: bool,
) -> Dict[str, Any]:
    logger.info("Step 6/7: Train Phase 2 learned reranker (bootstrap + optional implicit feedback)")
    from scripts.train_phase2_reranker import train_phase2_reranker

    t0 = time.time()
    feedback_url = "" if phase2_skip_feedback else db_url
    if phase2_skip_feedback:
        logger.info("Phase 2: implicit feedback from DB disabled (--skip-phase2-feedback)")
    else:
        logger.info(
            "Phase 2: merging datamart bootstrap with up to %s recent feedback requests from DB",
            phase2_feedback_limit,
        )

    result = train_phase2_reranker(
        db_url=feedback_url,
        base_bundle_dir=base_bundle_dir,
        artifact_root=artifact_root,
        objectives=["reach", "engagement", "conversion"],
        run_name="phase2-bootstrap-feedback",
        limit_requests=max(1, int(phase2_feedback_limit)),
        min_pairs_per_objective=12,
        include_synthetic=False,
        include_injected_failures=False,
        labeling_session_jsons=None,
        bootstrap_datamart_json=datamart_json,
        bootstrap_target_source="scalar_v1",
        bootstrap_include_neutral_pairs=True,
        feedback_max_served_rank=10,
        update_latest=True,
    )

    if result.get("feedback_db_error"):
        logger.warning("Phase 2 feedback DB note: %s", result["feedback_db_error"])

    logger.info("Reranker trained in %s", _elapsed(t0))
    logger.info("Bundle: %s", result['bundle_dir'])
    trained = result.get("trained_objectives", [])
    logger.info("Trained objectives: %s", trained)
    for obj, report in result.get("objective_reports", {}).items():
        status = report.get("status", "unknown")
        reason = report.get("reason", "")
        mix = report.get("training_source_mix", {}) or {}
        logger.info(
            "  %s: %s bootstrap=%s feedback=%s labeling=%s %s",
            obj,
            status,
            mix.get("bootstrap_pair_count", 0),
            mix.get("feedback_pair_count", 0),
            mix.get("labeling_pair_count", 0),
            reason,
        )
    return result


# ---------------------------------------------------------------------------
# Step 7: Fit fabric score calibration
# ---------------------------------------------------------------------------
def step_fit_calibration(
    datamart_json: Path,
    output_json: Path,
) -> Dict[str, Any]:
    logger.info("Step 7/7: Fit fabric score calibration")
    from src.recommendation.fabric import FeatureFabric

    t0 = time.time()
    with open(datamart_json, "r", encoding="utf-8") as f:
        datamart = json.load(f)

    # Build calibration observations from the datamart rows.
    # We use the engagement z-scores as targets and the row features as raw scores.
    rows = datamart.get("rows", [])
    if not rows:
        logger.info("No rows in datamart, skipping calibration.")
        return {"status": "skipped", "reason": "no_rows"}

    # Build synthetic calibration observations from feature distributions.
    # For each feature block, sample (raw_confidence, normalized_target) pairs
    # from the training rows to let the calibrator learn the mapping.
    observations: Dict[str, List[Tuple[float, float]]] = {
        "text": [],
        "structure": [],
        "audio": [],
        "visual": [],
    }

    for row in rows:
        features = row.get("features", {})
        targets_z = row.get("targets_z", {})
        # Use engagement z-score as the universal target, normalized to [0,1]
        eng_z = targets_z.get("engagement", 0.0)
        target = 1.0 / (1.0 + math.exp(-eng_z)) if isinstance(eng_z, (int, float)) else 0.5

        # Text features -> text calibrator
        text_feats = features.get("text", {})
        if text_feats.get("clarity_score") is not None:
            observations["text"].append((float(text_feats["clarity_score"]), target))

        # Structure features -> structure calibrator
        struct_feats = features.get("structure", {})
        if struct_feats.get("pacing_score") is not None:
            observations["structure"].append((float(struct_feats["pacing_score"]), target))

        # Audio features -> audio calibrator
        audio_feats = features.get("audio", {})
        if audio_feats.get("speech_ratio") is not None:
            observations["audio"].append((float(audio_feats["speech_ratio"]), target))

        # Visual features -> visual calibrator
        visual_feats = features.get("visual", {})
        if visual_feats.get("shot_change_rate") is not None:
            observations["visual"].append((float(visual_feats["shot_change_rate"]), target))

    # Filter out blocks with too few observations
    observations = {k: v for k, v in observations.items() if len(v) >= 10}

    if not observations:
        logger.info("Insufficient feature observations for calibration.")
        return {"status": "skipped", "reason": "insufficient_observations"}

    fabric = FeatureFabric()
    fitted = fabric.fit_calibrators(observations)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(fitted, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Calibration fitted in %s", _elapsed(t0))
    for block, obs in observations.items():
        logger.info("  %s: %d observations", block, len(obs))
    logger.info("Output: %s", output_json)

    return {"status": "ok", "blocks_fitted": list(observations.keys())}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--db-url",
        default=os.getenv("DATABASE_URL", ""),
        help="Postgres connection string. Defaults to DATABASE_URL.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts"),
        help="Root directory for all pipeline artifacts.",
    )
    parser.add_argument(
        "--skip-backfill",
        action="store_true",
        help="Skip hashtag bridge backfill step.",
    )
    parser.add_argument(
        "--skip-comments",
        action="store_true",
        help="Skip comment intelligence step.",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip fabric calibration step.",
    )
    parser.add_argument(
        "--as-of-time",
        type=str,
        default=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        help="As-of timestamp in ISO-8601.",
    )
    parser.add_argument(
        "--phase2-feedback-limit",
        type=int,
        default=5000,
        help=(
            "Max recommender requests to pull from DB when building Phase 2 implicit-feedback pairs "
            "(merged after datamart bootstrap rows)."
        ),
    )
    parser.add_argument(
        "--skip-phase2-feedback",
        action="store_true",
        help="Train Phase 2 from datamart bootstrap pairs only (no rec_* feedback tables).",
    )
    args = parser.parse_args()

    db_url = str(args.db_url).strip()
    if not db_url:
        raise SystemExit("DATABASE_URL or --db-url is required.")

    as_of_time = datetime.fromisoformat(args.as_of_time.replace("Z", "+00:00"))
    if as_of_time.tzinfo is None:
        as_of_time = as_of_time.replace(tzinfo=timezone.utc)

    artifact_root = args.artifact_root
    contract_root = artifact_root / "contracts"
    recommender_root = artifact_root / "recommender"
    comment_root = artifact_root / "comment_intelligence" / "features"
    calibration_output = artifact_root / "features" / "fabric_calibration.json"
    bundle_json = artifact_root / "contracts" / "bundle_export.json"
    datamart_json = artifact_root / "datamart" / "training_datamart.json"

    pipeline_start = time.time()
    results: Dict[str, Any] = {}

    logger.info("FULL TRAINING PIPELINE")
    logger.info("DB: ...%s", db_url[-30:])
    logger.info("As-of: %s", as_of_time.isoformat())
    logger.info("Artifacts: %s", artifact_root.resolve())

    # Step 1: Backfill hashtags
    if not args.skip_backfill:
        try:
            results["backfill"] = step_backfill_hashtags(db_url)
        except Exception as e:
            logger.warning("Backfill failed: %s", e, exc_info=True)
            results["backfill"] = {"status": "error", "error": str(e)}
    else:
        logger.info("[SKIPPED] Hashtag backfill")

    # Step 2: Export bundle
    try:
        results["export"] = step_export_bundle(
            db_url=db_url,
            as_of_time=as_of_time,
            manifest_root=contract_root,
            bundle_json_path=bundle_json,
        )
    except Exception as e:
        logger.error("Export failed: %s", e, exc_info=True)
        raise SystemExit(1)

    manifest_dir = Path(results["export"]["manifest"]["manifest_dir"])

    # Step 3: Comment intelligence
    comment_manifest_path = None
    if not args.skip_comments:
        try:
            ci_result = step_build_comment_intelligence(
                bundle_json_path=bundle_json,
                as_of_time=as_of_time,
                contract_manifest_root=contract_root,
                output_root=comment_root,
            )
            results["comment_intelligence"] = ci_result
            if ci_result.get("status") == "ok":
                ci_payload = ci_result.get("payload", {})
                comment_manifest_path = ci_payload.get("manifest_dir") or ci_payload.get("output_dir")
        except Exception as e:
            logger.warning("Comment intelligence failed: %s", e, exc_info=True)
            results["comment_intelligence"] = {"status": "error", "error": str(e)}
    else:
        logger.info("[SKIPPED] Comment intelligence")

    # Step 4: Build datamart
    try:
        results["datamart"] = step_build_datamart(
            manifest_path=manifest_dir,
            as_of_time=as_of_time,
            output_json=datamart_json,
            comment_feature_manifest_path=comment_manifest_path,
        )
    except Exception as e:
        logger.error("DataMart build failed: %s", e, exc_info=True)
        raise SystemExit(1)

    # Step 5: Train recommender
    try:
        results["recommender"] = step_train_recommender(
            datamart_json=datamart_json,
            artifact_root=recommender_root,
        )
    except Exception as e:
        logger.error("Recommender training failed: %s", e, exc_info=True)
        raise SystemExit(1)

    recommender_bundle_dir = Path(results["recommender"]["bundle_dir"])

    # Step 6: Train reranker
    try:
        results["reranker"] = step_train_reranker(
            datamart_json=datamart_json,
            base_bundle_dir=recommender_bundle_dir,
            artifact_root=recommender_root,
            db_url=db_url,
            phase2_feedback_limit=max(1, int(args.phase2_feedback_limit)),
            phase2_skip_feedback=bool(args.skip_phase2_feedback),
        )
    except Exception as e:
        logger.warning("Reranker training failed: %s", e, exc_info=True)
        results["reranker"] = {"status": "error", "error": str(e)}

    # Step 7: Fit calibration
    if not args.skip_calibration:
        try:
            results["calibration"] = step_fit_calibration(
                datamart_json=datamart_json,
                output_json=calibration_output,
            )
        except Exception as e:
            logger.warning("Calibration failed: %s", e, exc_info=True)
            results["calibration"] = {"status": "error", "error": str(e)}
    else:
        logger.info("[SKIPPED] Fabric calibration")

    # Final summary
    total_time = _elapsed(pipeline_start)
    logger.info("PIPELINE COMPLETE (%s)", total_time)

    summary = {
        "total_time": total_time,
        "as_of_time": as_of_time.isoformat(),
        "steps": {},
    }
    for step_name, step_result in results.items():
        if isinstance(step_result, dict):
            status = step_result.get("status", "ok")
            summary["steps"][step_name] = status
        else:
            summary["steps"][step_name] = "ok"

    for step_name, status in summary["steps"].items():
        icon = "OK" if status in ("ok", None) else status.upper()
        logger.info("[%10s] %s", icon, step_name)

    # Write pipeline report
    report_path = artifact_root / "pipeline_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report_path.write_text(
            json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Report: %s", report_path)
    except Exception:
        logger.warning("Failed to write pipeline report to %s", report_path, exc_info=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
