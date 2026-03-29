#!/usr/bin/env python3
"""Build a recommendation training data mart from JSONL input."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.recommendation import (
    BuildTrainingDataMartConfig,
    build_training_data_mart_from_manifest,
    build_training_data_mart_from_jsonl,
)


def _parse_iso_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {key: _to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(inner) for inner in value]
    return value


def _parse_csv_int_triplet(value: str) -> tuple[int, int, int]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("Expected exactly three comma-separated integers (e.g. 6,24,96).")
    out = tuple(int(item) for item in parts)
    return out[0], out[1], out[2]


def _parse_trajectory_weights_json(value: str | None) -> Dict[str, Dict[str, float]] | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("trajectory-objective-weights-json must decode to an object.")
    out: Dict[str, Dict[str, float]] = {}
    for objective, weights in parsed.items():
        if not isinstance(weights, dict):
            raise ValueError(f"trajectory weights for '{objective}' must be an object.")
        out[str(objective)] = {
            "early": float(weights.get("early", 0.0)),
            "stability": float(weights.get("stability", 0.0)),
            "late": float(weights.get("late", 0.0)),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build datamart.v1 from raw JSONL.")
    parser.add_argument(
        "input_jsonl",
        type=Path,
        nargs="?",
        default=None,
        help="Path to input JSONL dataset.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/mock/training_datamart.json"),
        help="Where to write the resulting data mart JSON.",
    )
    parser.add_argument(
        "--as-of-time",
        type=str,
        default=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        help="As-of timestamp (ISO-8601, UTC recommended).",
    )
    parser.add_argument(
        "--track",
        choices=["pre_publication", "post_publication"],
        default="pre_publication",
    )
    parser.add_argument("--min-history-hours", type=int, default=24)
    parser.add_argument("--label-window-hours", type=int, default=72)
    parser.add_argument(
        "--as-of-run-time",
        type=str,
        default=None,
        help="Optional explicit run cutoff for datamart label/snapshot visibility (ISO-8601).",
    )
    parser.add_argument(
        "--strict-timestamps",
        action="store_true",
        help="Fail if any snapshot timestamp must fallback from source data.",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Fail if contract normalization produces warnings.",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=None,
        help="Optional root folder where contract manifests are written (content-addressed).",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional manifest path/dir to replay exact as-of bundle instead of rebuilding from JSONL.",
    )
    parser.add_argument(
        "--comment-feature-manifest-path",
        type=Path,
        default=None,
        help="Optional comment-intelligence snapshot manifest path/dir.",
    )
    parser.add_argument(
        "--comment-priors-manifest-path",
        type=Path,
        default=None,
        help="Optional comment transfer priors manifest path/dir.",
    )
    parser.add_argument(
        "--pair-target-source",
        type=str,
        default="scalar_v1",
        choices=["scalar_v1", "trajectory_v2_composite"],
        help="Pair labeling source used for datamart pair_rows.",
    )
    parser.add_argument(
        "--enable-trajectory-labels",
        dest="enable_trajectory_labels",
        action="store_true",
        help="Enable additive trajectory labels and trajectory z-targets.",
    )
    parser.add_argument(
        "--disable-trajectory-labels",
        dest="enable_trajectory_labels",
        action="store_false",
        help="Disable additive trajectory labels and trajectory z-targets.",
    )
    parser.add_argument(
        "--trajectory-windows-hours",
        type=str,
        default="6,24,96",
        help="Comma-separated boundary windows in hours relative to row as_of_time.",
    )
    parser.add_argument(
        "--trajectory-objective-weights-json",
        type=str,
        default=None,
        help='Optional JSON mapping objective->weights, e.g. {"reach":{"early":0.45,"stability":0.2,"late":0.35}}',
    )
    parser.set_defaults(enable_trajectory_labels=True)
    args = parser.parse_args()

    trajectory_weights = _parse_trajectory_weights_json(args.trajectory_objective_weights_json)
    config_kwargs: Dict[str, Any] = {}
    if trajectory_weights is not None:
        config_kwargs["trajectory_objective_weights"] = trajectory_weights
    config = BuildTrainingDataMartConfig(
        track=args.track,
        min_history_hours=args.min_history_hours,
        label_window_hours=args.label_window_hours,
        pair_target_source=args.pair_target_source,
        enable_trajectory_labels=bool(args.enable_trajectory_labels),
        as_of_run_time=(
            _parse_iso_datetime(args.as_of_run_time) if args.as_of_run_time else None
        ),
        trajectory_windows_hours=_parse_csv_int_triplet(args.trajectory_windows_hours),
        comment_feature_manifest_path=(
            str(args.comment_feature_manifest_path) if args.comment_feature_manifest_path else None
        ),
        comment_priors_manifest_path=(
            str(args.comment_priors_manifest_path) if args.comment_priors_manifest_path else None
        ),
        **config_kwargs,
    )
    if args.manifest_path is not None:
        mart = build_training_data_mart_from_manifest(
            manifest_ref=args.manifest_path,
            config=config,
        )
    else:
        if args.input_jsonl is None:
            raise ValueError("input_jsonl is required when --manifest-path is not provided.")
        raw_jsonl = args.input_jsonl.read_text(encoding="utf-8")
        mart = build_training_data_mart_from_jsonl(
            raw_jsonl=raw_jsonl,
            as_of_time=_parse_iso_datetime(args.as_of_time),
            source="script_build_training_datamart",
            config=config,
            strict_timestamps=args.strict_timestamps,
            fail_on_warnings=args.fail_on_warnings,
            manifest_root=args.manifest_root,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(_to_jsonable(mart), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote datamart to {args.output_json}")
    print(f"Rows: {mart['stats']['rows_total']}, pairs: {mart['stats']['pair_rows_total']}")
    if mart.get("source_manifest_id"):
        print(f"Manifest: {mart['source_manifest_id']} @ {mart.get('source_manifest_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
