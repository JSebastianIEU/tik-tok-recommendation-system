#!/usr/bin/env python3
"""Fit fabric calibrators from held-out reliability observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.recommendation.fabric import FeatureFabric


def _load_observations(path: Path) -> Dict[str, List[Tuple[float, float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[Tuple[float, float]]] = {}
    if not isinstance(payload, dict):
        return out
    for key, items in payload.items():
        if not isinstance(items, list):
            continue
        parsed: List[Tuple[float, float]] = []
        for item in items:
            if isinstance(item, dict) and "raw" in item and "target" in item:
                parsed.append((float(item["raw"]), float(item["target"])))
            elif (
                isinstance(item, list)
                and len(item) == 2
            ):
                parsed.append((float(item[0]), float(item[1])))
        if parsed:
            out[str(key)] = parsed
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit fabric calibrators from held-out observations.")
    parser.add_argument("observations_json", type=Path, help="Path to observations JSON.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/features/fabric_calibration.json"),
        help="Output path for fitted calibration artifacts.",
    )
    args = parser.parse_args()

    observations = _load_observations(args.observations_json)
    fabric = FeatureFabric()
    fitted = fabric.fit_calibrators(observations)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(fitted, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote calibration artifacts to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
