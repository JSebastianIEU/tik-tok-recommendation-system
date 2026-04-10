#!/usr/bin/env python3
"""Evaluate shadow-promotion gates for fabric extractor rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.recommendation import PromotionThresholds, evaluate_shadow_promotion


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate fabric rollout promotion gates.")
    parser.add_argument("baseline_json", type=Path, help="Baseline metrics JSON path.")
    parser.add_argument("shadow_json", type=Path, help="Shadow metrics JSON path.")
    args = parser.parse_args()

    baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    shadow = json.loads(args.shadow_json.read_text(encoding="utf-8"))
    ok, failures = evaluate_shadow_promotion(
        baseline=baseline,
        shadow=shadow,
        thresholds=PromotionThresholds(),
    )
    print(
        json.dumps(
            {
                "ok": ok,
                "failures": failures,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
