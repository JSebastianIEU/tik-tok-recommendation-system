#!/usr/bin/env python3
"""Evaluate shadow-promotion gates for comment intelligence rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.recommendation import (
    CommentPromotionThresholds,
    evaluate_comment_shadow_promotion,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate comment intelligence rollout promotion gates."
    )
    parser.add_argument("baseline_json", type=Path, help="Baseline metrics JSON path.")
    parser.add_argument("shadow_json", type=Path, help="Shadow metrics JSON path.")
    args = parser.parse_args()

    baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    shadow = json.loads(args.shadow_json.read_text(encoding="utf-8"))
    ok, failures = evaluate_comment_shadow_promotion(
        baseline=baseline,
        shadow=shadow,
        thresholds=CommentPromotionThresholds(),
    )
    payload = {"ok": ok, "failures": failures}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
