#!/usr/bin/env python3
"""Unified evaluation entrypoint.

Dispatches to the appropriate specialised eval script based on ``--mode``:

    recommender   -> scripts/eval_recommender.py
    retriever     -> scripts/eval_retriever.py
    benchmark     -> scripts/eval_human_comparable_benchmark.py

Any extra arguments after ``--mode`` are forwarded verbatim to the target
script, so the full CLI of each sub-script is still available.

Examples:
    python scripts/run_eval.py --mode recommender artifacts/recommender/latest --show-manifest
    python scripts/run_eval.py --mode retriever data/mock/training_datamart.json --retrieve-k 100
    python scripts/run_eval.py --mode benchmark --bundle-dir artifacts/recommender/latest \\
        --benchmark-json artifacts/benchmarks/benchmark.json
    python scripts/run_eval.py --list
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_MODES: Dict[str, str] = {
    "recommender": "scripts.eval_recommender",
    "retriever": "scripts.eval_retriever",
    "benchmark": "scripts.eval_human_comparable_benchmark",
}


def _print_help() -> None:
    print("Usage: python scripts/run_eval.py --mode <MODE> [args...]")
    print("       python scripts/run_eval.py --list\n")
    print("Available modes:")
    for name, module in _MODES.items():
        print(f"  {name:16s} -> {module.replace('.', '/')}.py")
    print("\nAll extra arguments are forwarded to the target script.")
    print("Use  python scripts/run_eval.py --mode <MODE> --help  to see sub-script options.")


def main() -> int:
    if "--list" in sys.argv or (len(sys.argv) < 2):
        _print_help()
        return 0

    if "--help" == sys.argv[1] or "-h" == sys.argv[1]:
        _print_help()
        return 0

    if sys.argv[1] != "--mode" or len(sys.argv) < 3:
        print(f"Error: expected --mode <{'|'.join(_MODES)}>")
        _print_help()
        return 1

    mode = sys.argv[2]
    if mode not in _MODES:
        print(f"Error: unknown mode '{mode}'. Choose from: {', '.join(_MODES)}")
        return 1

    sys.argv = [f"scripts/run_eval.py ({mode})"] + sys.argv[3:]

    module = importlib.import_module(_MODES[mode])
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
