#!/usr/bin/env python3
"""Run FastAPI recommender service."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve recommender API.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--bundle-dir",
        type=str,
        default="artifacts/recommender/latest",
        help="Path to recommender bundle directory.",
    )
    args = parser.parse_args()

    bundle_dir = args.bundle_dir
    if os.path.isfile(bundle_dir):
        bundle_dir = Path(bundle_dir).read_text(encoding="utf-8").strip()
    os.environ["RECOMMENDER_BUNDLE_DIR"] = bundle_dir
    try:
        import uvicorn  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("uvicorn is required. Install uvicorn to serve recommender API.") from exc

    uvicorn.run("src.recommendation.service:app", host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
