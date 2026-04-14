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
        default="artifacts/recommender/20260414T050542Z-phase2-bootstrap-feedback",
        help="Path to recommender bundle directory.",
    )
    parser.add_argument(
        "--corpus-bundle-path",
        type=str,
        default="",
        help="Path to artifact bundle JSON used for corpus_scope resolution.",
    )
    args = parser.parse_args()

    bundle_dir = args.bundle_dir
    if os.path.isfile(bundle_dir):
        bundle_dir = Path(bundle_dir).read_text(encoding="utf-8").strip()
    os.environ["RECOMMENDER_BUNDLE_DIR"] = bundle_dir

    if args.corpus_bundle_path:
        os.environ.setdefault("RECOMMENDER_CORPUS_BUNDLE_PATH", args.corpus_bundle_path)
    elif not os.environ.get("RECOMMENDER_CORPUS_BUNDLE_PATH"):
        default_corpus = str(Path("artifacts/contracts/latest_supabase_bundle.json"))
        if os.path.isfile(default_corpus):
            os.environ["RECOMMENDER_CORPUS_BUNDLE_PATH"] = default_corpus
    try:
        import uvicorn  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("uvicorn is required. Install uvicorn to serve recommender API.") from exc

    uvicorn.run("src.recommendation.service:app", host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
