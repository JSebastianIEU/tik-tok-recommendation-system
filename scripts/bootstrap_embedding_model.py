#!/usr/bin/env python3
"""Bootstrap local sentence-transformers model cache."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and cache embedding model locally.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers is required for bootstrap. Install dependencies first."
        ) from exc

    model = SentenceTransformer(args.model_name)
    _ = model.encode(["bootstrap check"], convert_to_numpy=True)
    print(f"Model '{args.model_name}' is cached and ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

