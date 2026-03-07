#!/usr/bin/env python3
"""
Run full-scale scraping: hashtags + keywords only, persist to PostgreSQL/Supabase.

Usage:
  export DATABASE_URL="postgresql://..."  # Supabase connection string recommended
  export MS_TOKEN="your_ms_token"
  python -m scraper.run_full_scale scraper/configs/full_scale.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Delay between sources (seconds) to reduce rate limiting
DEFAULT_DELAY_BETWEEN_SOURCES = 5


def _load_config(path: Path) -> dict:
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML object")
    return data


def _run_scrape(
    mode: str,
    name: str | None,
    count: int,
    *,
    db_url: str,
    output_path: Path,
    env: dict,
    skip_existing: bool = False,
) -> int:
    """Run scrape_tiktok_sample for one mode. Returns exit code."""
    cmd = [
        sys.executable,
        "-m",
        "scraper.scrape_tiktok_sample",
        "--output", str(output_path),
        "--db-url", db_url,
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    if mode == "hashtag":
        cmd.extend(["hashtag", "--name", name or "", "--count", str(count)])
    elif mode == "keyword":
        cmd.extend(["keyword", "--name", name or "", "--count", str(count)])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    result = subprocess.run(cmd, env=env, cwd=str(ROOT))
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full-scale scrape: all modes into Postgres")
    parser.add_argument("config", help="Path to full_scale.yaml")
    parser.add_argument("--db-url", help="Override DB URL (else from config or DATABASE_URL)")
    parser.add_argument("--init-db", action="store_true", help="Apply schema before scraping")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip DB write for videos already in DB (avoids duplicate storage).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_BETWEEN_SOURCES,
        help=f"Seconds to wait between sources (default: {DEFAULT_DELAY_BETWEEN_SOURCES}).",
    )
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 1

    config = _load_config(cfg_path)
    db_url = (
        args.db_url
        or os.getenv("DATABASE_URL")
        or config.get("db_url")
    )
    if not db_url:
        print(
            "DATABASE_URL required. Set env or pass --db-url or set db_url in config.",
            file=sys.stderr,
        )
        return 1

    ms_token = os.getenv("MS_TOKEN") or config.get("ms_token")
    if not ms_token:
        print("MS_TOKEN required for TikTokApi. Set env or add ms_token to config.", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["MS_TOKEN"] = ms_token
    env["DATABASE_URL"] = db_url
    env["PYTHONPATH"] = str(ROOT)

    output_dir = Path(config.get("output_dir", "scraper/data/raw"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.init_db:
        from scraper.db.schema import apply_schema
        apply_schema(db_url=db_url)
        print("Schema applied.")

    total_modes = 0
    failed = 0
    delay = max(0, float(args.delay))

    # Hashtags
    for item in config.get("hashtags") or []:
        if isinstance(item, dict):
            name = item.get("name")
            count = item.get("count", 100)
        else:
            name = str(item)
            count = 100
        if name:
            if delay and total_modes > 0:
                time.sleep(delay)
            safe = name.replace(" ", "_").replace("#", "")
            out_path = output_dir / f"full_scale_hashtag_{safe}.jsonl"
            print(f"\n[hashtag] {name} count={count} -> {out_path}")
            code = _run_scrape(
                "hashtag", name, count,
                db_url=db_url, output_path=out_path, env=env,
                skip_existing=args.skip_existing,
            )
            total_modes += 1
            if code != 0:
                failed += 1

    # Keywords
    for item in config.get("keywords") or []:
        if isinstance(item, dict):
            name = item.get("name")
            count = item.get("count", 100)
        else:
            name = str(item)
            count = 100
        if name:
            if delay and total_modes > 0:
                time.sleep(delay)
            safe = name.replace(" ", "_")[:30]
            out_path = output_dir / f"full_scale_keyword_{safe}.jsonl"
            print(f"\n[keyword] {name} count={count} -> {out_path}")
            code = _run_scrape(
                "keyword", name, count,
                db_url=db_url, output_path=out_path, env=env,
                skip_existing=args.skip_existing,
            )
            total_modes += 1
            if code != 0:
                failed += 1

    print(f"\nDone. Modes run: {total_modes}, failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
