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
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

from scraper.config import VALID_MODES

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Delay between sources (seconds) to reduce rate limiting
DEFAULT_DELAY_BETWEEN_SOURCES = 5


@dataclass(frozen=True)
class SourceJob:
    mode: str
    name: str
    count: int
    output_path: Path

    @property
    def key(self) -> str:
        return f"{self.mode}:{self.name.strip().lower()}"


def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML object")
    return data


def _resolve_db_url(args_db_url: str | None, config: dict[str, Any]) -> str | None:
    raw = (args_db_url or os.getenv("DATABASE_URL") or config.get("db_url") or "").strip()
    return raw or None


def _resolve_int_option(
    cli_value: int | None,
    config: dict[str, Any],
    key: str,
    *,
    default: int,
    min_value: int = 0,
) -> int:
    value = cli_value if cli_value is not None else config.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"Config '{key}' must be an integer.")
    if value < min_value:
        raise ValueError(f"Config '{key}' must be >= {min_value}.")
    return value


def _resolve_float_option(
    cli_value: float | None,
    config: dict[str, Any],
    key: str,
    *,
    default: float,
    min_value: float = 0.0,
) -> float:
    value = cli_value if cli_value is not None else config.get(key, default)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Config '{key}' must be numeric.")
    as_float = float(value)
    if as_float < min_value:
        raise ValueError(f"Config '{key}' must be >= {min_value}.")
    return as_float


def _resolve_bool_option(
    cli_value: bool | None,
    config: dict[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    value = cli_value if cli_value is not None else config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Config '{key}' must be boolean.")
    return value


def _normalized_modes(config: dict[str, Any]) -> set[str]:
    raw = config.get("modes_enabled")
    if raw is None:
        return set(VALID_MODES)
    if not isinstance(raw, list):
        raise ValueError("Config 'modes_enabled' must be a list of strings.")
    out: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            raise ValueError("Config 'modes_enabled' must contain only strings.")
        mode = item.strip().lower()
        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{item}'. Valid values: {', '.join(sorted(VALID_MODES))}"
            )
        out.add(mode)
    return out or set(VALID_MODES)


def _item_name_and_count(
    item: Any,
    *,
    mode: str,
    default_count: int,
) -> tuple[str, int] | None:
    if isinstance(item, dict):
        name = str(item.get("name") or "").strip()
        raw_count = item.get("count", default_count)
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid count for {mode} source '{name}': {raw_count!r}") from None
    else:
        name = str(item).strip()
        count = default_count
    if not name:
        return None
    if count < 1:
        raise ValueError(f"Source '{mode}:{name}' has invalid count={count}. Must be >= 1.")
    return name, count


def _safe_name(value: str, limit: int | None = None) -> str:
    cleaned = value.replace(" ", "_").replace("#", "")
    return cleaned[:limit] if limit else cleaned


def _build_jobs(config: dict[str, Any], output_dir: Path) -> list[SourceJob]:
    jobs: list[SourceJob] = []
    enabled_modes = _normalized_modes(config)
    default_count = int(config.get("default_count", 100))
    default_per_query = int(config.get("per_query_video_limit", default_count))
    default_hashtag_count = int(config.get("default_hashtag_count", default_per_query))
    default_keyword_count = int(config.get("default_keyword_count", default_per_query))

    if "hashtag" in enabled_modes:
        for item in config.get("hashtags") or []:
            parsed = _item_name_and_count(item, mode="hashtag", default_count=default_hashtag_count)
            if parsed is None:
                continue
            name, count = parsed
            safe = _safe_name(name)
            jobs.append(
                SourceJob(
                    mode="hashtag",
                    name=name,
                    count=count,
                    output_path=output_dir / f"full_scale_hashtag_{safe}.jsonl",
                )
            )

    if "keyword" in enabled_modes:
        for item in config.get("keywords") or []:
            parsed = _item_name_and_count(item, mode="keyword", default_count=default_keyword_count)
            if parsed is None:
                continue
            name, count = parsed
            safe = _safe_name(name, limit=30)
            jobs.append(
                SourceJob(
                    mode="keyword",
                    name=name,
                    count=count,
                    output_path=output_dir / f"full_scale_keyword_{safe}.jsonl",
                )
            )
    return jobs


def _run_scrape(
    job: SourceJob,
    *,
    db_url: str,
    env: dict[str, str],
    skip_existing: bool = False,
    comments: int = 0,
    replies: int = 5,
    min_likes_for_replies: int = 10,
    proxies_file: str | None = None,
) -> tuple[int, int]:
    """Run scrape_tiktok_sample for one source job. Returns (exit_code, rows_written)."""
    cmd = [
        sys.executable,
        "-m",
        "scraper.scrape_tiktok_sample",
        "--output",
        str(job.output_path),
        "--db-url",
        db_url,
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    if comments > 0:
        cmd.extend(["--comments", str(comments)])
    if replies > 0:
        cmd.extend(["--replies", str(replies)])
    if min_likes_for_replies > 0:
        cmd.extend(["--min-likes-for-replies", str(min_likes_for_replies)])
    if proxies_file:
        cmd.extend(["--proxies-file", proxies_file])

    cmd.extend([job.mode, "--name", job.name, "--count", str(job.count)])
    result = subprocess.run(cmd, env=env, cwd=str(ROOT))
    rows_written = 0
    try:
        if job.output_path.exists():
            with job.output_path.open("r", encoding="utf-8") as handle:
                rows_written = sum(1 for line in handle if line.strip())
    except Exception:
        rows_written = 0
    return result.returncode, rows_written


def _ensure_job_state_table(db_url: str) -> None:
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS scrape_source_jobs (
                    source_key TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    started_at TIMESTAMPTZ,
                    finished_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_scrape_source_jobs_status ON scrape_source_jobs(status)"
            )
        conn.commit()


def _load_completed_jobs(db_url: str) -> set[str]:
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT source_key FROM scrape_source_jobs WHERE status = 'done'")
            return {str(row[0]) for row in cur.fetchall()}


def _mark_job_running(db_url: str, job: SourceJob) -> None:
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO scrape_source_jobs (
                    source_key, mode, name, status, attempt_count, started_at, finished_at, last_error, updated_at
                )
                VALUES (%s, %s, %s, 'running', 1, NOW(), NULL, NULL, NOW())
                ON CONFLICT (source_key) DO UPDATE SET
                    mode = EXCLUDED.mode,
                    name = EXCLUDED.name,
                    status = 'running',
                    attempt_count = scrape_source_jobs.attempt_count + 1,
                    started_at = NOW(),
                    finished_at = NULL,
                    last_error = NULL,
                    updated_at = NOW()
                """,
                (job.key, job.mode, job.name),
            )
        conn.commit()


def _mark_job_finished(db_url: str, job: SourceJob, *, status: str, last_error: str | None = None) -> None:
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE scrape_source_jobs
                SET status = %s,
                    finished_at = NOW(),
                    last_error = %s,
                    updated_at = NOW()
                WHERE source_key = %s
                """,
                (status, last_error, job.key),
            )
        conn.commit()


def _default_summary_path(output_dir: Path, started_at: datetime) -> Path:
    return output_dir / f"full_scale_summary_{started_at.strftime('%Y%m%dT%H%M%SZ')}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full-scale scrape: all modes into Postgres")
    parser.add_argument("config", help="Path to full_scale.yaml")
    parser.add_argument("--db-url", help="Override DB URL (else from env or config)")
    parser.add_argument("--init-db", action="store_true", help="Apply schema before scraping")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=None,
        help="Skip DB write for videos already in DB (default from config.skip_existing, fallback false).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help=f"Seconds to wait between sources (default from config.delay, fallback {DEFAULT_DELAY_BETWEEN_SOURCES}).",
    )
    parser.add_argument(
        "--comments",
        type=int,
        default=None,
        help="Max comments per video to fetch (default from config.comments, fallback 5).",
    )
    parser.add_argument(
        "--replies",
        type=int,
        default=None,
        help="Max replies per comment to fetch (default from config.replies, fallback 5).",
    )
    parser.add_argument(
        "--min-likes-for-replies",
        type=int,
        default=None,
        help="Only fetch replies for comments with at least this many likes (default from config.min_likes_for_replies, fallback 10).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=None,
        help="Disable checkpoint resume and run all sources regardless of previous status.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional path for run summary JSON (default: output_dir/full_scale_summary_<timestamp>.json).",
    )
    parser.add_argument(
        "--proxies-file",
        default=None,
        help="Optional proxies file path passed to scrape workers (default from config.proxies_file).",
    )
    parser.add_argument(
        "--retry-empty",
        type=int,
        default=None,
        help="Retry attempts when a source returns 0 rows due to blocking (default from config.retry_empty, fallback 2).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=None,
        help="Seconds to wait before retrying an empty source (default from config.retry_delay, fallback 20).",
    )
    parser.add_argument(
        "--max-consecutive-empty",
        type=int,
        default=None,
        help="Abort run after this many consecutive empty-result source failures (default from config.max_consecutive_empty, fallback 5). Use 0 to disable.",
    )
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 1

    config = _load_config(cfg_path)
    db_url = _resolve_db_url(args.db_url, config)
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
    env["MS_TOKEN"] = str(ms_token)
    env["DATABASE_URL"] = db_url
    env["PYTHONPATH"] = str(ROOT)

    output_dir = Path(config.get("output_dir", "scraper/data/raw"))
    output_dir.mkdir(parents=True, exist_ok=True)
    proxies_file = args.proxies_file or config.get("proxies_file")
    if proxies_file is not None and not isinstance(proxies_file, str):
        raise ValueError("Config 'proxies_file' must be a string when provided.")
    skip_existing = _resolve_bool_option(args.skip_existing, config, "skip_existing", default=False)
    no_resume = _resolve_bool_option(args.no_resume, config, "no_resume", default=False)
    comments = _resolve_int_option(args.comments, config, "comments", default=5, min_value=0)
    replies = _resolve_int_option(args.replies, config, "replies", default=5, min_value=0)
    min_likes_for_replies = _resolve_int_option(
        args.min_likes_for_replies,
        config,
        "min_likes_for_replies",
        default=10,
        min_value=0,
    )
    retry_empty = _resolve_int_option(args.retry_empty, config, "retry_empty", default=2, min_value=0)
    retry_delay = _resolve_float_option(args.retry_delay, config, "retry_delay", default=20.0, min_value=0.0)
    max_consecutive_empty = _resolve_int_option(
        args.max_consecutive_empty,
        config,
        "max_consecutive_empty",
        default=5,
        min_value=0,
    )
    delay = _resolve_float_option(args.delay, config, "delay", default=DEFAULT_DELAY_BETWEEN_SOURCES, min_value=0.0)

    if args.init_db:
        from scraper.db.schema import apply_schema

        apply_schema(db_url=db_url)
        print("Schema applied.")

    _ensure_job_state_table(db_url)
    resume_enabled = not no_resume
    completed_keys = _load_completed_jobs(db_url) if resume_enabled else set()

    started_at = datetime.now(timezone.utc)
    summary_path = Path(args.summary_path).resolve() if args.summary_path else _default_summary_path(output_dir, started_at)
    jobs = _build_jobs(config, output_dir)
    failed = 0
    skipped = 0
    executed = 0
    aborted_early = False
    abort_reason: str | None = None
    consecutive_empty = 0
    source_results: list[dict[str, Any]] = []

    for idx, job in enumerate(jobs):
        if delay and idx > 0:
            time.sleep(delay)

        if resume_enabled and job.key in completed_keys:
            skipped += 1
            print(f"\n[{job.mode}] {job.name} -> skipped (already done)")
            source_results.append(
                {
                    "key": job.key,
                    "mode": job.mode,
                    "name": job.name,
                    "count": job.count,
                    "output_path": str(job.output_path),
                    "status": "skipped_done",
                    "return_code": 0,
                    "duration_sec": 0.0,
                }
            )
            continue

        print(f"\n[{job.mode}] {job.name} count={job.count} -> {job.output_path}")
        _mark_job_running(db_url, job)
        started_source = time.time()
        attempts = max(1, int(retry_empty) + 1)
        code = 1
        rows_written = 0
        for attempt in range(1, attempts + 1):
            code, rows_written = _run_scrape(
                job,
                db_url=db_url,
                env=env,
                skip_existing=skip_existing,
                comments=comments,
                replies=replies,
                min_likes_for_replies=min_likes_for_replies,
                proxies_file=proxies_file,
            )
            if code == 0 and rows_written > 0:
                break
            if attempt < attempts:
                print(
                    f"[retry] {job.key} attempt {attempt}/{attempts - 1} yielded rows={rows_written}, retrying..."
                )
                time.sleep(retry_delay)
        duration_sec = round(time.time() - started_source, 2)
        executed += 1
        if code != 0:
            failed += 1
            _mark_job_finished(db_url, job, status="failed", last_error=f"exit_code={code}")
            status = "failed"
            consecutive_empty = 0
        elif rows_written == 0:
            failed += 1
            _mark_job_finished(
                db_url,
                job,
                status="failed",
                last_error="empty_result_set_after_retries",
            )
            status = "failed_empty"
            consecutive_empty += 1
        else:
            _mark_job_finished(db_url, job, status="done")
            status = "done"
            consecutive_empty = 0
        source_results.append(
            {
                "key": job.key,
                "mode": job.mode,
                "name": job.name,
                "count": job.count,
                "output_path": str(job.output_path),
                "status": status,
                "return_code": code,
                "rows_written": rows_written,
                "duration_sec": duration_sec,
            }
        )
        if max_consecutive_empty > 0 and consecutive_empty >= max_consecutive_empty:
            aborted_early = True
            abort_reason = (
                f"aborted_after_{consecutive_empty}_consecutive_empty_sources"
            )
            print(
                f"[abort] stopping early after {consecutive_empty} consecutive empty sources."
            )
            break

    ended_at = datetime.now(timezone.utc)
    summary = {
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_sec": int((ended_at - started_at).total_seconds()),
        "resume_enabled": resume_enabled,
        "jobs_total": len(jobs),
        "jobs_executed": executed,
        "jobs_skipped_done": skipped,
        "jobs_failed": failed,
        "jobs_succeeded": executed - failed,
        "aborted_early": aborted_early,
        "abort_reason": abort_reason,
        "sources": source_results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDone. executed={executed} skipped={skipped} failed={failed}")
    print(f"Summary written to {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
