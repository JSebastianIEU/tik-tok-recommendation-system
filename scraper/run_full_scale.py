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


def _safe_name(value: str, limit: int | None = None) -> str:
    cleaned = value.replace(" ", "_").replace("#", "")
    return cleaned[:limit] if limit else cleaned


def _build_jobs(config: dict[str, Any], output_dir: Path) -> list[SourceJob]:
    jobs: list[SourceJob] = []
    for item in config.get("hashtags") or []:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            count = int(item.get("count", 100))
        else:
            name = str(item).strip()
            count = 100
        if not name:
            continue
        safe = _safe_name(name)
        jobs.append(
            SourceJob(
                mode="hashtag",
                name=name,
                count=count,
                output_path=output_dir / f"full_scale_hashtag_{safe}.jsonl",
            )
        )

    for item in config.get("keywords") or []:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            count = int(item.get("count", 100))
        else:
            name = str(item).strip()
            count = 100
        if not name:
            continue
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
        help="Skip DB write for videos already in DB (avoids duplicate storage).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_BETWEEN_SOURCES,
        help=f"Seconds to wait between sources (default: {DEFAULT_DELAY_BETWEEN_SOURCES}).",
    )
    parser.add_argument(
        "--comments",
        type=int,
        default=5,
        help="Max comments per video to fetch (default 5).",
    )
    parser.add_argument(
        "--replies",
        type=int,
        default=5,
        help="Max replies per comment to fetch (default 5).",
    )
    parser.add_argument(
        "--min-likes-for-replies",
        type=int,
        default=10,
        help="Only fetch replies for comments with at least this many likes (default 10).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
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
        help="Optional proxies file path passed to scrape workers.",
    )
    parser.add_argument(
        "--retry-empty",
        type=int,
        default=2,
        help="Retry attempts when a source returns 0 rows due to blocking (default: 2).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=20.0,
        help="Seconds to wait before retrying an empty source (default: 20).",
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

    if args.init_db:
        from scraper.db.schema import apply_schema

        apply_schema(db_url=db_url)
        print("Schema applied.")

    _ensure_job_state_table(db_url)
    resume_enabled = not args.no_resume
    completed_keys = _load_completed_jobs(db_url) if resume_enabled else set()

    started_at = datetime.now(timezone.utc)
    summary_path = Path(args.summary_path).resolve() if args.summary_path else _default_summary_path(output_dir, started_at)
    delay = max(0.0, float(args.delay))

    jobs = _build_jobs(config, output_dir)
    failed = 0
    skipped = 0
    executed = 0
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
        attempts = max(1, int(args.retry_empty) + 1)
        code = 1
        rows_written = 0
        for attempt in range(1, attempts + 1):
            code, rows_written = _run_scrape(
                job,
                db_url=db_url,
                env=env,
                skip_existing=args.skip_existing,
                comments=args.comments,
                replies=args.replies,
                min_likes_for_replies=args.min_likes_for_replies,
                proxies_file=args.proxies_file,
            )
            if code == 0 and rows_written > 0:
                break
            if attempt < attempts:
                print(
                    f"[retry] {job.key} attempt {attempt}/{attempts - 1} yielded rows={rows_written}, retrying..."
                )
                time.sleep(max(0.0, float(args.retry_delay)))
        duration_sec = round(time.time() - started_source, 2)
        executed += 1
        if code != 0:
            failed += 1
            _mark_job_finished(db_url, job, status="failed", last_error=f"exit_code={code}")
            status = "failed"
        elif rows_written == 0:
            failed += 1
            _mark_job_finished(
                db_url,
                job,
                status="failed",
                last_error="empty_result_set_after_retries",
            )
            status = "failed_empty"
        else:
            _mark_job_finished(db_url, job, status="done")
            status = "done"
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
        "sources": source_results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDone. executed={executed} skipped={skipped} failed={failed}")
    print(f"Summary written to {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
