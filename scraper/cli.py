from __future__ import annotations

import argparse

from scraper.config import load_pipeline_config
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scraper.pipeline import PipelineSummary


def _print_summary(summary: "PipelineSummary") -> None:
    duration_sec = int((summary.ended_at - summary.started_at).total_seconds())
    print("\nPipeline summary")
    print(f"- scrape_run_id: {summary.scrape_run_id}")
    print(f"- total_queries: {summary.total_queries}")
    print(f"- discovered_total: {summary.discovered_total}")
    print(f"- unique_videos: {summary.unique_videos}")
    print(f"- scraped_ok: {summary.scraped_ok}")
    print(f"- scraped_failed: {summary.scraped_failed}")
    print(f"- persisted_ok: {summary.persisted_ok}")
    print(f"- persisted_skipped: {summary.persisted_skipped}")
    print(f"- persist_failed: {summary.persist_failed}")
    print(f"- comments_written: {summary.comments_written}")
    print(f"- unique_authors: {summary.unique_authors}")
    print(f"- duration_sec: {duration_sec}")


def _print_merge_summary(summary: object) -> None:
    duration_sec = int((summary.ended_at - summary.started_at).total_seconds())
    print("\nMerge summary")
    print(f"- target_db: {summary.target_db}")
    print(f"- source_count: {summary.source_count}")
    for table, count in summary.table_counts.items():
        print(f"- {table}: {count}")
    print(f"- duration_sec: {duration_sec}")


def _print_json_merge_summary(summary: object) -> None:
    print("\nJSON merge summary")
    print(f"- input_files: {summary.input_files}")
    print(f"- rows_read: {summary.rows_read}")
    print(f"- rows_written: {summary.rows_written}")
    print(f"- duplicates_skipped: {summary.duplicates_skipped}")
    print(f"- output_path: {summary.output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TikTok scraper CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_run = subparsers.add_parser("run", help="Run the scraper pipeline from config")
    p_run.add_argument(
        "--config",
        required=True,
        help="Path to YAML/JSON config file.",
    )
    p_run.add_argument(
        "--db-url",
        default=None,
        help="Optional DB URL override (takes precedence over config.db_url and env).",
    )

    p_init = subparsers.add_parser("init-db", help="Apply DB schema DDL")
    p_init.add_argument(
        "--db-url",
        default=None,
        help="Optional DB URL override (falls back to DATABASE_URL env var).",
    )
    p_init.add_argument(
        "--schema",
        default=None,
        help="Optional path to schema SQL (defaults to scraper/db/init/001_schema.sql).",
    )

    p_merge = subparsers.add_parser("merge", help="Merge one or more source DBs into a target DB")
    p_merge.add_argument(
        "--target-db",
        required=True,
        help="Destination DB URL where merged data will be written.",
    )
    p_merge.add_argument(
        "--source-db",
        dest="source_dbs",
        action="append",
        required=True,
        help="Source DB URL to merge. Repeat this flag for multiple sources.",
    )
    p_merge.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Read batch size per table (default: 1000).",
    )
    p_merge.add_argument(
        "--no-init-target-schema",
        action="store_true",
        help="Skip applying schema on target before merge.",
    )

    p_json_merge = subparsers.add_parser(
        "merge-json",
        help="Merge multiple JSONL datasets (DB-free workflow).",
    )
    p_json_merge.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="Input JSONL file. Repeat this flag for multiple files.",
    )
    p_json_merge.add_argument(
        "--output",
        required=True,
        help="Output merged JSONL file.",
    )

    p_scrape_all = subparsers.add_parser(
        "scrape-all",
        help="Full-scale scrape: hashtags + keywords into Postgres/Supabase.",
    )
    p_scrape_all.add_argument(
        "config",
        help="Path to full_scale.yaml config.",
    )
    p_scrape_all.add_argument(
        "--db-url",
        help="Override DB URL (else from config or DATABASE_URL).",
    )
    p_scrape_all.add_argument(
        "--init-db",
        action="store_true",
        help="Apply schema before scraping.",
    )
    p_scrape_all.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip DB write for videos already in DB (avoids duplicate storage).",
    )
    p_scrape_all.add_argument(
        "--delay",
        type=float,
        default=5,
        help="Seconds between sources (default: 5).",
    )
    p_scrape_all.add_argument(
        "--comments",
        type=int,
        default=5,
        help="Max comments per video to fetch (default 5).",
    )
    p_scrape_all.add_argument(
        "--replies",
        type=int,
        default=5,
        help="Max replies per comment to fetch (default 5).",
    )
    p_scrape_all.add_argument(
        "--min-likes-for-replies",
        type=int,
        default=10,
        help="Only fetch replies for comments with at least this many likes (default 10).",
    )
    p_scrape_all.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable source checkpoint resume and execute all sources.",
    )
    p_scrape_all.add_argument(
        "--summary-path",
        default=None,
        help="Optional path for run summary JSON.",
    )
    p_scrape_all.add_argument(
        "--proxies-file",
        default=None,
        help="Optional proxies file path passed to scrape workers.",
    )
    p_scrape_all.add_argument(
        "--max-consecutive-empty",
        type=int,
        default=5,
        help="Abort run after N consecutive empty-result sources (default 5, 0 disables).",
    )
    p_scrape_all.add_argument(
        "--retry-empty",
        type=int,
        default=2,
        help="Retry attempts when a source returns 0 rows due to blocking (default 2).",
    )
    p_scrape_all.add_argument(
        "--retry-delay",
        type=float,
        default=20.0,
        help="Seconds to wait before retrying an empty source (default 20).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init-db":
        from scraper.db.schema import apply_schema

        applied = apply_schema(db_url=args.db_url, schema_path=args.schema)
        print(f"Schema applied successfully from: {applied}")
        return 0

    if args.command == "run":
        from scraper.pipeline import run_pipeline

        config = load_pipeline_config(args.config)
        summary = run_pipeline(config, db_url_override=args.db_url)
        _print_summary(summary)
        return 0

    if args.command == "merge":
        from scraper.db.merge import merge_databases

        summary = merge_databases(
            target_db_url=args.target_db,
            source_db_urls=args.source_dbs,
            batch_size=args.batch_size,
            init_target_schema=not args.no_init_target_schema,
        )
        _print_merge_summary(summary)
        return 0

    if args.command == "merge-json":
        from scraper.json_merge import merge_jsonl_files

        summary = merge_jsonl_files(args.inputs, args.output)
        _print_json_merge_summary(summary)
        return 0

    if args.command == "scrape-all":
        from scraper.run_full_scale import main as scrape_all_main

        argv = [args.config]
        if args.db_url:
            argv.extend(["--db-url", args.db_url])
        if args.init_db:
            argv.append("--init-db")
        if args.skip_existing:
            argv.append("--skip-existing")
        if args.delay != 5:
            argv.extend(["--delay", str(args.delay)])
        if args.comments != 5:
            argv.extend(["--comments", str(args.comments)])
        if args.replies != 5:
            argv.extend(["--replies", str(args.replies)])
        if args.min_likes_for_replies != 10:
            argv.extend(["--min-likes-for-replies", str(args.min_likes_for_replies)])
        if args.no_resume:
            argv.append("--no-resume")
        if args.summary_path:
            argv.extend(["--summary-path", args.summary_path])
        if args.proxies_file:
            argv.extend(["--proxies-file", args.proxies_file])
        if args.max_consecutive_empty != 5:
            argv.extend(["--max-consecutive-empty", str(args.max_consecutive_empty)])
        if args.retry_empty != 2:
            argv.extend(["--retry-empty", str(args.retry_empty)])
        if args.retry_delay != 20.0:
            argv.extend(["--retry-delay", str(args.retry_delay)])
        return scrape_all_main(argv)

    parser.print_help()
    return 1
