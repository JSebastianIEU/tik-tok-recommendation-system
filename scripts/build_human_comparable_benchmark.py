#!/usr/bin/env python3
"""Build a human-labeled comparable benchmark from real DB-backed videos."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scraper.data_requests import get_full_data
from src.recommendation.learning import (
    BenchmarkCandidate,
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkQuery,
    RecommenderRuntime,
    default_rubric,
    save_benchmark_dataset,
)
from src.recommendation.learning.query_contract import infer_topic_key


def _to_iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_hashtags(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        out.append(text if text.startswith("#") else f"#{text}")
    return out


def _extract_hashtags_from_text(text: str) -> List[str]:
    return _clean_hashtags(re.findall(r"(#[^\s#]+)", str(text or "")))


def _row_hashtags(row: Dict[str, Any]) -> List[str]:
    from_row = _clean_hashtags(list(row.get("hashtags") or []))
    if from_row:
        return from_row
    return _extract_hashtags_from_text(str(row.get("caption") or ""))


def _keywords_from_hashtags(values: Sequence[str]) -> List[str]:
    return [value.lstrip("#").strip().lower() for value in values if value.lstrip("#").strip()]


def _comments_preview(row: Dict[str, Any], limit: int = 3) -> List[str]:
    comments = row.get("comments")
    if not isinstance(comments, list):
        return []
    out: List[str] = []
    for item in comments[: max(1, int(limit))]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if text:
            out.append(text)
    return out


def _runtime_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    hashtags = _row_hashtags(row)
    caption = str(row.get("caption") or "").strip()
    created_at = _to_iso(row.get("created_at"))
    return {
        "candidate_id": str(row.get("video_id") or ""),
        "video_id": str(row.get("video_id") or ""),
        "video_url": str(row.get("url") or ""),
        "text": caption,
        "caption": caption,
        "hashtags": hashtags,
        "keywords": _keywords_from_hashtags(hashtags),
        "posted_at": created_at,
        "as_of_time": created_at,
        "author_id": str(row.get("video_author_id") or ""),
    }


def _query_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = _runtime_payload(row)
    return {
        "query_id": payload["candidate_id"],
        "video_id": payload["video_id"],
        "text": payload["text"],
        "description": payload["caption"],
        "hashtags": list(payload["hashtags"]),
        "keywords": list(payload["keywords"]),
        "author_id": payload["author_id"],
        "as_of_time": payload["as_of_time"],
    }


def _display_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    hashtags = _row_hashtags(row)
    return {
        "video_id": str(row.get("video_id") or ""),
        "video_url": str(row.get("url") or ""),
        "caption": str(row.get("caption") or ""),
        "hashtags": hashtags,
        "created_at": _to_iso(row.get("created_at")),
        "author_id": str(row.get("video_author_id") or ""),
        "author_username": str(row.get("video_author_username") or ""),
        "author_display_name": str(row.get("video_author_display_name") or ""),
        "comments_preview": _comments_preview(row),
    }


def _fetch_full_rows(
    *,
    db_url: str,
    source_limit: int,
    since: Optional[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cursor_created_at: Optional[str] = None
    cursor_id: Optional[str] = None
    while len(rows) < max(1, int(source_limit)):
        page = get_full_data(
            db_url=db_url,
            limit=min(200, max(1, int(source_limit) - len(rows))),
            offset=0,
            since=since,
            cursor_created_at=cursor_created_at,
            cursor_id=cursor_id,
        )
        if not page.rows:
            break
        rows.extend(page.rows)
        if not page.next_cursor:
            break
        cursor_created_at = str(page.next_cursor.get("cursor_created_at") or "")
        cursor_id = str(page.next_cursor.get("cursor_id") or "")
        if not cursor_created_at or not cursor_id:
            break
    return rows[: max(1, int(source_limit))]


def _created_at_sort_key(row: Dict[str, Any]) -> str:
    return str(_to_iso(row.get("created_at")) or "")


def _row_topic(row: Dict[str, Any]) -> str:
    return infer_topic_key(
        {
            "description": row.get("caption"),
            "hashtags": _row_hashtags(row),
            "keywords": _keywords_from_hashtags(_row_hashtags(row)),
        }
    )


def _build_case(
    *,
    runtime: RecommenderRuntime,
    row: Dict[str, Any],
    objective: str,
    source_rows: Sequence[Dict[str, Any]],
    candidate_pool_cap: int,
    label_pool_size: int,
    retrieve_k: int,
    min_label_pool_size: int,
) -> Optional[BenchmarkCase]:
    query_time = _to_iso(row.get("created_at"))
    if not query_time:
        return None
    query_id = str(row.get("video_id") or "").strip()
    if not query_id:
        return None

    candidate_rows = [
        item
        for item in source_rows
        if str(item.get("video_id") or "").strip()
        and str(item.get("video_id") or "").strip() != query_id
        and (_to_iso(item.get("created_at")) or "") < query_time
    ]
    if len(candidate_rows) < max(1, int(min_label_pool_size)):
        return None
    candidate_rows = sorted(candidate_rows, key=_created_at_sort_key, reverse=True)[
        : max(label_pool_size, candidate_pool_cap)
    ]
    if len(candidate_rows) < max(1, int(min_label_pool_size)):
        return None

    response = runtime.recommend(
        objective=objective,
        as_of_time=query_time,
        query=_query_payload(row),
        candidates=[_runtime_payload(item) for item in candidate_rows],
        top_k=min(max(1, int(label_pool_size)), len(candidate_rows)),
        retrieve_k=min(max(1, int(retrieve_k)), len(candidate_rows)),
        explainability={"enabled": False},
        debug=False,
    )
    items = list(response.get("items") or [])
    if len(items) < max(1, int(min_label_pool_size)):
        return None

    rows_by_id = {str(item.get("video_id") or ""): item for item in candidate_rows}
    candidates: List[BenchmarkCandidate] = []
    for item in items[: max(1, int(label_pool_size))]:
        candidate_id = str(item.get("candidate_id") or "").strip()
        source_row = rows_by_id.get(candidate_id)
        if not candidate_id or source_row is None:
            continue
        candidates.append(
            BenchmarkCandidate(
                candidate_id=candidate_id,
                display=_display_payload(source_row),
                candidate_payload=_runtime_payload(source_row),
                baseline_rank=int(item.get("rank") or 0),
                baseline_score=float(item.get("score") or 0.0),
                support_level=str(item.get("support_level") or ""),
                ranking_reasons=[
                    str(reason) for reason in list(item.get("ranking_reasons") or [])
                ],
            )
        )
    if len(candidates) < max(1, int(min_label_pool_size)):
        return None

    return BenchmarkCase(
        case_id=f"{objective}::{query_id}",
        objective=objective,
        query=BenchmarkQuery(
            query_id=query_id,
            display=_display_payload(row),
            query_payload=_query_payload(row),
        ),
        candidates=candidates,
        retrieve_k=min(max(1, int(retrieve_k)), len(candidate_rows)),
        label_pool_size=len(candidates),
        source_candidate_pool_size=len(candidate_rows),
        notes=f"topic={_row_topic(row)}",
    )


def _select_cases(
    *,
    runtime: RecommenderRuntime,
    source_rows: Sequence[Dict[str, Any]],
    objectives: Sequence[str],
    query_count: int,
    seed: int,
    candidate_pool_cap: int,
    label_pool_size: int,
    retrieve_k: int,
    min_history_candidates: int,
    min_label_pool_size: int,
    max_per_topic: int,
) -> List[BenchmarkCase]:
    rows = [row for row in source_rows if _to_iso(row.get("created_at"))]
    rng = random.Random(int(seed))
    shuffled = list(rows)
    rng.shuffle(shuffled)

    selected: List[BenchmarkCase] = []
    used_query_ids: set[str] = set()
    topic_counts: Counter[str] = Counter()

    def _attempt(allow_topic_overflow: bool) -> None:
        for row in shuffled:
            if len(selected) >= max(1, int(query_count)):
                return
            query_id = str(row.get("video_id") or "").strip()
            if not query_id or query_id in used_query_ids:
                continue
            query_time = _to_iso(row.get("created_at")) or ""
            history_count = sum(
                1
                for item in source_rows
                if str(item.get("video_id") or "").strip() != query_id
                and (_to_iso(item.get("created_at")) or "") < query_time
            )
            if history_count < max(1, int(min_history_candidates)):
                continue
            topic = _row_topic(row)
            if not allow_topic_overflow and topic_counts.get(topic, 0) >= max(1, int(max_per_topic)):
                continue
            objective = objectives[len(selected) % len(objectives)]
            case = _build_case(
                runtime=runtime,
                row=row,
                objective=objective,
                source_rows=source_rows,
                candidate_pool_cap=candidate_pool_cap,
                label_pool_size=label_pool_size,
                retrieve_k=retrieve_k,
                min_label_pool_size=min_label_pool_size,
            )
            if case is None:
                continue
            selected.append(case)
            used_query_ids.add(query_id)
            topic_counts[topic] += 1

    _attempt(allow_topic_overflow=False)
    if len(selected) < max(1, int(query_count)):
        _attempt(allow_topic_overflow=True)
    return selected


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a human-labeled comparable benchmark from real DB-backed videos."
    )
    parser.add_argument("--db-url", type=str, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--objectives",
        type=str,
        default="engagement",
        help="Comma-separated objectives to cycle across benchmark cases.",
    )
    parser.add_argument("--query-count", type=int, default=24)
    parser.add_argument("--source-limit", type=int, default=400)
    parser.add_argument("--candidate-pool-cap", type=int, default=120)
    parser.add_argument("--label-pool-size", type=int, default=12)
    parser.add_argument("--min-label-pool-size", type=int, default=8)
    parser.add_argument("--retrieve-k", type=int, default=80)
    parser.add_argument("--min-history-candidates", type=int, default=40)
    parser.add_argument("--max-per-topic", type=int, default=3)
    parser.add_argument("--since", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    objectives = [item.strip() for item in str(args.objectives).split(",") if item.strip()]
    if not objectives:
        raise ValueError("At least one objective is required.")

    runtime = RecommenderRuntime(bundle_dir=args.bundle_dir)
    source_rows = _fetch_full_rows(
        db_url=args.db_url,
        source_limit=max(50, int(args.source_limit)),
        since=args.since,
    )
    if not source_rows:
        raise ValueError("No source rows were fetched from the database.")

    cases = _select_cases(
        runtime=runtime,
        source_rows=source_rows,
        objectives=objectives,
        query_count=max(1, int(args.query_count)),
        seed=int(args.seed),
        candidate_pool_cap=max(1, int(args.candidate_pool_cap)),
        label_pool_size=max(1, int(args.label_pool_size)),
        retrieve_k=max(1, int(args.retrieve_k)),
        min_history_candidates=max(1, int(args.min_history_candidates)),
        min_label_pool_size=max(1, int(args.min_label_pool_size)),
        max_per_topic=max(1, int(args.max_per_topic)),
    )

    dataset = BenchmarkDataset(
        version="recommender.human_comparable_benchmark.v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        bundle_dir=str(args.bundle_dir.resolve()),
        sample_metadata={
            "source": "supabase_full_rows",
            "source_limit": int(args.source_limit),
            "source_rows_fetched": len(source_rows),
            "query_count_requested": int(args.query_count),
            "query_count_built": len(cases),
            "objectives": objectives,
            "candidate_pool_cap": int(args.candidate_pool_cap),
            "label_pool_size": int(args.label_pool_size),
            "min_label_pool_size": int(args.min_label_pool_size),
            "retrieve_k": int(args.retrieve_k),
            "min_history_candidates": int(args.min_history_candidates),
            "max_per_topic": int(args.max_per_topic),
            "seed": int(args.seed),
            "since": args.since,
        },
        rubric=default_rubric(),
        cases=cases,
    )
    save_benchmark_dataset(dataset, args.output_path)

    topic_counts = Counter()
    for case in cases:
        topic = str(case.notes or "").replace("topic=", "", 1) if case.notes.startswith("topic=") else "unknown"
        topic_counts[topic] += 1
    summary = {
        "output_path": str(args.output_path.resolve()),
        "case_count": len(cases),
        "source_rows_fetched": len(source_rows),
        "objective_counts": dict(Counter(case.objective for case in cases)),
        "topic_counts": dict(topic_counts),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
