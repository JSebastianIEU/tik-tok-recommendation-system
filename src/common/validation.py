"""JSONL validation helpers for mock TikTok-shaped records (`TikTokPost` + rules)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import ValidationError

from src.common.schemas import Comment, TikTokPost


def validate_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a single record dict.

    Returns (is_valid, errors) where errors is a list of human-readable
    error messages. Validation includes Pydantic parsing plus extra
    checks:
      - numeric counts are >= 0 (Pydantic already enforces some)
      - `comments` length >= 3
      - each comment has a parsable `created_at` datetime
      - `video_meta.duration_seconds` > 0
    """
    errors: List[str] = []

    try:
        post = TikTokPost.model_validate(record)
    except ValidationError as exc:
        errors.extend(
            [
                f"schema: {err['msg']} (loc={'.'.join(map(str, err['loc']))})"
                for err in exc.errors()
            ]
        )
        return False, errors

    if post.comments_count < 0:
        errors.append("comments_count must be >= 0")

    if post.likes < 0:
        errors.append("likes must be >= 0")

    if post.shares < 0:
        errors.append("shares must be >= 0")

    if post.views < 0:
        errors.append("views must be >= 0")

    if post.video_meta.duration_seconds <= 0:
        errors.append("video_meta.duration_seconds must be > 0")

    if not isinstance(post.comments, list) or len(post.comments) < 3:
        errors.append("each post must have at least 3 comments")

    now = datetime.now(timezone.utc)
    for idx, c in enumerate(post.comments):
        if not isinstance(c, Comment):
            errors.append(f"comment[{idx}] is not a Comment instance")
            continue
        if c.likes < 0:
            errors.append(f"comment[{idx}].likes must be >= 0")
        if c.created_at > now:
            errors.append(
                f"comment[{idx}].created_at is in the future: {c.created_at.isoformat()}"
            )

    return (len(errors) == 0), errors


def validate_stream(records: Iterable[Dict[str, Any]]) -> Tuple[int, int, List[Tuple[int, List[str]]]]:
    """Validate an iterable of record dicts.

    Returns (total, failed, failures) where failures is a list of
    (index, errors) for records that failed validation (1-based index).
    """
    total = 0
    failed = 0
    failures: List[Tuple[int, List[str]]] = []

    for i, rec in enumerate(records, start=1):
        total += 1
        ok, errs = validate_record(rec)
        if not ok:
            failed += 1
            failures.append((i, errs))

    return total, failed, failures


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file (one object per non-empty line).

    Raises ``json.JSONDecodeError`` if a line is not valid JSON.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def validate_jsonl_file(jsonl_path: Path) -> Tuple[int, int, List[Tuple[int, List[str]]]]:
    """Validate a JSONL file line-by-line.

    Malformed JSON lines are counted as failures with a clear error message.
    Each logical record uses the same rules as :func:`validate_record`.

    Returns (total_non_empty_lines, failed_count, failures) with 1-based line
    numbers in ``failures``.
    """
    total = 0
    failed = 0
    failures: List[Tuple[int, List[str]]] = []
    text = jsonl_path.read_text(encoding="utf-8")
    for line_no, raw in enumerate(text.splitlines(), start=1):
        raw_stripped = raw.strip()
        if not raw_stripped:
            continue
        total += 1
        try:
            record = json.loads(raw_stripped)
        except json.JSONDecodeError as exc:
            failed += 1
            failures.append((line_no, [f"json: {exc.msg} (column {exc.colno})"]))
            continue
        if not isinstance(record, dict):
            failed += 1
            failures.append((line_no, ["json: each line must be a JSON object"]))
            continue
        ok, errs = validate_record(record)
        if not ok:
            failed += 1
            failures.append((line_no, errs))
    return total, failed, failures


def validate_file(jsonl_path: Path) -> Tuple[int, List[str]]:
    """Backward-compatible summary: (passed_count, flat_error_messages).

    Each error string is prefixed with ``Line N:`` for the source line number.
    """
    total, failed, failures = validate_jsonl_file(jsonl_path)
    passed = total - failed
    errors: List[str] = []
    for line_no, errs in failures:
        errors.append(f"Line {line_no}: " + "; ".join(errs))
    return passed, errors


def format_validation_summary(total: int, failed: int, *, label: str = "records") -> str:
    """One-line human summary including pass rate."""
    passed = total - failed
    pct = (100.0 * passed / total) if total else 100.0
    return (
        f"Validated {total} {label}: {passed} passed, {failed} failed "
        f"({pct:.1f}% pass rate)."
    )


def format_failure_details(
    failures: List[Tuple[int, List[str]]],
    *,
    max_failures: Optional[int] = None,
    max_errors_per_record: int = 10,
) -> str:
    """Multi-line details for failing records (line number = JSONL line index)."""
    lines: List[str] = []
    cap = len(failures) if max_failures is None else min(max_failures, len(failures))
    for line_no, errs in failures[:cap]:
        lines.append(f"  Line {line_no}:")
        show = errs[:max_errors_per_record]
        for err in show:
            lines.append(f"    - {err}")
        if len(errs) > max_errors_per_record:
            extra = len(errs) - max_errors_per_record
            lines.append(f"    - ... ({extra} more errors on this line)")
    if max_failures is not None and len(failures) > max_failures:
        lines.append(f"  ... ({len(failures) - max_failures} more failing lines not shown)")
    return "\n".join(lines)
