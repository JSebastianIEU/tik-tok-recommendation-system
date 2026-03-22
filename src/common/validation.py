import json
from pathlib import Path
from typing import Tuple, List

from .schemas import TikTokPost

from typing import Any, Dict, Iterable, List, Tuple
from datetime import datetime, timezone

from pydantic import ValidationError

from src.common.schemas import TikTokPost, Comment


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
    except ValidationError as e:
        errors.extend([f"schema: {err['msg']} (loc={'.'.join(map(str, err['loc']))})" for err in e.errors()])
        return False, errors

    # Extra business rules
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

    # Validate each comment created_at is not in the future and is a datetime
    now = datetime.now(timezone.utc)
    for idx, c in enumerate(post.comments):
        if not isinstance(c, Comment):
            errors.append(f"comment[{idx}] is not a Comment instance")
            continue
        if c.likes < 0:
            errors.append(f"comment[{idx}].likes must be >= 0")
        if c.created_at > now:
            errors.append(f"comment[{idx}].created_at is in the future: {c.created_at.isoformat()}")

    return (len(errors) == 0), errors


def validate_stream(records: Iterable[Dict[str, Any]]) -> Tuple[int, int, List[Tuple[int, List[str]]]]:
    """Validate an iterable of record dicts.

    Returns (total, failed, failures) where failures is a list of
    (index, errors) for records that failed validation.
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
    """Simple JSONL loader that yields parsed objects.

    Note: this helper is intentionally minimal to keep dependencies low.
    """
    import json

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
import json
from pathlib import Path
from typing import Tuple, List

from .schemas import TikTokPost

def validate_file(jsonl_path: Path) -> Tuple[int, List[str]]:
    """Validate mocked JSONL records against TikTokPost schema."""
    errors: List[str] = []
    count = 0
    for line_no, raw in enumerate(jsonl_path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            TikTokPost.model_validate_json(raw)
            count += 1
        except Exception as exc:  # TODO: tighten exception types
            errors.append(f"Line {line_no}: {exc}")
    return count, errors

# TODO: add CLI-friendly reporter with severity levels and data quality scores.
