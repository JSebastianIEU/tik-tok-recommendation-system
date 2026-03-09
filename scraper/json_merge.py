from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class JsonMergeSummary:
    input_files: int
    rows_read: int
    rows_written: int
    duplicates_skipped: int
    output_path: str


def _record_key(row: Dict[str, Any]) -> str | None:
    normalized = row.get("normalized")
    if isinstance(normalized, dict):
        video = normalized.get("video")
        if isinstance(video, dict):
            vid = video.get("video_id")
            if isinstance(vid, str) and vid:
                return f"video_id:{vid}"
    url = row.get("url")
    if isinstance(url, str) and url:
        return f"url:{url}"
    return None


def _record_score(row: Dict[str, Any]) -> int:
    score = 0
    if row.get("success"):
        score += 1000
    normalized = row.get("normalized")
    if isinstance(normalized, dict):
        comments = normalized.get("comments")
        if isinstance(comments, list):
            score += len(comments)
        video = normalized.get("video")
        if isinstance(video, dict):
            for k in ("likes", "comments_count", "shares", "plays"):
                if video.get(k) is not None:
                    score += 5
    return score


def merge_jsonl_files(
    input_paths: Iterable[str],
    output_path: str,
) -> JsonMergeSummary:
    paths = [Path(p) for p in input_paths]
    if not paths:
        raise ValueError("No input files provided.")

    rows_read = 0
    dedup: Dict[str, Dict[str, Any]] = {}
    passthrough: List[Dict[str, Any]] = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input JSONL not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                rows_read += 1
                row = json.loads(raw)
                if not isinstance(row, dict):
                    continue
                key = _record_key(row)
                if not key:
                    passthrough.append(row)
                    continue
                previous = dedup.get(key)
                if previous is None or _record_score(row) > _record_score(previous):
                    dedup[key] = row

    merged_rows: List[Dict[str, Any]] = list(dedup.values()) + passthrough
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows_written = len(merged_rows)
    duplicates_skipped = max(0, rows_read - rows_written)
    return JsonMergeSummary(
        input_files=len(paths),
        rows_read=rows_read,
        rows_written=rows_written,
        duplicates_skipped=duplicates_skipped,
        output_path=str(out),
    )

