from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return _to_utc(value)
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return _to_utc(parsed)


def row_as_of_time(row: Dict[str, Any]) -> Optional[datetime]:
    return parse_dt(row.get("as_of_time"))


def row_text(row: Dict[str, Any]) -> str:
    topic_key = str(row.get("topic_key", "")).strip().lower()
    row_id = str(row.get("row_id", "")).strip().lower()
    missingness = row.get("features", {}).get("missingness_flags", [])
    if not isinstance(missingness, list):
        missingness = []
    missing_tokens = [
        str(item).strip().lower()
        for item in missingness
        if isinstance(item, str) and item.strip()
    ]
    text_parts = [topic_key, row_id, *missing_tokens]
    return " ".join(part for part in text_parts if part).strip()


@dataclass
class TemporalCandidatePoolConfig:
    max_age_days: int = 180
    min_pool_size: int = 30
    enforce_index_cutoff: bool = True


class TemporalCandidatePool:
    def __init__(self, config: Optional[TemporalCandidatePoolConfig] = None) -> None:
        self.config = config or TemporalCandidatePoolConfig()

    def for_query(
        self,
        query_row: Dict[str, Any],
        candidate_rows: Iterable[Dict[str, Any]],
        index_cutoff_time: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        query_as_of = row_as_of_time(query_row)
        if query_as_of is None:
            return []
        index_cutoff = parse_dt(index_cutoff_time) if index_cutoff_time else query_as_of
        if index_cutoff is None:
            index_cutoff = query_as_of
        max_age = timedelta(days=max(1, self.config.max_age_days))

        strict_pool: List[Dict[str, Any]] = []
        relaxed_pool: List[Dict[str, Any]] = []
        for candidate in candidate_rows:
            candidate_as_of = row_as_of_time(candidate)
            if candidate_as_of is None:
                continue
            if candidate.get("row_id") == query_row.get("row_id"):
                continue
            if candidate_as_of >= query_as_of:
                continue
            if self.config.enforce_index_cutoff and candidate_as_of > index_cutoff:
                continue

            age = query_as_of - candidate_as_of
            if age <= max_age:
                strict_pool.append(candidate)
            relaxed_pool.append(candidate)

        strict_pool.sort(
            key=lambda row: row_as_of_time(row) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        relaxed_pool.sort(
            key=lambda row: row_as_of_time(row) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        if len(strict_pool) >= self.config.min_pool_size:
            return strict_pool
        return relaxed_pool


def split_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out = {"train": [], "validation": [], "test": []}
    for row in rows:
        split = row.get("split")
        if split in out:
            out[split].append(row)
    for key in out:
        out[key].sort(
            key=lambda row: row_as_of_time(row)
            or datetime.min.replace(tzinfo=timezone.utc)
        )
    return out
