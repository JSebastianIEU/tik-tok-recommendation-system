"""Shared helpers for pipeline CLI scripts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_iso_datetime(value: str) -> datetime:
    """Parse an ISO-8601 string (with optional Z suffix) into a UTC datetime."""
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def to_jsonable(value: Any) -> Any:
    """Recursively convert datetime/Path objects for JSON serialization."""
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {key: to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [to_jsonable(inner) for inner in value]
    return value
