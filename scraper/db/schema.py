from __future__ import annotations

from pathlib import Path
from typing import Optional

from .client import get_connection


DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent / "init" / "001_schema.sql"


def apply_schema(*, db_url: Optional[str] = None, schema_path: Optional[str] = None) -> Path:
    path = Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")

    sql = path.read_text(encoding="utf-8")
    with get_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    return path

