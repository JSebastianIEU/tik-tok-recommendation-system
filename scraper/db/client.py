from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional

import psycopg
from psycopg import Connection


def get_database_url(override: Optional[str] = None) -> str:
    """
    Resolve the Postgres connection URL.

    Priority:
    1. Explicit override argument
    2. DATABASE_URL environment variable
    """
    url = override or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. "
            "Set it to something like postgresql://tiktok:tiktok@localhost:5433/tiktok"
        )
    return url


@contextmanager
def get_connection(db_url: Optional[str] = None) -> Iterator[Connection]:
    """
    Context manager that yields a psycopg Connection.

    Uses autocommit by default so each writer function can manage its own
    transaction boundaries explicitly when needed.
    """
    conn = psycopg.connect(get_database_url(db_url), autocommit=False)
    try:
        yield conn
        conn.commit()
    except Exception:  # noqa: BLE001
        conn.rollback()
        raise
    finally:
        conn.close()

