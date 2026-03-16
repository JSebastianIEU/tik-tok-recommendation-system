from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from urllib.parse import urlparse


def _safe_git_sha() -> str | None:
    env_sha = os.getenv("GITHUB_SHA")
    if env_sha:
        return env_sha
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out or None
    except Exception:  # noqa: BLE001
        return None


def _db_fingerprint(db_url: str) -> dict[str, str | None]:
    parsed = urlparse(db_url)
    return {
        "scheme": parsed.scheme or None,
        "host": parsed.hostname or None,
        "port": str(parsed.port) if parsed.port else None,
        "database": parsed.path.lstrip("/") or None,
        "user": parsed.username or None,
    }


def build_lineage(*, db_url: str, plan_path: str) -> dict[str, object]:
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "plan_path": plan_path,
        "git_sha": _safe_git_sha(),
        "db_fingerprint": _db_fingerprint(db_url),
    }
