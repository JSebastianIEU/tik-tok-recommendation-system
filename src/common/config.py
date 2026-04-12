"""Centralized configuration loaded from environment variables and .env files.

Loads ``.env`` and ``.env.local`` from the repository root (non-destructive:
existing process env wins). Import this module early in Python entrypoints so
local env files apply before reading variables.

Usage:
    from src.common.config import settings
    print(settings.database_url)
    print(settings.recommender_bundle_dir)

Variable reference: ``docs/setup_workflows.md`` (Environment variables).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(_ROOT / ".env", override=False)
load_dotenv(_ROOT / ".env.local", override=False)


class _Settings:
    """Lazy accessor for environment-backed configuration.

    All values are read from ``os.environ`` at access time so they respect
    any overrides applied by CLI scripts (e.g. ``serve_recommender.py``
    sets ``RECOMMENDER_BUNDLE_DIR`` before import).
    """

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _str(key: str, default: str = "") -> str:
        return os.getenv(key, default).strip()

    @staticmethod
    def _int(key: str, default: int = 0) -> int:
        raw = os.getenv(key, "").strip()
        return int(raw) if raw else default

    @staticmethod
    def _float(key: str, default: float = 0.0) -> float:
        raw = os.getenv(key, "").strip()
        return float(raw) if raw else default

    @staticmethod
    def _bool(key: str, default: bool = False) -> bool:
        raw = os.getenv(key, "").strip().lower()
        if not raw:
            return default
        return raw in ("1", "true", "yes")

    # -- database ---------------------------------------------------------

    @property
    def database_url(self) -> str:
        return self._str("DATABASE_URL")

    # -- Python recommender service ---------------------------------------

    @property
    def recommender_bundle_dir(self) -> str:
        return self._str("RECOMMENDER_BUNDLE_DIR", "artifacts/recommender_real/latest")

    @property
    def recommender_host(self) -> str:
        return self._str("RECOMMENDER_HOST", "127.0.0.1")

    @property
    def recommender_port(self) -> int:
        return self._int("RECOMMENDER_PORT", 8081)

    @property
    def fabric_calibration_path(self) -> str:
        return self._str("FABRIC_CALIBRATION_PATH")

    @property
    def recommender_corpus_bundle_path(self) -> str:
        return self._str("RECOMMENDER_CORPUS_BUNDLE_PATH")

    @property
    def hashtag_recommender_dir(self) -> str:
        return self._str("HASHTAG_RECOMMENDER_DIR", "artifacts/hashtag_recommender")

    # -- scraper ----------------------------------------------------------

    @property
    def ms_token(self) -> str:
        return self._str("MS_TOKEN")

    @property
    def scraper_db_write_retries(self) -> int:
        return self._int("SCRAPER_DB_WRITE_RETRIES", 1)

    @property
    def scraper_db_commit_every(self) -> int:
        return self._int("SCRAPER_DB_COMMIT_EVERY", 50)

    # -- defaults used by baseline / demo scripts -------------------------

    @property
    def default_top_k(self) -> int:
        return self._int("DEFAULT_TOP_K", 3)

    @property
    def default_report_path(self) -> str:
        return self._str("DEFAULT_REPORT_PATH", "src/baseline/report.md")


settings = _Settings()

DEFAULT_TOP_K = settings.default_top_k
DEFAULT_REPORT_PATH = settings.default_report_path
