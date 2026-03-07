from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root or scraper/ (for DATABASE_URL, MS_TOKEN)
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "scraper" / ".env")

from scraper.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

