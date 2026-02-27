## Scraper subsystem

This folder contains optional, **non-CI-critical** tooling for collecting real TikTok data and persisting it to Postgres. It is isolated from the core `requirements.txt` so that experiments here do not affect the main scaffold.

### Components

- `requirements-scraper.txt` — Python dependencies for all scrapers (Selenium, TikTokApi/Playwright, Postgres client).
- `docker-compose.yml` — local Postgres instance for development.
- `db/` — schema DDL and small DB helpers.
- `tiktok_post_scraper.py` — Selenium-based scraper for a single TikTok post (or batches of URLs).
- `scrape_tiktok_sample.py` — TikTokApi-based sampler for trending / hashtag / user videos.

### Setup

1. Create and activate a virtualenv (recommended):

   ```bash
   python -m venv .scraper-venv
   source .scraper-venv/bin/activate
   ```

2. Install scraper-specific dependencies:

   ```bash
   pip install -r scraper/requirements-scraper.txt
   # TikTokApi/Playwright browsers (once):
   python -m playwright install
   ```

3. Start Postgres via Docker Compose (runs on `localhost:5433` by default):

   ```bash
   cd scraper
   docker compose up -d
   ```

   The default connection URL is:

   ```bash
   export DATABASE_URL="postgresql://tiktok:tiktok@localhost:5433/tiktok"
   ```

### Running the Selenium post scraper

Scrape a single TikTok video URL, optionally downloading media and writing to Postgres:

```bash
PYTHONPATH=. DATABASE_URL=$DATABASE_URL \
python scraper/tiktok_post_scraper.py \
  "https://www.tiktok.com/@user/video/123" \
  --comments 5 \
  --source single_url
```

Key behaviours:

- Uses Selenium + `__UNIVERSAL_DATA_FOR_REHYDRATION__` JSON for robust metadata.
- Falls back to DOM scraping for top comments when available.
- Handles TikTok overlays (cookie banner, communication popups) including elements rendered in Shadow DOM.

### Running the TikTokApi sampler

The sampler uses the unofficial `TikTokApi` wrapper on top of Playwright. You must provide a valid `MS_TOKEN` (cookie value) and have Playwright browsers installed.

```bash
export MS_TOKEN="your_ms_token_here"
PYTHONPATH=. DATABASE_URL=$DATABASE_URL \
python scraper/scrape_tiktok_sample.py trending \
  --count 100 \
  --output data/raw/trending_100.jsonl
```

Other modes:

- Hashtag: `python scraper/scrape_tiktok_sample.py hashtag --name kencarson --count 300`
- User: `python scraper/scrape_tiktok_sample.py user --name rollingloud --count 150`

Each run:

- Writes a flattened JSONL file (for offline analysis).
- Inserts authors, videos, audio tracks, and snapshot rows into Postgres when `DATABASE_URL` is set.

