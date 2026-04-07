## TikTok Scraper

This scraper supports both:

- JSON-only collection (local files)
- Persistent Postgres/Supabase storage (recommended for team workflows)

## What You Get

- One JSONL dataset per teammate (separate files under `scraper/data/raw/`)
- Scraped video URL + metadata + author stats + many comments
- Easy final merge into one JSONL file (`python -m scraper merge-json ...`)

## Team Configs

- `scraper/configs/juan_sebastian.yaml`
- `scraper/configs/jad_chebly.yaml`
- `scraper/configs/omar_mekawi.yaml`
- `scraper/configs/alp_arslan.yaml`
- `scraper/configs/arielo_moreira.yaml`
- `scraper/configs/fares_rafiq.yaml`

Each config already includes:

- Assigned keywords/hashtags
- `db_url: null` (JSON-only mode)
- Dedicated output JSONL path
- Unique `source_label`

## Install (Windows PowerShell and macOS)

### PowerShell (Windows)

```powershell
pip install -r scraper/requirements.txt
python -m playwright install
```

### macOS

```bash
pip install -r scraper/requirements.txt
python -m playwright install
```

Optional but recommended for better discovery:

- Set `MS_TOKEN` in your environment.

PowerShell:

```powershell
$env:MS_TOKEN="your_ms_token_here"
```

macOS:

```bash
export MS_TOKEN="your_ms_token_here"
```

## Run Per Teammate (No DB Needed)

### PowerShell

```powershell
python -m scraper run --config "scraper/configs/juan_sebastian.yaml"
python -m scraper run --config "scraper/configs/jad_chebly.yaml"
python -m scraper run --config "scraper/configs/omar_mekawi.yaml"
python -m scraper run --config "scraper/configs/alp_arslan.yaml"
python -m scraper run --config "scraper/configs/arielo_moreira.yaml"
python -m scraper run --config "scraper/configs/fares_rafiq.yaml"
```

### macOS

```bash
python -m scraper run --config "scraper/configs/juan_sebastian.yaml"
python -m scraper run --config "scraper/configs/jad_chebly.yaml"
python -m scraper run --config "scraper/configs/omar_mekawi.yaml"
python -m scraper run --config "scraper/configs/alp_arslan.yaml"
python -m scraper run --config "scraper/configs/arielo_moreira.yaml"
python -m scraper run --config "scraper/configs/fares_rafiq.yaml"
```

## Merge JSON Datasets (No DB Needed)

### PowerShell

```powershell
python -m scraper merge-json --output "scraper/data/raw/team_merged.jsonl" --input "scraper/data/raw/juan_sebastian_fitness_self_improvement.jsonl" --input "scraper/data/raw/jad_chebly_finance_investing.jsonl" --input "scraper/data/raw/omar_mekawi_relationships_dating.jsonl" --input "scraper/data/raw/alp_arslan_politics_social_issues.jsonl" --input "scraper/data/raw/arielo_moreira_lifestyle_luxury_aspirational.jsonl" --input "scraper/data/raw/fares_rafiq_entertainment_trends_viral.jsonl"
```

### macOS

```bash
python -m scraper merge-json \
  --output "scraper/data/raw/team_merged.jsonl" \
  --input "scraper/data/raw/juan_sebastian_fitness_self_improvement.jsonl" \
  --input "scraper/data/raw/jad_chebly_finance_investing.jsonl" \
  --input "scraper/data/raw/omar_mekawi_relationships_dating.jsonl" \
  --input "scraper/data/raw/alp_arslan_politics_social_issues.jsonl" \
  --input "scraper/data/raw/arielo_moreira_lifestyle_luxury_aspirational.jsonl" \
  --input "scraper/data/raw/fares_rafiq_entertainment_trends_viral.jsonl"
```

`merge-json` behavior:

- Deduplicates by `normalized.video.video_id` when available (fallback: URL)
- Keeps the richer duplicate record (more useful fields/comments)
- Writes one merged JSONL output

## Full-Scale Scraping (Hashtags + Keywords → Supabase)

Scrape **hashtags and keywords** at scale and persist to **Supabase** (or any Postgres). No user scraping (avoids bot detection).

### Recommended operating modes

For stability and better recovery, run in two phases:

1. Video index mode (primary pipeline): collect videos + metadata reliably.
2. Comment enrichment mode (secondary pipeline): enrich comments for already stored `video_id`s.

This repo now includes both commands:

- `scrape-all` for full source traversal.
- `scrape-comments` for comment-only enrichment from existing DB videos.

### Option A: Supabase (recommended for persistent team storage)

1. **Create project** at [supabase.com](https://supabase.com) → New project
2. **Get connection string**: Settings → Database → Connection string → URI
3. **Set env**:
   ```bash
   cp scraper/.env.example scraper/.env
   # Edit scraper/.env: paste DATABASE_URL and MS_TOKEN
   ```
4. **Init schema** (once):
   ```bash
   python -m scraper init-db
   ```
5. **Scrape**:
   ```bash
   python -m scraper scrape-all scraper/configs/full_scale.yaml
   ```

Data persists in Supabase. Teammates use the same `DATABASE_URL` to share data.
Each run writes a JSON summary file in `scraper/data/raw/` and tracks source-level checkpoint state in DB (`scrape_source_jobs`) so reruns can resume.

### Send data to Supabase (quick reference)

#### 1) Send new scraped data directly to Supabase

Set your Supabase Postgres URL, then run scraper commands normally:

```bash
export DATABASE_URL="postgresql://postgres:<PASSWORD>@db.<PROJECT_REF>.supabase.co:5432/postgres"
export MS_TOKEN="<your_ms_token>"
python -m scraper init-db
python -m scraper scrape-all scraper/configs/full_scale.yaml
```

Anything persisted by the scraper writer is inserted into Supabase tables.

#### 2) Backfill existing Docker Postgres data into Supabase

If you already scraped into a local Docker Postgres DB, dump then restore into Supabase:

```bash
# Export from local Docker Postgres
docker exec tiktok-postgres pg_dump -U tiktok tiktok > backup.sql

# Import into Supabase Postgres
/opt/homebrew/opt/libpq/bin/psql "postgresql://postgres:<PASSWORD>@db.<PROJECT_REF>.supabase.co:5432/postgres" < backup.sql
```

Notes:

- `pg_dump` is read-only on source DB (it copies; it does not delete source data).
- If your network is IPv4-only and direct host is IPv6-only, use Supabase Session Pooler URI instead of direct DB host.
- Run `python -m scraper init-db` against Supabase before restore if schema is not present.

### Team Supabase Setup (shared DB, local runs)

Use this flow when multiple teammates run the scraper locally and all write into the same Supabase project.

1. Share secrets securely (do not commit to git):
   - `DATABASE_URL` (prefer Supabase Session Pooler URI)
   - `MS_TOKEN` (each teammate can use their own token)
2. Each teammate creates local env file (`scraper/.env`):
   ```env
   DATABASE_URL=postgresql://postgres:<PASSWORD>@<POOLER_HOST>:6543/postgres?sslmode=require
   MS_TOKEN=<your_ms_token>
   TIKTOK_BROWSER=webkit
   TIKTOK_HEADLESS=false
   ```
3. Initialize schema once (one teammate only):
   ```bash
   python -m scraper init-db
   ```
4. Run assigned sources locally and write to shared Supabase:
   ```bash
   python -m scraper scrape-all scraper/configs/full_scale.yaml
   ```
5. Coordinate source ownership to reduce overlap (split hashtags/keywords per teammate).
6. Verify ingest:
   ```sql
   SELECT COUNT(*) AS videos FROM videos;
   SELECT COUNT(*) AS comments FROM comments;
   ```

Notes:

- Default behavior updates/upserts existing videos (fresh snapshots/metadata). Add `--skip-existing` only when you explicitly want insert-only behavior.
- Use `--no-resume` only for intentional full reruns/debugging.
- If direct DB host fails from local network, switch to Supabase Session Pooler connection string.
- CI scraper workflow defaults to `comments=0` and `replies=0` for stability. Run local/manual enrichment when you need comment depth.
- Use GitHub workflow `.github/workflows/scraper-comments.yml` for manual comment-only enrichment runs.
- `scraper-comments.yml` is decoupled from the core scraper pipeline and non-blocking. Its schedule is disabled by default (`SCRAPER_COMMENTS_SCHEDULE_ENABLED=false`).

### Option B: Local Docker (solo dev)

```bash
cd scraper && docker compose up -d
export DATABASE_URL="postgresql://tiktok:tiktok@localhost:5433/tiktok"
export MS_TOKEN="your_ms_token"
python -m scraper init-db
python -m scraper scrape-all scraper/configs/full_scale.yaml
```

### Tips

- **Default update mode**: Existing videos are upserted so metadata/snapshots stay fresh.
- **`--skip-existing`**: Optional insert-only mode when you want to avoid touching existing `video_id`s.
- **`--init-db`**: Apply schema before first scrape.
- **`--delay 10`**: Increase seconds between sources if rate-limited.
- **Resume behavior**: `scrape-all` resumes by default and skips completed sources; use `--no-resume` to force rerun.
- **Summary output**: set `--summary-path` to control where run metrics JSON is written.
- **Customize**: Edit `scraper/configs/full_scale.yaml` for hashtags, keywords, counts.
- **Config compatibility**: `scrape-all` now supports both:
  - full-scale style (`hashtags/keywords` as `{name, count}` objects)
  - member style (`hashtags/keywords` as string lists with `per_query_video_limit`)
- **Config-driven runtime knobs** (overridable by CLI flags): `comments`, `replies`, `min_likes_for_replies`, `delay`, `retry_empty`, `retry_delay`, `max_consecutive_empty`, `skip_existing`, `no_resume`, `modes_enabled`.
- **Bot detection**: Scraper uses `headless=false` and `webkit` by default. If issues persist, try `TIKTOK_BROWSER=chromium` or `playwright install webkit`.
- **DB config precedence**: CLI `--db-url` > `DATABASE_URL` env > config `db_url`.
- **DB commit batching**: tune with `SCRAPER_DB_COMMIT_EVERY` (default `50`).

## Local Run Quickstart (Recommended)

Use this when you want to run scraping on your own machine (more stable than CI for browser-based TikTok scraping).

1. Create and activate a virtualenv:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r scraper/requirements.txt
   python -m playwright install --with-deps webkit
   ```
3. Set required environment variables:
   ```bash
   export DATABASE_URL="postgresql://postgres:<PASSWORD>@db.<PROJECT_REF>.supabase.co:5432/postgres"
   export MS_TOKEN="<your_ms_token>"
   ```
4. Initialize DB schema once:
   ```bash
   python -m scraper init-db
   ```
5. Run full-scale scraping:
   ```bash
   python -m scraper scrape-all scraper/configs/full_scale.yaml --no-resume --comments 5 --replies 2 --delay 15
   ```

Notes:

- If `python` is not found, use `python3` for all commands.
- `--no-resume` forces all sources to run even if previously completed.
- Omit `--no-resume` to reuse checkpoint state and skip completed sources.
- If TikTok blocks requests, try:
  - `export TIKTOK_BROWSER=webkit`
  - `export TIKTOK_HEADLESS=false`

## Optional PostgreSQL Mode (pipeline)

If later you want DB persistence for the Selenium pipeline:

- Set `db_url` in config (or `DATABASE_URL` env var)
- Use:
  - `python -m scraper init-db ...`
  - `python -m scraper merge ...`

## CLI Commands

- `python -m scraper run --config <path>`
- `python -m scraper scrape-all <full_scale.yaml>` – hashtags + keywords into Supabase/Postgres (default behavior updates existing videos)
- `python -m scraper scrape-comments --limit 300 --comments 5 --replies 2` – enrich comments for videos already stored in DB
- `python -m scraper scrape-comments --max-attempts-per-video 5 --retry-backoff-base-sec 900 --stale-running-minutes 60` – tune retry/exhaustion behavior for comment enrichment jobs
- `python -m scraper scrape-comments --max-existing-comments 2 --summary-path /tmp/comment_summary.json` – include low-coverage videos and write structured summary JSON
- `python -m scraper export-data --dataset full --limit 1000` – retrieve pre-joined datasets (`full|videos|comments|authors`)
- `python -m scraper backfill-comment-lineage` – recompute `comments.root_comment_id` + `comments.comment_level` for existing rows
- `python -m scraper merge-json --input <file> --input <file> --output <file>`
- Optional DB mode:
  - `python -m scraper init-db --db-url <url>`
  - `python -m scraper merge --target-db <url> --source-db <url> ...`

`scrape-comments` uses `comment_enrichment_jobs` in Postgres/Supabase to track `pending/running/failed/exhausted/done` per `video_id`. This makes reruns deterministic and safe across teammates.

Workflow quality gates for comment enrichment:

- `max_video_error_rate`: fails run if `video_fetch_errors / processed` is above configured threshold.
- `min_comments_written`: fails run if written comment volume is too low (applies only when `processed > 0`).
- `summary artifact`: `/tmp/comment_enrichment_summary.json` is uploaded on every run for debugging/trend tracking.

## Data Retrieval API (Pre-Joined Reads)

Use retrieval functions/CLI when you need model-ready data views from DB without writing ad-hoc SQL.

Datasets:

- `full`: 1 row per video with nested `comments` JSON array.
- `videos`: 1 row per video (+ author + latest snapshot + hashtags).
- `comments`: 1 row per comment (+ parent video + comment author + latest comment snapshot).
- `authors`: 1 row per author (+ aggregated video/comment summaries).

Safety defaults:

- Default `--limit` is `1000`.
- Hard cap is `10000` rows/page.
- Full-table retrieval requires explicit `--all`.

Examples:

```bash
# Page through videos (safe default)
python -m scraper export-data --dataset videos --limit 1000 --out /tmp/videos.jsonl

# Incremental comments export since a timestamp
python -m scraper export-data --dataset comments --since 2026-03-01T00:00:00Z --format csv --out /tmp/comments.csv

# Explicit full export (keyset pagination loop)
python -m scraper export-data --dataset full --all --limit 1000 --out /tmp/full.jsonl
```

Comment/reply normalization in DB:

- `comments.parent_comment_id`: direct parent (NULL for top-level comments).
- `comments.root_comment_id`: thread root comment id (self for top-level).
- `comments.comment_level`: depth in thread (`0` top-level, `1+` replies).
