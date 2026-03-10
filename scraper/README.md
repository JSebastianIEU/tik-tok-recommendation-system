## TikTok Scraper (JSON-Only First)

This scraper can now run **without Docker, PostgreSQL, or `psycopg`**.

Default team configs are set to `db_url: null`, so each member generates JSONL datasets only.

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
   python -m scraper scrape-all scraper/configs/full_scale.yaml --skip-existing
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
python -m scraper scrape-all scraper/configs/full_scale.yaml --skip-existing
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

### Option B: Local Docker (solo dev)

```bash
cd scraper && docker compose up -d
export DATABASE_URL="postgresql://tiktok:tiktok@localhost:5433/tiktok"
export MS_TOKEN="your_ms_token"
python -m scraper init-db
python -m scraper scrape-all scraper/configs/full_scale.yaml --skip-existing
```

### Tips

- **`--skip-existing`**: Skips videos already in DB; use on repeated runs.
- **`--init-db`**: Apply schema before first scrape.
- **`--delay 10`**: Increase seconds between sources if rate-limited.
- **Resume behavior**: `scrape-all` resumes by default and skips completed sources; use `--no-resume` to force rerun.
- **Summary output**: set `--summary-path` to control where run metrics JSON is written.
- **Customize**: Edit `scraper/configs/full_scale.yaml` for hashtags, keywords, counts.
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
   python -m scraper scrape-all scraper/configs/full_scale.yaml --skip-existing --no-resume --comments 5 --replies 2 --delay 15
   ```

Notes:

- If `python` is not found, use `python3` for all commands.
- `--no-resume` forces all sources to run even if previously completed.
- Omit `--no-resume` to reuse checkpoint state and skip completed sources.
- If TikTok blocks requests, try:
  - `export TIKTOK_BROWSER=webkit`
  - `export TIKTOK_HEADLESS=false`
  - add proxies via `--proxies-file <path>`

## Optional PostgreSQL Mode (pipeline)

If later you want DB persistence for the Selenium pipeline:

- Set `db_url` in config (or `DATABASE_URL` env var)
- Use:
  - `python -m scraper init-db ...`
  - `python -m scraper merge ...`

## CLI Commands

- `python -m scraper run --config <path>`
- `python -m scraper scrape-all <full_scale.yaml>` – hashtags + keywords into Supabase/Postgres (`--skip-existing` recommended)
- `python -m scraper merge-json --input <file> --input <file> --output <file>`
- Optional DB mode:
  - `python -m scraper init-db --db-url <url>`
  - `python -m scraper merge --target-db <url> --source-db <url> ...`
