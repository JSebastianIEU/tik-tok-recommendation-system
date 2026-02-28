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

## Optional PostgreSQL Mode

If later you want DB persistence again:

- Set `db_url` in config (or `DATABASE_URL` env var)
- Use:
  - `python -m scraper init-db ...`
  - `python -m scraper merge ...`

## CLI Commands

- `python -m scraper run --config <path>`
- `python -m scraper merge-json --input <file> --input <file> --output <file>`
- Optional DB mode:
  - `python -m scraper init-db --db-url <url>`
  - `python -m scraper merge --target-db <url> --source-db <url> ...`

