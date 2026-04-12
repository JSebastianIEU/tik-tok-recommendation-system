# Setup Workflows

This repository supports several different workflows. You do not need the same dependencies for all of them.

## Python Dependency Files

- `requirements-base.txt`: shared Python data/runtime dependencies
- `requirements-service.txt`: Python recommender serving stack
- `requirements-training.txt`: recommender training and retrieval extras
- `requirements-dev.txt`: service + training + test/lint tooling
- `requirements.txt`: convenience alias for the full developer environment
- `scraper/requirements.txt`: scraper environment
- `eda/requirements-eda.txt`: EDA-only environment

## Recommended Python Environments

### 1. Full contributor environment

Use this if you plan to run tests, train models, and serve the recommender locally.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Python service only

Use this if you only need the FastAPI recommender service.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-service.txt
```

### 3. Training and offline evaluation

Use this if you need datamart builds, recommender training, or offline evaluation.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-training.txt
```

If you also want tests and linting, install `requirements-dev.txt` instead.

### 4. Scraper only

```bash
python3 -m venv .venv-scraper
source .venv-scraper/bin/activate
pip install -r scraper/requirements.txt
```

### 5. EDA only

```bash
python3 -m venv .venv-eda
source .venv-eda/bin/activate
pip install -r eda/requirements-eda.txt
```

## Common Commands By Workflow

### Python tests and lint

```bash
make test
make lint
```

### Train and evaluate recommender artifacts

```bash
make datamart
make train-recommender
make eval-recommender
```

### Serve the recommender API

```bash
make serve-recommender
```

### Frontend and Node gateway

```bash
cd frontend
npm install
npm run dev:all
```

### Scraper

```bash
python -m scraper run --config "scraper/configs/juan_sebastian.yaml"
```

## Environment Variables

Python variables below are read through `src/common/config.py` (`settings` properties). That module loads `.env` and `.env.local` from the **repository root** when imported (without overriding variables already set in the process environment). Import order matters: anything that needs env files should transitively import `src.common.config` before reading settings—starting the FastAPI app via `src.recommendation.service` does this.

### Python recommender service

| Variable | Default (if unset) | Purpose |
|----------|-------------------|---------|
| `RECOMMENDER_BUNDLE_DIR` | `artifacts/recommender_real/latest` | Trained recommender bundle directory |
| `FABRIC_CALIBRATION_PATH` | *(empty)* | Optional JSON calibration file for `FeatureFabric` |
| `HASHTAG_RECOMMENDER_DIR` | `artifacts/hashtag_recommender` | Hashtag recommender artifact directory |
| `RECOMMENDER_HOST` | `127.0.0.1` | Documented default for local serve scripts (CLI may override) |
| `RECOMMENDER_PORT` | `8081` | Same as above |
| `DATABASE_URL` | *(empty)* | Postgres URL for training scripts and scraper-backed flows that read this module |
| `DEFAULT_TOP_K` | `3` | Demo/baseline helpers (`DEFAULT_TOP_K` export) |
| `DEFAULT_REPORT_PATH` | `src/baseline/report.md` | Baseline report output path (`DEFAULT_REPORT_PATH` export) |

Fast serving refresh after scraping/comments:

```bash
PYTHONPATH=. python3 scripts/refresh_serving_bundle.py --db-url "$DATABASE_URL"
```

This refreshes the Supabase-derived contract bundle and retriever-backed serving
bundle without running full ranker retraining.

CLI scripts often accept `--db-url` or `--bundle-dir`; those flags typically override or set `os.environ` before the heavy modules load, and `settings` reads the environment **at access time**, so overrides still apply.

### Python scraper-related (also on `settings`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `MS_TOKEN` | *(empty)* | TikTok / scraper token when required |
| `SCRAPER_DB_WRITE_RETRIES` | `1` | DB write retry count |
| `SCRAPER_DB_COMMIT_EVERY` | `50` | Batch commit size |

### Frontend and Node gateway

Common variables:

- `PORT`
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_MODEL`
- `DEEPSEEK_BASE_URL`
- `RECOMMENDER_ENABLED`
- `RECOMMENDER_BASE_URL`
- `RECOMMENDER_TIMEOUT_MS`
- `RECOMMENDER_RETRY_COUNT`
- `RECOMMENDER_FALLBACK_BUNDLE_DIR`
- `RECOMMENDER_FEEDBACK_ENABLED`
- `RECOMMENDER_FEEDBACK_DB_URL`
- `RECOMMENDER_EXPERIMENT_DEFAULT_ID`
- `RECOMMENDER_EXPERIMENT_TREATMENT_RATIO`
- `RECOMMENDER_EXPERIMENT_SALT`

See `frontend/server/config.ts` for the full list. A starter file is available at `frontend/.env.example`.

### Scraper

Common variables:

- `DATABASE_URL`
- `MS_TOKEN`
- `TIKTOK_BROWSER`
- `SCRAPER_DB_COMMIT_EVERY`
- `TIKTOK_MAX_SESSION_RECOVERIES`

### EDA

Example DB variables are listed in `eda/.env.example`.

## Notes

- `requirements.txt` remains the easiest choice for a full Python contributor setup.
- The frontend uses npm dependencies from `frontend/package.json`; it does not use the Python requirements files.
- The scraper has its own dependency set and should usually live in a separate environment from the main recommender stack.
