## EDA Architecture

This folder is a reproducible exploratory data analysis workspace layered on top of scraper retrieval contracts.

### Architecture Goals

- Enforce stable data contracts at EDA ingestion boundaries.
- Keep lineage and run manifests for reproducibility.
- Separate immutable raw extracts from curated analytical data.
- Enable headless notebook execution in CI/local pipelines.
- Produce quality scorecards and report artifacts automatically.
- Track EDA runs in a lightweight registry.

### Folder Design

- `config/`: profile-based DB resolution (`profiles.yaml`).
- `configs/`: extraction plans (dataset/limit/since/all_pages/format).
- `sql/`: SQL templates for domain sanity checks.
- `extracts/bronze/`: immutable retrieval outputs (template committed, data local).
- `extracts/silver/`: curated outputs (template committed, data local).
- `metadata/`: JSONL run registry (`runs.jsonl`).
- `notebooks/`: exploratory notebooks.
- `reports/latest/`: generated summary and figures.
- `src/`: contracts, loaders, quality, lineage, plotting/feature templates, notebook runner.
- `pipeline.py`: orchestrates extraction, contracts, quality, lineage, and reports.

### Mock JSONL at repo root (baseline / smoke tests)

The small TikTok-shaped fixture under `data/mock/` (e.g. `tiktok_posts_mock.jsonl`) is **not** the EDA scraper contract. It is validated by **`src/common/schemas.TikTokPost`**: ISO-like `video_meta.language`, normalized `#hashtag` tokens, and shared **engagement** definitions (`engagement_total`, `engagement_rate`) used by `src/baseline/baseline_stats.py` and documented in the generated baseline report. Production training uses **`src/recommendation/contracts.py`** instead.

### Data Contracts (EDA Boundary)

`eda/src/contracts.py` validates required columns per dataset:

- `videos`: `video_id`, `created_at`, `video_author_id`
- `comments`: `comment_id`, `video_id`, `text`, `row_created_at`
- `authors`: `author_id`, `row_created_at`
- `full`: `video_id`, `created_at`, `comments`

Runs fail contract checks only when required columns are missing/null-heavy in extracted rows; results are stored in manifest.

### Lineage + Manifest

Each run writes:

- `manifest.json`: run metadata, dataset outputs, contract checks, quality metrics.
- `quality_report.json`: per-dataset scorecards.

Lineage includes:

- `generated_at`
- `plan_path`
- `git_sha` (if available)
- DB fingerprint (host/port/database/user, no password)

### Bronze/Silver Policy

- Bronze: immutable raw retrieval outputs.
- Silver: curated derivative outputs built from Bronze in deterministic transforms.

Implemented Silver transforms (`eda/src/silver.py`) include:

- Type normalization (timestamps and numeric fields).
- Primary-key deduplication (`video_id`, `comment_id`, `author_id`).
- Text normalization for comments.
- Lightweight derived fields:
  - videos/full: `engagement_rate`
  - comments/full comments: `text_length`, `is_reply`
  - authors: `comments_per_video`

### Notebook Execution Pipeline

`eda/src/notebook_runner.py` provides a headless execution hook using `papermill`.

- If `papermill` is not installed, it raises a clear error.
- This keeps notebook automation optional but pipeline-ready.

### Quality Scorecard

`eda/src/quality.py` computes baseline quality metrics per dataset:

- row count
- duplicate rate on primary key
- null-rate map by sampled keys

These metrics are persisted in manifests and in `quality_report.json`.

### Run Registry (Lightweight Metadata Store)

`eda/src/metadata_store.py` appends each manifest as one JSON line in:

- `eda/metadata/runs.jsonl`

This is a lightweight, append-only metadata log without requiring additional DB setup.

### Profiles and Secrets

- Profiles are in `eda/config/profiles.yaml`.
- Profiles reference env vars only (`db_env_var`), never inline credentials.
- Use `eda/.env.example` as a template.

### Usage

1. Set DB env variable used by your profile.
2. Run extraction plan:

```bash
python3 scripts/eda_bootstrap.py \
  --plan eda/configs/plan.example.yaml \
  --profile dev \
  --build-silver
```

3. Inspect outputs:

- `eda/extracts/bronze/<run_id>/manifest.json`
- `eda/extracts/bronze/<run_id>/quality_report.json`
- `eda/extracts/silver/<run_id>/` (when `--build-silver` is enabled)
- `eda/reports/latest/summary.md`
- `eda/metadata/runs.jsonl`

### Notes

- Prefer retrieval APIs (`python -m scraper export-data`) over notebook-local SQL.
- Keep large extract files out of git.
- Silver transforms are deterministic and versioned in `eda/src/silver.py`.
