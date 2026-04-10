# Training Pipeline Guide

## Prerequisites

- Python 3.11+
- Install dependencies: `pip install -r requirements.txt`
- Set the Supabase Postgres connection string as an environment variable:
  ```bash
  export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
  ```

## Quick Start (Full Pipeline)

Run all 7 steps in sequence:

```bash
python scripts/train_full_pipeline.py
```

Or with an explicit DB URL:

```bash
python scripts/train_full_pipeline.py --db-url "$DATABASE_URL"
```

## Pipeline Steps

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `backfill_video_hashtags.py` | Parse `#hashtags` from video captions and populate the `video_hashtags` bridge table |
| 2 | `export_db_contract_bundle.py` | Export a canonical contract bundle (videos, snapshots, comments, authors) from Supabase |
| 3 | *(internal)* | Build comment intelligence snapshots (8 intent classes, alignment scores) |
| 4 | `build_training_datamart.py` | Build training data mart with features, labels, z-scores, and pairwise rows |
| 5 | `train_recommender.py` | Train Phase 1: hybrid retriever (5 branches) + baseline ranker + graph embeddings |
| 6 | `train_phase2_reranker.py` | Train Phase 2: learned pairwise reranker (LightGBM, bootstrap from datamart) |
| 7 | *(internal)* | Fit per-block fabric score calibration from engagement z-scores |

## Skip Options

```bash
python scripts/train_full_pipeline.py --skip-backfill      # skip hashtag backfill (step 1)
python scripts/train_full_pipeline.py --skip-comments       # skip comment intelligence (step 3)
python scripts/train_full_pipeline.py --skip-calibration    # skip fabric calibration (step 7)
```

## Running Individual Steps

Each step can be run standalone. Use `--help` for full options:

```bash
# Step 1: Backfill hashtags
python scripts/backfill_video_hashtags.py --db-url "$DATABASE_URL"

# Step 2: Export contract bundle
python scripts/export_db_contract_bundle.py --db-url "$DATABASE_URL" \
  --output-dir artifacts/contracts

# Step 4: Build datamart from manifest
python scripts/build_training_datamart.py --manifest-path artifacts/contracts/<manifest_id> \
  --output-json artifacts/datamart/training_datamart.json

# Step 5: Train recommender
python scripts/train_recommender.py artifacts/datamart/training_datamart.json \
  --artifact-root artifacts/recommender \
  --objectives reach,engagement,conversion

# Step 6: Train reranker
python scripts/train_phase2_reranker.py \
  --base-bundle-dir artifacts/recommender/latest \
  --bootstrap-datamart-json artifacts/datamart/training_datamart.json \
  --artifact-root artifacts/recommender
```

## Expected Outputs

All artifacts are written to `artifacts/` (gitignored):

```
artifacts/
  contracts/              # Contract manifests and bundle JSON
    <manifest_id>/
    bundle_export.json
  datamart/
    training_datamart.json  # Training data mart (features + labels + pairs)
  recommender/
    <bundle_id>/            # Trained model bundle
    latest -> <bundle_id>   # Symlink to latest bundle
  comment_intelligence/
    features/               # Comment intelligence snapshots (parquet)
  features/
    fabric_calibration.json # Per-block calibration coefficients
  pipeline_report.json      # Pipeline run summary
```

## Verification

After training, verify with:

```bash
# Run offline evaluation
make eval-recommender

# Serve the recommendation API
python scripts/serve_recommender.py --bundle-dir artifacts/recommender/latest --port 8081

# Test a recommendation request
curl -X POST http://localhost:8081/v1/recommendations \
  -H "Content-Type: application/json" \
  -d '{"objective": "engagement", "query": {"description": "fitness tutorial", "hashtags": ["fitness"]}, "candidates": [], "top_k": 10}'
```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'psycopg'`**: Install with `pip install "psycopg[binary]>=3.2"`
- **`MemoryError` on large datamarts**: The pipeline uses streaming JSON I/O — ensure at least 4GB free RAM
- **Slow training (>1hr)**: Reduce dataset size with `--retrieve-k 50` or use `--no-graph --no-trajectory` flags on `train_recommender.py`
