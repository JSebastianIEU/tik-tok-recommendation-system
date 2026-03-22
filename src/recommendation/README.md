# Recommendation Components (Python)

This package contains Python-native recommendation modeling components.

## Implemented

- `contracts.py`:
  - canonical contract entities (`contract.v1`)
  - raw JSONL normalization to canonical bundle
  - duplicate `video_id` support as time-series snapshots (single canonical video + many snapshots)
  - referential integrity checks
  - point-in-time (`as_of_time`) policy checks
  - pre/post-publication feature access policy checks
  - timestamp precedence for snapshots with optional strict mode
- `datamart.py`:
  - training data mart builder (`datamart.v1`)
  - split-aware, train-only normalization for targets
  - right-censoring for incomplete horizons
  - train-safe author-baseline residualization (leave-one-out for train rows)
  - deterministic time-based train/validation/test splits
  - typed output contracts (`TrainingRow`, `PairTrainingRow`, `TrainingDataMart`)
  - structured exclusion telemetry (`excluded_video_records`, `excluded_by_reason`)
  - optional pair-row generation for retrieval/ranking training
- `learning/`:
  - temporal candidate pooling (`candidate_as_of_time < query_as_of_time`)
  - deterministic negative sampling (hard / semi-hard / easy mix)
  - hybrid retriever training/inference (BM25-or-TFIDF sparse + sentence-transformers-or-char-TFIDF dense)
  - objective-specific rankers (`reach`, `engagement`, `conversion`) with calibration
  - artifact registry + compatibility checks + offline evaluator metrics
- `service.py`:
  - FastAPI serving layer:
    - `GET /v1/health`
    - `POST /v1/recommendations`
  - explicit degradable error payloads when runtime/models are unavailable

## Entry points

- `build_training_data_mart(bundle, config=...)`
- `build_training_data_mart_from_jsonl(raw_jsonl, as_of_time, config=..., strict_timestamps=False, fail_on_warnings=False)`
- `train_recommender_from_datamart(datamart, artifact_root, config=...)`
- `RecommenderRuntime(bundle_dir).recommend(...)`
