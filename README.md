# Tik-Tok-Recommendation-System

This repository has evolved from an early TikTok experimentation scaffold into a recommendation platform with:

- canonical data contracts and temporal validation
- training datamart generation
- deterministic multimodal feature extraction
- recommender training and artifact packaging
- FastAPI serving for online recommendations
- a frontend and Node gateway for report-generation workflows
- control-plane jobs for attribution, drift monitoring, and retrain decisions

The core Python recommendation stack lives in `src/recommendation/`.

For a guided architecture walkthrough, start with [docs/architecture/recommender_overview.md](/Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/docs/architecture/recommender_overview.md).

## Architecture At A Glance

The main pipeline in this repo is:

1. Raw JSONL or exported source data is normalized into a canonical contract bundle.
2. The canonical bundle is transformed into a training datamart with point-in-time and label-horizon safeguards.
3. Optional feature-fabric and comment-intelligence snapshots are materialized for richer retrieval and ranking signals.
4. Recommender artifacts are trained and packaged under `artifacts/recommender/`.
5. The FastAPI recommender service loads the latest compatible bundle and serves ranked candidates.
6. The frontend and Node gateway call the Python service, format reports, collect feedback, and support fallback behavior.
7. Control-plane jobs attribute outcomes, analyze experiments, monitor drift, and gate retraining.

## Repository Map

Core paths:

- `src/recommendation/`: contract layer, datamart builder, feature fabric, comment intelligence, training/inference runtime, control plane
- `scripts/`: operational entrypoints for building data products, training artifacts, serving, evaluation, and control-plane jobs
- `tests/`: service, contract, datamart, fabric, learning, control-plane, scraper, and E2E coverage
- `artifacts/`: generated manifests, feature snapshots, recommender bundles, and control-plane outputs

Supporting paths:

- `frontend/`: React + Vite UI with a Node gateway that proxies to the Python recommender service
- `scraper/`: data acquisition and DB-writing utilities
- `eda/`: exploratory data analysis workspace, metadata, reports, and supporting tests
- `src/baseline/`: lightweight baseline analytics and reporting
- `src/common/`: shared validation and schema helpers for the mock JSONL scaffold path
- `data/mock/`: mock data for local validation, baselines, and smaller end-to-end exercises

## Start Here

If you are onboarding to the current implementation, read files in this order:

1. [docs/architecture/recommender_overview.md](/Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/docs/architecture/recommender_overview.md)
2. [src/recommendation/contracts.py](/Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/src/recommendation/contracts.py)
3. [src/recommendation/datamart.py](/Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/src/recommendation/datamart.py)
4. [src/recommendation/fabric/core.py](/Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/src/recommendation/fabric/core.py)
5. [src/recommendation/learning/inference.py](/Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/src/recommendation/learning/inference.py)
6. [src/recommendation/service.py](/Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/src/recommendation/service.py)
7. [frontend/README.md](/Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/frontend/README.md)

## Common Workflows

Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you only need a subset of the stack:

- Python service only: `pip install -r requirements-service.txt`
- training/offline eval: `pip install -r requirements-training.txt`
- full Python contributor setup: `pip install -r requirements.txt`
- scraper only: `pip install -r scraper/requirements.txt`
- EDA only: `pip install -r eda/requirements-eda.txt`

See [docs/setup_workflows.md](/Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/docs/setup_workflows.md) for the workflow-by-workflow setup guide and environment variable reference.

Run the Python test suite:

```bash
make test
```

Validate the mock dataset:

```bash
make validate-data
```

Build a training datamart:

```bash
make datamart
```

Train recommender artifacts:

```bash
make train-recommender
```

Inspect the latest bundle:

```bash
make eval-recommender
```

Serve the recommender API:

```bash
make serve-recommender
```

Run the frontend locally:

```bash
cd frontend
npm install
npm run dev:all
```

## Main Entry Points

Training and artifact generation:

- `scripts/build_training_datamart.py`
- `scripts/build_feature_fabric_snapshot.py`
- `scripts/build_comment_intelligence_snapshot.py`
- `scripts/train_recommender.py`
- `scripts/train_phase2_reranker.py`

Serving and product flows:

- `scripts/serve_recommender.py`
- `src/recommendation/service.py`
- `frontend/server/index.ts`

Operations and evaluation:

- `scripts/eval_recommender.py`
- `scripts/eval_retriever.py`
- `scripts/run_outcome_attribution.py`
- `scripts/run_drift_monitor.py`
- `scripts/run_retrain_controller.py`
- `scripts/run_experiment_analysis.py`
- `scripts/run_live_e2e_validation.py`

## Notes

- The recommendation API normalizes the incoming objective alias `community` to the effective objective `engagement`.
- The first iteration of this repo included a small TF-IDF-only retrieval demo under `src/retrieval/` with `build_index.py` / `query_index.py`; that path was removed so retrieval is not confused with the production hybrid retriever in `src/recommendation/learning/retriever.py`. Older `src/common/` and `src/baseline/` helpers for mock JSONL remain for local baselines and validation.
- Generated outputs under `artifacts/` are part of the normal workflow and should be read as build/runtime products rather than source-of-truth code.
