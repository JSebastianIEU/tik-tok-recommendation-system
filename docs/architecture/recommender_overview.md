# Recommender Overview

This document is the shortest useful map of the current recommendation platform in this repository.

## What This System Does

At a high level, the platform turns TikTok-like content records into trainable datasets, packages ranking artifacts, serves recommendations online, and monitors the health of the system after deployment.

The architecture is centered on a Python recommendation stack in `src/recommendation/`, with a frontend and Node gateway in `frontend/` for interactive report-generation and feedback workflows.

## End-To-End Pipeline

1. Source data is loaded from raw JSONL exports or scraper-backed extracts.
2. `contracts.py` normalizes that data into a canonical bundle with referential integrity, temporal semantics, watermark metadata, and quality telemetry.
3. `datamart.py` converts the canonical bundle into a training data mart with point-in-time safety, label-horizon censorship, objective labels, and exclusion telemetry.
4. `fabric/` optionally computes deterministic text, structure, audio, and visual feature snapshots.
5. `comment_intelligence/` optionally computes comment-derived alignment and intent signals.
6. `learning/` trains retrievers, baseline rankers, and additive learned rerankers, then writes bundle artifacts under `artifacts/recommender/`.
7. `learning/inference.py` loads those artifacts at runtime, validates compatibility, retrieves candidates, ranks them, and returns explainable recommendation payloads.
8. `service.py` exposes the runtime through FastAPI endpoints for health, compatibility, recommendations, feature extraction, metrics, and hashtag suggestion.
9. `frontend/` calls the Python service through a Node gateway, renders reports, and persists user feedback and experiment metadata.
10. `control_plane.py` and related scripts attribute outcomes, summarize drift, analyze experiments, and gate retraining decisions.

## Core Modules

### `src/recommendation/contracts.py`

Purpose:
- canonical schema definitions
- raw-to-canonical normalization
- temporal policy validation
- feature-access policy validation
- manifest export and replay

This is the foundation of the whole stack. If the contract layer is wrong, every downstream stage becomes unreliable.

### `src/recommendation/datamart.py`

Purpose:
- build trainable rows from canonical bundles
- enforce `as_of_time` and run-cutoff rules
- create objective labels and trajectory targets
- record exclusions and missingness explicitly

This is the boundary between data correctness and modeling.

### `src/recommendation/fabric/`

Purpose:
- deterministic multimodal feature extraction
- extractor registry and compatibility tracking
- missingness semantics and calibrated confidence
- feature snapshot manifests for offline reuse

The fabric is the structured feature layer that enriches training and serving without hiding how signals are produced.

### `src/recommendation/comment_intelligence/`

Purpose:
- comment intent and sentiment summaries
- comment-to-content alignment features
- comment snapshot manifests and transfer priors

This layer is additive, but it is important for explainability and more nuanced ranking support.

### `src/recommendation/learning/`

Purpose:
- retrieval training
- ranking and reranking
- runtime inference orchestration
- artifact compatibility checks
- explainability payload assembly

The most important runtime file here is `learning/inference.py`.

### `src/recommendation/service.py`

Purpose:
- online serving boundary
- request validation
- runtime loading
- error translation
- metrics collection

This is the entrypoint used by the Python recommender service.

### `src/recommendation/control_plane.py`

Purpose:
- outcome attribution
- drift metrics
- retrain trigger logic
- experiment-analysis support

This is the operational layer that helps keep the system healthy after deployment.

## Online Serving Path

The serving path is:

1. A caller sends a request to the FastAPI service.
2. `service.py` validates the request and lazy-loads the active artifact bundle.
3. `RecommenderRuntime` in `learning/inference.py` validates bundle compatibility and routing constraints.
4. The runtime builds a normalized query profile.
5. It retrieves a shortlist from bundle-backed retrieval artifacts when available, or falls back to request-scoped shortlist retrieval.
6. It prepares candidate features, applies ranking logic, and attaches explainability metadata.
7. The service returns the recommendation payload and updates in-memory metrics.

## Offline Training Path

The training path is:

1. Export or load raw source data.
2. Build a canonical bundle.
3. Build a training datamart.
4. Optionally build fabric and comment-intelligence snapshots.
5. Train recommender artifacts.
6. Evaluate the resulting bundle.
7. Promote or serve a compatible bundle from `artifacts/recommender_real/latest` (or override `RECOMMENDER_BUNDLE_DIR` explicitly).

## Product And Gateway Layer

The frontend stack lives in `frontend/`.

It is responsible for:

- user-facing upload/report workflows
- Node-side API composition
- proxying recommendation calls to the Python service
- deterministic fallback behavior when the recommender is unavailable
- feedback capture for downstream analysis and reranker training

If you are tracing a product issue rather than a model/runtime issue, start there after reading the Python service boundary.

## Core Vs Supporting Code

Core code for understanding the platform:

- `src/recommendation/contracts.py`
- `src/recommendation/datamart.py`
- `src/recommendation/fabric/core.py`
- `src/recommendation/learning/inference.py`
- `src/recommendation/service.py`
- `src/recommendation/control_plane.py`

Supporting or adjacent code:

- `frontend/`
- `scraper/`
- `eda/`
- `src/baseline/`
- `src/common/`

Those supporting paths still matter, but they are not the shortest route to understanding the current recommender architecture.

**Historical note:** an early TF-IDF-only retrieval scaffold lived under `src/retrieval/` (with CLI scripts `build_index.py` / `query_index.py`). It was removed; production retrieval is `src/recommendation/learning/retriever.py` and related training scripts.

## Recommended Read Order

For a new contributor:

1. `README.md`
2. `src/recommendation/README.md`
3. `src/recommendation/contracts.py`
4. `src/recommendation/datamart.py`
5. `src/recommendation/fabric/core.py`
6. `src/recommendation/learning/inference.py`
7. `src/recommendation/service.py`
8. `src/recommendation/control_plane.py`
9. `frontend/README.md`

## Useful Commands

Build a datamart:

```bash
make datamart
```

Train recommender artifacts:

```bash
make train-recommender
```

Evaluate the latest bundle:

```bash
make eval-recommender
```

Serve the recommender:

```bash
make serve-recommender
```

Run live end-to-end validation:

```bash
make live-e2e-validate
```
