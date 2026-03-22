# Tik-Tok-Recommendation-System

An explainable, multimodal recommendation system for TikTok that predicts content performance before publication. Combines data scraping, NLP, computer vision, audio analysis, and retrieval-based recommendations to empower creators with actionable, transparent insights.

# Predictive Social Core (Skeleton)

This repo is a lightweight scaffold for TikTok-style recommendation experiments. No scraping is included; data is assumed to come from an external source.

## Layout

- `data/mock/`: small mocked JSONL dataset for experiments.
- `eda/`: reproducible EDA workspace (plans, extracts, notebooks, reports).
- `src/common/`: shared schemas, validation utilities, constants.
- `src/data/`: stubs for data generation/ingestion helpers (no scrapers).
- `src/retrieval/`: retrieval skeleton (index + search abstractions).
- `src/recommendation/`: Python contract layer, datamart builder, learned recommender training/inference, and FastAPI service.
- `frontend/`: React app + Node API that proxies learned recommendations and falls back to deterministic logic.
- `src/baseline/`: simple baseline stats and reporting.
- `src/research/`: notes and TODOs for comparing retrieval approaches.
- `scripts/`: CLI entrypoints for validation, datamart, training, eval, and serving.
- `tests/`: smoke and recommendation tests.
- `.github/workflows/ci.yml`: CI skeleton for lint + tests on PRs.
- `Makefile`: convenience targets.

## Getting Started

1) Create a virtualenv and install deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2) Validate mocked data:
   ```bash
   make validate-data
   ```
3) Run baseline stats and generate a starter report:
   ```bash
   make baseline
   ```
4) Try a placeholder retrieval query:
   ```bash
   make query
   ```
5) Build a training data mart snapshot:
   ```bash
   make datamart
   ```
6) Bootstrap an EDA extract run from the scraper DB:
   ```bash
   python3 scripts/eda_bootstrap.py --plan eda/configs/plan.example.yaml
   ```
7) Train recommender artifacts:
   ```bash
   make train-recommender
   ```
8) Inspect offline metrics from latest bundle:
   ```bash
   make eval-recommender
   ```
9) Start the Python recommender service (for Node proxy):
   ```bash
   make serve-recommender
   ```
10) Run local latency/memory smoke benchmark:
   ```bash
   make benchmark-recommender
   ```

## Notes

- All logic is minimal by design. Replace TODOs with real implementations.
- Retrieval comparisons (TF-IDF vs SBERT vs BM25 vs FAISS) live in `src/research/`.
- Objective alias policy at API boundary is `community -> engagement`.
