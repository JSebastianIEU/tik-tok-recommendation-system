# Tik-Tok-Recommendation-System

An explainable, multimodal recommendation system for TikTok that predicts content performance before publication. Combines data scraping, NLP, computer vision, audio analysis, and retrieval-based recommendations to empower creators with actionable, transparent insights.

# Predictive Social Core (Skeleton)

This repo is a lightweight scaffold for TikTok-style recommendation experiments. No scraping is included; data is assumed to come from an external source.

## Layout

- `data/mock/` � small mocked JSONL dataset for experiments.
- `src/common/` � shared schemas, validation utilities, constants.
- `src/data/` � stubs for data generation/ingestion helpers (no scrapers).
- `src/retrieval/` � retrieval skeleton (index + search abstractions).
- `src/baseline/` � simple baseline stats and reporting.
- `src/research/` � notes and TODOs for comparing retrieval approaches.
- `scripts/` � CLI entrypoints for validation, baselines, and retrieval.
- `tests/` � smoke tests to keep the scaffold wired up.
- `.github/workflows/ci.yml` � CI skeleton for lint + tests on PRs.
- `Makefile` � convenience targets.

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

## Notes

- All logic is minimal by design. Replace TODOs with real implementations.
- Retrieval comparisons (TF-IDF vs SBERT vs BM25 vs FAISS) live in `src/research/`.
