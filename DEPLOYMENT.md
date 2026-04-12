# Deployment Guide

This repository has more than one deployable surface:

- the Python recommender service in `src/recommendation/service.py`
- the frontend and Node gateway in `frontend/`
- supporting control-plane jobs in `scripts/`

This document is a short deployment map, not a full production runbook.
For runtime checks and incident handling, see [docs/recommender_runbook.md](/Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/docs/recommender_runbook.md).

## Deployable Components

### Python recommender service

Purpose:
- loads a recommender bundle from `artifacts/recommender_real/latest` or `RECOMMENDER_BUNDLE_DIR`
- serves FastAPI endpoints such as `/v1/health`, `/v1/compatibility`, and `/v1/recommendations`

Local launch:

```bash
make serve-recommender
```

Equivalent script:

```bash
PYTHONPATH=. python3 scripts/serve_recommender.py \
  --host 127.0.0.1 \
  --port 8081 \
  --bundle-dir artifacts/recommender_real/latest
```

Key requirements:
- Python environment with `fastapi` and `uvicorn`
- compatible bundle under `artifacts/recommender_real/` or an explicit `RECOMMENDER_BUNDLE_DIR`

### Frontend and Node gateway

Purpose:
- serves the React app
- hosts the Node API in `frontend/server/index.ts`
- proxies recommendation calls to the Python recommender service
- formats reports, applies fallback behavior, and records feedback

Local launch:

```bash
cd frontend
npm install
npm run dev:all
```

The gateway expects the Python recommender service to be reachable when recommender integration is enabled.

## Frontend Hosting

### Vercel or Azure Static Web Apps

Use the `frontend/` directory as the app root.

Recommended settings:
- Root Directory: `frontend`
- Build Command: `npm run build`
- Output Directory: `dist`
- Install Command: `npm install`

Important note:
- Static hosting only covers the React build output.
- The Node gateway in `frontend/server/index.ts` is not provided by static hosting alone.
- If you deploy only the static frontend, it must rely on its fallback/mock behavior or an externally hosted API stack.

### GitHub Pages

GitHub Pages can host only the static frontend build.

Build:

```bash
cd frontend
npm install
npm run build
```

Publish:
- deploy `frontend/dist`

Constraints:
- no Node gateway
- no local thumbnail/report/chat endpoints from `frontend/server/index.ts`
- browser app must rely on configured fallbacks or external APIs

## Environment Notes

Frontend/gateway environment commonly includes:

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_MODEL`
- `DEEPSEEK_BASE_URL`
- `RECOMMENDER_CORPUS_PROVIDER`
- `RECOMMENDER_CORPUS_BUNDLE_PATH`
- `RECOMMENDER_ENABLED`
- `RECOMMENDER_BASE_URL`
- `RECOMMENDER_TIMEOUT_MS`
- `RECOMMENDER_FALLBACK_BUNDLE_DIR`

Python recommender service environment commonly includes:

- `RECOMMENDER_BUNDLE_DIR`
- `FABRIC_CALIBRATION_PATH`
- `HASHTAG_RECOMMENDER_DIR`

For the fuller frontend integration contract, see [frontend/README.md](/Users/ayoisthegoat/Desktop/Education/Chatbots/Tik-Tok/Tik-Tok-Recommendation-System/frontend/README.md).

## Recommended Deployment Order

1. Build or select the recommender bundle.
2. Start and verify the Python recommender service.
3. Start the frontend and Node gateway against that Python service.
4. Run health and compatibility checks.
5. Run live end-to-end validation before promotion.

## Freshness Refresh

For serving freshness after scraping/comment jobs, refresh the contract bundle and
retriever-backed serving bundle without full model retraining:

```bash
PYTHONPATH=. python3 scripts/refresh_serving_bundle.py --db-url "$DATABASE_URL"
```

What this updates:
- `artifacts/contracts/latest_supabase_bundle.json` for the Node gateway corpus
- `artifacts/datamart/supabase_latest_datamart.json` for the refreshed serving snapshot
- a new bundle under `artifacts/recommender_real/`
- the `artifacts/recommender_real/latest` symlink when promotion is enabled

This is the fast-refresh path. Full ranker retraining remains a separate workflow.

## Verification Commands

Python service:

```bash
curl -s http://127.0.0.1:8081/v1/health
curl -s http://127.0.0.1:8081/v1/compatibility
curl -s http://127.0.0.1:8081/v1/metrics
```

Node gateway:

```bash
curl -s http://127.0.0.1:5174/recommender-gateway-metrics
```

Release validation:

```bash
make live-e2e-validate
```
