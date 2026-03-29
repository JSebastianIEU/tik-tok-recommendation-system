# Frontend

React + Vite + TypeScript frontend with an optional local Node API for DeepSeek.

## Run locally (full experience)

1. Install dependencies:
   ```bash
   npm install
   ```
2. Configure local server credentials in `frontend/.env.local`:
   ```bash
   DEEPSEEK_API_KEY=your_key_here
   DEEPSEEK_MODEL=deepseek-chat
   DEEPSEEK_BASE_URL=https://api.deepseek.com
   ```
3. Start frontend + local API:
   ```bash
   npm run dev:all
   ```

## Run locally (frontend only, static mock mode)

Use this when you do not want to run the local API server:

```bash
npm run build
npm run preview
```

In this mode, report/chat use local mock fallbacks so the app still works.

## Build commands

- Standard build:
  ```bash
  npm run build
  ```

## Modeling tests

Run deterministic modeling tests (data contracts, Step 1, Step 2, Step 3, request parser):

```bash
npm run test:modeling
```

## Report API inputs

`POST /generate-report` supports:

- core fields: `description`, `hashtags[]`, `mentions[]`
- intent fields: `objective`, `audience`, `content_type`, `primary_cta`, `locale`
- optional signal hints: `signal_hints` (`duration_seconds`, `transcript_text`, `ocr_text`, `estimated_scene_cuts`, audio/tempo hints)

These inputs feed:

- Step 1 candidate profile (`core.v1`)
- Part 2 signal profile (`extractors.v1`)
- Step 2 comparable neighborhood (`step2.v1`)

## Recommender integration (Node -> Python)

The Node API includes:

- `POST /recommendations` (proxy to Python `POST /v1/recommendations`)
- `POST /generate-report` auto-uses recommender-ranked comparables when available
- deterministic fallback if recommender is unavailable/invalid

`POST /recommendations` supports additive optional retrieval controls:

- `language`, `locale`, `content_type`
- `candidate_ids[]` (optional scoped intersection with Python global index universe)
- `policy_overrides` (`strict_language`, `strict_locale`, `max_items_per_author`)
- `portfolio` (`enabled`, `weights.reach|conversion|durability`, `risk_aversion`, `candidate_pool_cap`)
- `graph_controls` (`enable_graph_branch`)
- `trajectory_controls` (`enabled`)
- `explainability` (`enabled`, `top_features`, `neighbor_k`, `run_counterfactuals`)
- `routing` (`track`, `allow_fallback`, `required_compat`)
- `experiment` (`id`, `force_variant`) for deterministic query-level assignment (`control`/`treatment`)

Recommender responses include additive metadata when available:

- top-level: `request_id`, `experiment_id`, `variant`
- top-level: `retrieval_mode`, `constraint_tier_used`, `retriever_artifact_version`
- per-item: `retrieval_branch_scores` (`lexical`, `dense_text`, `multimodal`, `graph_dense`, `trajectory_dense`, `fused`)
- top-level: `graph_bundle_id`, `graph_version`, `graph_coverage`, `graph_fallback_mode`
- top-level: `trajectory_manifest_id`, `trajectory_version`, `trajectory_mode`, `trajectory_prediction`
- top-level: `portfolio_mode`, `portfolio_metadata`
- per-item: `graph_trace`, `trajectory_trace`
- per-item: `portfolio_trace`
- per-item: `comment_trace` now includes comment-to-content alignment fields (`alignment_score`, `value_prop_coverage`, `on_topic_ratio`, `artifact_drift_ratio`, `alignment_shift_early_late`, `alignment_confidence`)
- top-level: `explainability_metadata`
- per-item: `evidence_cards`, `temporal_confidence_band`, `counterfactual_scenarios`
- top-level: `routing_decision`, `compatibility_status`, `fallback_reason`, `latency_breakdown_ms`, `circuit_state`

`POST /generate-report` now appends an additive `report.explainability` block with auditable cards
and counterfactual summaries, including comment-alignment evidence and strategy hints when available.
When recommender explainability is unavailable, the block is still
returned with fallback metadata.

Gateway observability endpoint:

- `GET /recommender-gateway-metrics` (stage latency summaries, breaker transitions, fallback reasons, compatibility/fallback-bundle counters)

Environment variables for recommender proxy:

```bash
RECOMMENDER_ENABLED=true
RECOMMENDER_BASE_URL=http://127.0.0.1:8081
RECOMMENDER_TIMEOUT_MS=3500
RECOMMENDER_RETRY_COUNT=1
RECOMMENDER_COMPAT_CHECK_INTERVAL_MS=30000
RECOMMENDER_FALLBACK_BUNDLE_DIR=artifacts/recommender_fallback
RECOMMENDER_FEEDBACK_ENABLED=true
RECOMMENDER_FEEDBACK_DB_URL=postgresql://user:pass@localhost:5432/tiktok
RECOMMENDER_EXPERIMENT_DEFAULT_ID=rec_v2_default
RECOMMENDER_EXPERIMENT_TREATMENT_RATIO=0.5
RECOMMENDER_EXPERIMENT_SALT=rec_v2_salt
```

Optional hardening knobs (all additive): `RECOMMENDER_BUDGET_*`, `RECOMMENDER_BREAKER_*`,
`RECOMMENDER_COMPAT_CACHE_TTL_MS`, `RECOMMENDER_FALLBACK_CACHE_TTL_MS`,
`RECOMMENDER_EXPERIMENT_CONTROL_BUNDLE_ID`, `RECOMMENDER_EXPERIMENT_TREATMENT_BUNDLE_ID`.

Objective alias mapping at boundary:

- incoming `community` objective maps to effective `engagement` model
- response keeps `objective` and includes `objective_effective`

## GitHub Pages deployment

Because GitHub Pages is static hosting, the Node API (`server/index.ts`) is not available there.

Use static mode:

1. Build:
   ```bash
   npm run build
   ```
2. Publish the `frontend/dist` contents to GitHub Pages (docs folder or `gh-pages` branch).

Notes:
- In GitHub Pages mode, report and chat automatically fallback to local mock logic.
- No API key is exposed in the browser.

## Vercel / Azure Static Web Apps

- Root Directory: `frontend`
- Build Command: `npm run build`
- Output Directory: `dist`

If no backend API is deployed, frontend automatically falls back to mock report/chat generation.

## Security

- DeepSeek key is used only by local server code (`server/index.ts`).
- Do not commit `frontend/.env.local`.

## Current architecture

- Frontend services call local API when available.
- If API is unavailable (or static mode is enabled), frontend falls back to typed mock services and local dataset generation.
- Dataset source: `src/data/demodata.jsonl`.
