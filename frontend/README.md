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

Environment variables for recommender proxy:

```bash
RECOMMENDER_ENABLED=true
RECOMMENDER_BASE_URL=http://127.0.0.1:8081
RECOMMENDER_TIMEOUT_MS=3500
RECOMMENDER_RETRY_COUNT=1
```

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
