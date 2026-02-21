# MVP Mock UI - Menu #1

React + Vite + TypeScript frontend with an optional local Node API for DeepSeek.

## Run locally (full experience)

1. Install dependencies:
   ```bash
   npm install
   ```
2. Configure local server credentials in `mvp-mock-ui/.env.local`:
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
npm run build:gh
npm run preview
```

In this mode, report/chat use local mock fallbacks so the app still works.

## Build commands

- Standard build:
  ```bash
  npm run build
  ```
- GitHub Pages static build (mock-only):
  ```bash
  npm run build:gh
  ```

## GitHub Pages deployment

Because GitHub Pages is static hosting, the Node API (`server/index.ts`) is not available there.

Use static mode:

1. Build:
   ```bash
   npm run build:gh
   ```
2. Publish the `mvp-mock-ui/dist` contents to GitHub Pages (docs folder or `gh-pages` branch).

Notes:
- In GitHub Pages mode, report and chat automatically fallback to local mock logic.
- No API key is exposed in the browser.

## Vercel / Azure Static Web Apps

- Root Directory: `mvp-mock-ui`
- Build Command: `npm run build`
- Output Directory: `dist`

If no backend API is deployed, frontend automatically falls back to mock report/chat generation.

## Security

- DeepSeek key is used only by local server code (`server/index.ts`).
- Do not commit `mvp-mock-ui/.env.local`.

## Current architecture

- Frontend services call local API when available.
- If API is unavailable (or static mode is enabled), frontend falls back to typed mock services and local dataset generation.
- Dataset source: `src/data/demodata.jsonl`.
