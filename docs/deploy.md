# Deploying Cortex Studio (Render + saikanagat.com/cortex)

The **backend** (multi-agent API + image generation) runs on Render. The **microsite**
(`/cortex`) is a static page in the portfolio repo that calls the backend.

## 1. Backend on Render (Blueprint)

1. Push this repo to GitHub (already done: `Sai-Kanagat/cortex-studio`).
2. Render dashboard -> **New +** -> **Blueprint**.
3. Connect the `cortex-studio` repo. Render reads `render.yaml` and proposes the
   `cortex-api` web service (Docker, free plan, 1 GB persistent disk).
4. Before the first deploy, set the two secret env vars (they are `sync: false`):
   - `GEMINI_API_KEY` = your Google AI Studio key.
   - `CORTEX_PASSCODE` = `0000` (or your choice).
5. Click **Apply / Deploy**. First build takes a few minutes (installs ffmpeg + deps).
6. When live, note the URL, e.g. `https://cortex-api.onrender.com`. Check
   `https://cortex-api.onrender.com/health` returns `{"status":"ok",...}`.

Notes:
- Free plan **sleeps after ~15 min idle**; the first request then cold-starts (~30-60s).
  The microsite shows a "warming up" state, so this is fine for a preview.
- Gemini **free tier is tiny** (≈20 req/day/model; images even less). The per-IP daily
  cap (`CORTEX_DAILY_RUN_CAP`) protects it; recruiters hitting the cap see the instant demo.

## 2. Wire the microsite to the backend

In `portfolio-v2/cortex/index.html`, set:

```js
const BACKEND_URL = "https://cortex-api.onrender.com";   // your Render URL
```

Commit + push portfolio-v2; the live "try it" now calls the real backend.

## 3. Routing saikanagat.com/cortex

The microsite lives at `portfolio-v2/cortex/index.html`, so the host serves it at
`/cortex` automatically (static hosting). No extra config for Vercel/Netlify. If a
rewrite is needed, add `/cortex -> /cortex/index.html`.

## Local full stack

```bash
cp .env.example .env    # add GEMINI_API_KEY, LLM_PROVIDER=gemini
docker compose up --build
# API at http://localhost:8000  (also serves apps/web)
```
