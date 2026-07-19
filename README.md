# Cortex Studio

A production-grade **multi-agent AI system for marketing campaigns**. You hand it a
brief; a crew of specialist agents research the market, set strategy, write copy and
creative briefs, self-critique against brand guardrails, pause for human approval, and
assemble a final campaign package.

Built as a reference implementation of the ten pillars that separate a demo from a
production applied-AI system: guardrails, agentic memory (short/long/episodic),
security, observability with agent tracing, evaluation, testing, cost optimization,
human-in-the-loop, scaling, and reliable tool integrations.

## Status

| Milestone | Scope | State |
|-----------|-------|-------|
| **M0** | Repo + Docker stack (api, postgres/pgvector, redis) | done |
| **M1** | 8-node LangGraph agent graph, streaming API, mock-LLM offline mode | done |
| **M2** | Vector memory + RAG + web-search/image/publish tools + episodic recall | done |
| **M3** | Input+output guardrails, publish-gating, auth, rate limiting, secrets doc | done |
| **M4** | Per-agent tracing (Langfuse + local), model-tier router, prompt cache | done |
| **M5** | Golden-set evals (heuristic + judge + faithfulness), CI, promptfoo | done |
| **M6** | HITL interrupt + approval UI, async queue/scaling, dashboard, demo doc | done |

All six milestones landed and verified on the offline mock path (25 passing tests + eval
gate + a live browser run). Production swaps (real Claude, pgvector, Langfuse, Redis queue,
Tavily) are documented drop-ins that don't change the graph or the interfaces.

## Architecture

```
brief -> ingest(guardrails) -> planner -> research -> strategy -> copywriter
      -> creative_director -> critic --approved--> packager -> package
                                    \--revise--/  (bounded by MAX_CRITIC_LOOPS)
```

Shared typed `CampaignState` flows through a LangGraph state machine. Every LLM call
goes through one client layer that (a) routes to the cheapest capable model tier,
(b) meters tokens + cost per agent, and (c) can swap in a deterministic **mock** so
the entire graph runs offline with no API key.

- `apps/api/app/graph/` — state, agents (the 8 nodes), graph wiring
- `apps/api/app/llm/` — model router + cost meter + mock
- `apps/api/app/core/` — config + guardrails (injection/PII)
- `apps/api/app/api/` — FastAPI routes (`/api/runs`, `/api/runs/stream`)
- `apps/web/` — Next.js frontend (scaffolding in M1, UI in M6)

## Run it

Offline, no key needed (mock LLM):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r apps/api/requirements.txt
python scripts/smoke.py                    # end-to-end run, printed package + cost
cd apps/api && python -m pytest -q         # tests
uvicorn app.main:app --reload              # API on :8000  (GET /health)
```

Real LLM (Gemini has a free tier — key from https://aistudio.google.com, no card):

```bash
export LLM_PROVIDER=gemini GEMINI_API_KEY=...        # free
# or: export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-...
python scripts/smoke.py
```

Model tiers auto-select per provider (Gemini: flash-lite / flash / pro), so the cheap
routing steps and the heavy critic use the right model with no extra config.

Full stack (once Docker Desktop is running):

```bash
cp .env.example .env
docker compose up --build                  # api + postgres/pgvector + redis
```

## Pillar → where it lives (interview cheat-sheet)

- **Guardrails** — `core/guardrails.py` (injection neutralize + PII redaction) + Pydantic
  output schemas on every agent + the brand-safety critic loop.
- **Memory** — short-term: LangGraph checkpointer (resumable threads); long-term + episodic:
  pgvector (M2).
- **Security** — API-key auth (`require_api_key`), PII kept out of traces, env-only secrets.
- **Observability + tracing** — cost meter live now; Langfuse spans in M4.
- **Evaluation** — pytest trajectory tests now; Ragas + promptfoo + LLM-judge golden set in M5.
- **Cost optimization** — model-tier router + per-agent cost meter (`llm/client.py`).
- **HITL** — critic loop now; `interrupt()` approval gate + UI in M6.
- **Scaling** — async FastAPI + stateless graph; Redis queue + workers in M6.
- **Tool integrations** — web search + RAG retriever + image-brief + mock publish in M2.
