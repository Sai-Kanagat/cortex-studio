# Scaling (M6)

## What's implemented (runnable now)
- **Non-blocking API:** async FastAPI. The blocking graph run is offloaded to a worker
  thread (`run_in_executor`) so the event loop stays responsive.
- **Queue-based execution:** `POST /api/runs/async` enqueues a job and returns a `run_id`
  immediately; `GET /api/runs/{id}` polls status/result. Many runs proceed concurrently.
- **Stateless graph:** all run state lives in the checkpointer, not in the process, so any
  worker can pick up any thread. This is what makes horizontal scale possible.

## Production path
- **Job queue:** replace the in-memory `_JOBS` dict + thread executor with Redis + a worker
  pool (Celery / arq / RQ). Jobs survive restarts; scale workers independently of the API.
- **Checkpointer:** swap `MemorySaver` for the Postgres checkpointer (`langgraph-checkpoint-postgres`).
  Same interface, so `build_graph(checkpointer=...)` is the only change. Threads then persist
  across restarts and are shared by all workers.
- **Rate limiting:** move the token bucket (`core/ratelimit.py`) to Redis so limits hold
  across instances.
- **Autoscaling:** API and workers are stateless containers; scale on CPU/queue-depth.

## Load characteristics
- The critic uses the heavy model tier and dominates cost/latency; the router keeps cheap
  steps (planner) on Haiku. Prompt caching zeroes repeat-run cost. Together these bound the
  per-run spend the cost meter reports.
