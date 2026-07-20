"""Concurrency / load stress: many runs in parallel complete correctly, and the
prompt cache demonstrably cuts cost on repeats. Mock path so there are no rate limits."""
from __future__ import annotations

import concurrent.futures as cf

from app.graph import agents
from app.graph.build import build_graph
from app.llm import client

BRIEF = "Refill-signup campaign for Sai Energy LPG, Coimbatore households."


def _one(i: int) -> str:
    # distinct thread_id per run; graph + checkpointer are cheap to build per call
    agents.reset_run(f"stress-{i}")
    state = build_graph().invoke({"run_id": f"stress-{i}", "brief": f"{BRIEF} variant {i}"})
    return state["package"]["status"]


def test_many_concurrent_runs_all_complete():
    n = 24
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_one, range(n)))
    assert len(results) == n
    assert all(r in {"approved", "blocked"} for r in results)


def test_cache_cuts_cost_on_repeat_run():
    client._CACHE.clear()
    agents.reset_run("warm")
    build_graph().invoke({"run_id": "warm", "brief": BRIEF})
    first_cost = agents.METER.total.cost_usd
    assert first_cost > 0

    agents.reset_run("repeat")  # cache is NOT cleared -> identical prompts hit it
    build_graph().invoke({"run_id": "repeat", "brief": BRIEF})
    repeat_cost = agents.METER.total.cost_usd
    # repeat run reuses cached completions -> materially cheaper
    assert repeat_cost < first_cost
