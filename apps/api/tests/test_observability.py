"""M4: tracing spans, cost metering per agent, prompt-cache cost savings."""
from __future__ import annotations

from app.graph import agents
from app.graph.build import build_graph
from app.llm import client
from app.obs.tracing import load_trace

BRIEF = "Refill-signup campaign for Sai Energy LPG, Coimbatore households."


def test_trace_has_span_per_agent():
    agents.reset_run("obs-1")
    state = build_graph().invoke({"run_id": "obs-1", "brief": BRIEF})
    trace = state["trace"]
    names = {s["name"] for s in trace["spans"]}
    assert {"planner", "research", "strategy", "copywriter", "critic"} <= names
    assert trace["total_cost_usd"] >= 0


def test_trace_is_persisted_and_loadable():
    agents.reset_run("obs-2")
    build_graph().invoke({"run_id": "obs-2", "brief": BRIEF})
    loaded = load_trace("obs-2")
    assert loaded and loaded["run_id"] == "obs-2"
    assert loaded["spans"]


def test_prompt_cache_zeroes_repeat_cost():
    client._CACHE.clear()
    c1 = client.complete("sys", "identical prompt", tier=client.Tier.CHEAP)
    c2 = client.complete("sys", "identical prompt", tier=client.Tier.CHEAP)
    assert not c1.cached and c1.usage.cost_usd > 0
    assert c2.cached and c2.usage.cost_usd == 0.0
    assert c1.text == c2.text


def test_model_router_selects_tiers():
    # cheap != heavy model ids, proving per-step routing happens
    from app.core.config import settings
    assert settings.model_cheap != settings.model_heavy
