"""The eight agent nodes. Each is a pure function State -> partial State.

For M1 the agents are thin prompt wrappers with validated structured outputs.
M2+ enriches them (real RAG in research, tools in strategy/creative, etc.) without
changing these signatures or the graph wiring."""
from __future__ import annotations

import json
from typing import Any

from app.core.config import settings
from app.core.guardrails import sanitize_input
from app.core.moderation import moderate_copy
from app.memory import episodic
from app.tools.registry import call_tool
from app.graph.state import (
    CampaignState,
    CopyOutput,
    CreativeOutput,
    CritiqueOutput,
    ResearchOutput,
    StrategyOutput,
)
from app.llm.client import CostMeter, Tier, complete
from app.obs.tracing import Tracer, get_tracer

# One meter + tracer per process run. Reset per run in the API layer / tests.
METER = CostMeter()
TRACER: Tracer = get_tracer("default")


def reset_run(run_id: str) -> None:
    """Reset the per-run cost meter and tracer. Call before invoking the graph."""
    global TRACER
    METER.__init__()
    TRACER = get_tracer(run_id)


def _event(agent: str, msg: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"agent": agent, "message": msg, "data": data or {}}


def _run(agent: str, system: str, prompt: str, tier: Tier) -> str:
    with TRACER.span(agent, kind="agent", input=prompt[:200]) as s:
        c = complete(system, prompt, tier=tier)
        METER.record(agent, c.usage)
        s.set_output(c.text[:200])
        s.set_usage(c.usage.input_tokens, c.usage.output_tokens, c.usage.cost_usd)
    return c.text


def _parse(model_cls, text: str):
    """Best-effort parse of model text into a Pydantic schema. On the mock path the
    text is valid JSON; on real runs we tolerate prose by falling back to defaults."""
    try:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            return model_cls.model_validate(json.loads(text[start : end + 1]))
    except Exception:
        pass
    return model_cls()


# --- nodes ---

def ingest(state: CampaignState) -> CampaignState:
    """Guardrail entry point: sanitize the brief before any agent sees it."""
    safe, flags = sanitize_input(state.get("brief", ""))
    return {
        "sanitized_brief": safe,
        "guardrail_flags": flags,
        "critic_loops": 0,
        "events": [_event("ingest", "brief sanitized", {"flags": flags})],
    }


_DEFAULT_PLAN = ["research", "strategy", "copy", "creative"]


def planner(state: CampaignState) -> CampaignState:
    txt = _run(
        "planner",
        "You are an orchestrator. Break a marketing brief into an ordered task plan. Reply JSON {\"items\": [...]}.",
        state["sanitized_brief"],
        Tier.CHEAP,
    )
    plan = _DEFAULT_PLAN
    try:
        start, end = txt.find("{"), txt.rfind("}")
        if start != -1 and end != -1:
            items = json.loads(txt[start : end + 1]).get("items")
            if isinstance(items, list) and items:
                plan = [str(i) for i in items]
    except Exception:
        pass
    # Episodic memory: recall similar past campaigns to prime this run.
    recalled = episodic.recall_similar(state["sanitized_brief"])
    return {
        "plan": plan,
        "recalled_episodes": recalled,
        "events": [_event("planner", "plan built", {"steps": plan, "recalled": len(recalled)})],
    }


def research(state: CampaignState) -> CampaignState:
    """Grounded research: pull brand-doc chunks (RAG) + web signals, then have the
    model synthesize findings that cite those real sources."""
    brief = state["sanitized_brief"]
    rag = call_tool("rag_retrieve", query=brief, k=3)
    web = call_tool("web_search", query=brief, k=2)
    sources = [c["source"] for c in rag["chunks"]] + [r["url"] for r in web["results"]]
    context = "\n".join(
        [f"[brand] {c['text']}" for c in rag["chunks"]]
        + [f"[web] {r['snippet']} ({r['url']})" for r in web["results"]]
    )
    txt = _run(
        "research",
        "You are a market researcher. Ground every claim in the provided sources. "
        "Output JSON {\"findings\":[{\"claim\":..,\"source\":..}],\"summary\":..}.",
        f"Brief: {brief}\n\nSources:\n{context}",
        Tier.MID,
    )
    out: ResearchOutput = _parse(ResearchOutput, txt)
    dump = out.model_dump()
    dump["retrieved_sources"] = sources  # provenance for eval + traces
    return {
        "research": dump,
        "events": [_event("research", "grounded research done", {"sources": len(sources)})],
    }


def strategy(state: CampaignState) -> CampaignState:
    txt = _run(
        "strategy",
        "You are a brand strategist. Output JSON {\"segments\":[..],\"positioning\":..,\"channels\":[..]}.",
        f"Brief: {state['sanitized_brief']}\nResearch: {state.get('research', {})}",
        Tier.MID,
    )
    out: StrategyOutput = _parse(StrategyOutput, txt)
    return {"strategy": out.model_dump(), "events": [_event("strategy", "strategy set")]}


def copywriter(state: CampaignState) -> CampaignState:
    txt = _run(
        "copywriter",
        "You are a copywriter. Output JSON {\"headline\":..,\"captions\":[..],\"email_subject\":..,\"email_body\":..}.",
        f"Strategy: {state.get('strategy', {})}",
        Tier.MID,
    )
    out: CopyOutput = _parse(CopyOutput, txt)
    return {"copy": out.model_dump(), "events": [_event("copywriter", "copy drafted")]}


def creative_director(state: CampaignState) -> CampaignState:
    txt = _run(
        "creative_director",
        "You are a creative director. Output JSON {\"concept\":..,\"image_prompts\":[..]}.",
        f"Strategy: {state.get('strategy', {})}\nCopy: {state.get('copy', {})}",
        Tier.MID,
    )
    out: CreativeOutput = _parse(CreativeOutput, txt)
    dump = out.model_dump()
    # Tool-augment: expand the concept into concrete image prompts.
    if dump.get("concept"):
        dump["image_prompts"] = call_tool("image_brief", concept=dump["concept"], n=2)["prompts"]
    return {"creative": dump, "events": [_event("creative_director", "creative briefs ready")]}


def critic(state: CampaignState) -> CampaignState:
    """Brand-safety + quality gate. Uses the heavy tier because catching off-brand
    output is the highest-leverage judgment in the graph."""
    loops = state.get("critic_loops", 0) + 1
    txt = _run(
        "critic",
        "You are a strict brand-safety critic. Output JSON {\"approved\":bool,\"score\":0..1,\"issues\":[..]}.",
        f"Copy: {state.get('copy', {})}\nCreative: {state.get('creative', {})}",
        Tier.HEAVY,
    )
    out: CritiqueOutput = _parse(CritiqueOutput, txt)
    # Deterministic output-moderation gate: hard-block banned content regardless of
    # what the LLM critic says. A real violation forces a revision loop.
    violations = moderate_copy(state.get("copy", {}) or {})
    issues = list(out.issues) + [f"{v['category']}:{v['pattern']}" for v in violations]
    approved = (
        (out.approved or out.score >= 0.8)
        and not violations
    ) or (loops >= settings.max_critic_loops and not violations)
    return {
        "critique": {"approved": approved, "score": out.score, "issues": issues, "violations": violations},
        "critic_loops": loops,
        "events": [_event("critic", f"critique loop {loops}", {"approved": approved, "violations": len(violations)})],
    }


def human_gate(state: CampaignState) -> CampaignState:
    """Human-in-the-loop approval. Pauses the graph via interrupt() and surfaces the
    draft; the run resumes when a human posts a decision (approve / edit / reject).

    Resume payload: {"action": "approve"|"edit"|"reject", "edits": {<copy overrides>}}

    HITL is opt-in per run (`hitl_enabled` in the initial state). Batch/eval runs leave
    it off and auto-approve so they complete without a human."""
    if not state.get("hitl_enabled"):
        return {"hitl_decision": "approve", "events": [_event("human_gate", "auto-approved (hitl off)")]}

    from langgraph.types import interrupt

    decision = interrupt(
        {
            "draft": {
                "copy": state.get("copy"),
                "creative": state.get("creative"),
                "critique": state.get("critique"),
            },
            "prompt": "Approve, edit, or reject this campaign draft.",
        }
    )
    action = (decision or {}).get("action", "approve")
    edits = (decision or {}).get("edits") or {}
    update: CampaignState = {
        "hitl_decision": action,
        "events": [_event("human_gate", f"human decision: {action}")],
    }
    if action == "edit" and edits:
        update["copy"] = {**(state.get("copy") or {}), **edits}
    return update


def route_after_gate(state: CampaignState) -> str:
    decision = state.get("hitl_decision", "approve")
    return "revise" if decision == "reject" else "package"


def packager(state: CampaignState) -> CampaignState:
    package = {
        "brief": state.get("sanitized_brief"),
        "research": state.get("research"),
        "strategy": state.get("strategy"),
        "copy": state.get("copy"),
        "creative": state.get("creative"),
        "critique": state.get("critique"),
    }
    approved = (state.get("critique") or {}).get("approved", False)
    package["status"] = "approved" if approved else "blocked"
    # Episodic memory: persist this run so future planners can recall it.
    episodic.record_episode(state.get("run_id", "unknown"), state.get("sanitized_brief", ""), package)
    # Security/guardrail: only publish content that cleared the critic + moderation.
    published = call_tool("publish", package=package, channel="mock") if approved else {"published": False, "reason": "blocked by guardrails"}
    TRACER.flush()  # persist spans (+ mirror to Langfuse if configured)
    return {
        "package": package,
        "cost": METER.snapshot(),
        "trace": TRACER.summary(),
        "events": [_event("packager", f"package {package['status']}", {"published": published})],
    }


def route_after_critic(state: CampaignState) -> str:
    """Conditional edge: loop back to copywriter if rejected and under the cap."""
    crit = state.get("critique", {})
    if crit.get("approved"):
        return "package"
    if state.get("critic_loops", 0) >= settings.max_critic_loops:
        return "package"
    return "revise"
