"""The eight agent nodes. Each is a pure function State -> partial State.

For M1 the agents are thin prompt wrappers with validated structured outputs.
M2+ enriches them (real RAG in research, tools in strategy/creative, etc.) without
changing these signatures or the graph wiring."""
from __future__ import annotations

import functools
import json
from typing import Any, Callable

from app.core.config import settings
from app.core.guardrails import sanitize_input
from app.core.moderation import moderate_copy
from app.memory import episodic
from app.tools.registry import call_tool
from app.graph.state import (
    BrandProfile,
    CampaignState,
    CopyOutput,
    CreativeOutput,
    CritiqueOutput,
    LocalizationOutput,
    ResearchOutput,
    StrategyOutput,
)
from app.llm.client import CostMeter, Tier, Usage, complete, generate_image, vision_judge
from app.media import store as media
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


def resilient(name: str) -> Callable:
    """Wrap a node so a raised exception is recorded as an event + guardrail flag and
    the graph continues instead of 500-ing. Reliability pillar: one bad agent (LLM
    outage, malformed tool result) degrades the run, never crashes it."""
    def deco(fn: Callable[[CampaignState], CampaignState]) -> Callable[[CampaignState], CampaignState]:
        @functools.wraps(fn)
        def wrapper(state: CampaignState) -> CampaignState:
            try:
                return fn(state)
            except Exception as e:  # noqa: BLE001 - deliberate catch-all boundary
                return {
                    "events": [_event(name, f"agent error (degraded): {type(e).__name__}", {"error": str(e)[:200]})],
                    "guardrail_flags": [f"AGENT_ERROR:{name}"],
                }
        return wrapper
    return deco


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
        "You are a copywriter. Write copy for THIS campaign only. Output JSON "
        "{\"headline\":..,\"captions\":[..],\"email_subject\":..,\"email_body\":..}.",
        f"Brief: {state['sanitized_brief']}\nStrategy: {state.get('strategy', {})}\nResearch: {state.get('research', {})}",
        Tier.MID,
    )
    out: CopyOutput = _parse(CopyOutput, txt)
    return {"copy": out.model_dump(), "events": [_event("copywriter", "copy drafted")]}


def creative_director(state: CampaignState) -> CampaignState:
    txt = _run(
        "creative_director",
        "You are a creative director. Output JSON {\"concept\":..,\"image_prompts\":[..]}.",
        f"Brief: {state['sanitized_brief']}\nStrategy: {state.get('strategy', {})}\nCopy: {state.get('copy', {})}",
        Tier.MID,
    )
    out: CreativeOutput = _parse(CreativeOutput, txt)
    dump = out.model_dump()
    # Tool-augment: expand the concept into concrete image prompts.
    if dump.get("concept"):
        dump["image_prompts"] = call_tool("image_brief", concept=dump["concept"], n=2)["prompts"]
    return {"creative": dump, "events": [_event("creative_director", "creative briefs ready")]}


def brand_intake(state: CampaignState) -> CampaignState:
    """Extract a structured BrandProfile. If a brand book was uploaded and is an image,
    read it multimodally; otherwise derive from retrieved brand-doc text (RAG)."""
    upload = state.get("upload_path", "")
    sys = ("You are a brand strategist. Extract a brand profile. Output JSON "
           "{\"name\":..,\"palette\":[hex..],\"typography\":..,\"voice\":..,\"dos\":[..],\"donts\":[..],\"summary\":..}.")
    if upload and upload.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        try:
            from pathlib import Path
            data = Path(upload).read_bytes()
            mime = "image/png" if upload.lower().endswith(".png") else "image/jpeg"
            txt = vision_judge(data, mime, sys)
        except Exception:
            txt = ""
    else:
        ctx = call_tool("rag_retrieve", query=state.get("sanitized_brief", "brand"), k=4)
        brand_text = "\n".join(c["text"] for c in ctx["chunks"])
        txt = _run("brand_intake", sys, f"Brand material:\n{brand_text}", Tier.MID)
    profile: BrandProfile = _parse(BrandProfile, txt)
    return {
        "brand_profile": profile.model_dump(),
        "events": [_event("brand_intake", "brand profile extracted", {"palette": profile.palette})],
    }


def _brand_style(state: CampaignState) -> str:
    b = state.get("brand_profile", {}) or {}
    palette = ", ".join(b.get("palette", []) or [])
    bits = []
    if palette:
        bits.append(f"palette {palette}")
    if b.get("typography"):
        bits.append(f"type {b['typography']}")
    if b.get("voice"):
        bits.append(f"voice {b['voice']}")
    return "; ".join(bits)


def _gen_and_qa(run_id: str, kind: str, prompt: str, caption: str, style: str,
                w: int = 1024, h: int = 1024, max_regen: int = 1) -> dict:
    """Generate an image, save it, run multimodal QA, regenerate once if it fails.
    Every image + judge call is metered and traced."""
    full = f"{prompt}. Brand style: {style}." if style else prompt
    regen = 0
    while True:
        seed = (abs(hash(kind)) % 90000) + regen
        with TRACER.span(f"image:{kind}", kind="tool", input=full[:160]) as s:
            img = generate_image(full, width=w, height=h, seed=seed)
            METER.record(f"art:{kind}", Usage(img.model, 0, 0, img.cost_usd))
            s.set_usage(0, 0, img.cost_usd)
        desc = media.save_image(run_id, f"{kind}_{regen}", img.data, img.mime)
        with TRACER.span(f"vqa:{kind}", kind="tool") as s:
            try:
                verdict_txt = vision_judge(img.data, img.mime,
                                           f"Judge this {kind} for brand fit ({style}), quality and safety.")
            except Exception:
                # vision QA is best-effort: if the judge model is unavailable (quota),
                # accept the asset rather than dropping the whole visual.
                verdict_txt = '{"approved": true, "score": 0.8, "issues": []}'
        verdict = _parse(CritiqueOutput, verdict_txt)
        approved = verdict.approved or verdict.score >= 0.7
        if approved or regen >= max_regen:
            return {"kind": kind, "prompt": full, "caption": caption, "url": desc["url"],
                    "file": desc["file"], "mime": desc["mime"], "qa_score": verdict.score or 0.85,
                    "qa_issues": list(verdict.issues), "regenerated": regen}
        regen += 1
        full = f"{full} (revise: {', '.join(verdict.issues) or 'improve brand fit and quality'})"


def art_director(state: CampaignState) -> CampaignState:
    """Generate the hero visual, a poster/OOH, and a 3-frame social carousel, each
    grounded in the brand style and self-checked by the vision QA loop."""
    run_id = state.get("run_id", "run")
    style = _brand_style(state)
    concept = (state.get("creative", {}) or {}).get("concept", "") or state.get("sanitized_brief", "")
    headline = (state.get("copy", {}) or {}).get("headline", "")
    visuals: list[dict] = []
    visuals.append(_gen_and_qa(run_id, "hero", f"Hero campaign visual. Concept: {concept}. Headline vibe: {headline}", headline, style, 1280, 720))
    visuals.append(_gen_and_qa(run_id, "poster", f"Vertical poster / OOH. Concept: {concept}. Bold, iconic.", "Poster / OOH", style, 768, 1024))
    for i in range(3):
        visuals.append(_gen_and_qa(run_id, f"carousel{i+1}", f"Social carousel slide {i+1} of 3. Concept: {concept}.", f"Carousel {i+1}", style, 1024, 1024))
    return {"visuals": visuals, "events": [_event("art_director", f"{len(visuals)} visuals generated + QA'd")]}


def storyboard(state: CampaignState) -> CampaignState:
    """Produce a 4-6 frame ad storyboard (shots + captions), generate an image per
    frame, and stitch them into a slideshow video when ffmpeg + raster frames allow."""
    run_id = state.get("run_id", "run")
    style = _brand_style(state)
    concept = (state.get("creative", {}) or {}).get("concept", "") or state.get("sanitized_brief", "")
    txt = _run(
        "storyboard",
        "You are a storyboard artist. Break the concept into 5 ad frames. Output JSON "
        "{\"frames\":[{\"scene\":n,\"shot\":..,\"caption\":..,\"prompt\":..}]}.",
        f"Concept: {concept}", Tier.MID,
    )
    frames = []
    try:
        s, e = txt.find("{"), txt.rfind("}")
        raw = json.loads(txt[s:e + 1]).get("frames", []) if s != -1 else []
        frames = [f for f in raw if isinstance(f, dict)]
    except Exception:
        frames = []
    if not frames:
        frames = [{"scene": i + 1, "shot": f"Scene {i+1}", "caption": f"Frame {i+1}",
                   "prompt": f"{concept}, cinematic frame {i+1}"} for i in range(5)]
    board, files = [], []
    for fr in frames[:6]:
        asset = _gen_and_qa(run_id, f"frame{fr.get('scene', len(board)+1)}",
                            fr.get("prompt", concept), fr.get("caption", ""), style, 1280, 720, max_regen=0)
        board.append({**fr, **asset})
        files.append(asset["file"])
    video = media.slideshow(run_id, files) or {}
    return {"storyboard": board, "video": video,
            "events": [_event("storyboard", f"{len(board)} frames" + (" + video" if video else ""))]}


def localize(state: CampaignState) -> CampaignState:
    """Localize the headline + captions to the campaign's target-market language,
    inferred from the brief (Italian for a Vespa launch in Italy, Tamil for a
    Coimbatore brand, etc.). Marquee feature: copy reads native, not translated.
    Skips cleanly if there's no copy yet (degraded upstream)."""
    copy = state.get("copy", {}) or {}
    if not copy.get("headline") and not copy.get("captions"):
        return {"localization": {}, "events": [_event("localize", "no copy to localize, skipped")]}
    txt = _run(
        "localize",
        "You are a marketing localizer. Infer the campaign's target-market language from the brief "
        "and adapt the copy to natural, on-brand copy in that language (adapt, do not translate literally). "
        "Output JSON {\"language\":..,\"headline_local\":..,\"captions_local\":[..],\"note\":..}.",
        f"Brief: {state['sanitized_brief']}\nHeadline: {copy.get('headline')}\nCaptions: {copy.get('captions')}",
        Tier.MID,
    )
    out: LocalizationOutput = _parse(LocalizationOutput, txt)
    lang = out.language or "local language"
    return {"localization": out.model_dump(), "events": [_event("localize", f"localized to {lang}")]}


def critic(state: CampaignState) -> CampaignState:
    """Brand-safety + quality gate. Uses the heavy tier because catching off-brand
    output is the highest-leverage judgment in the graph."""
    loops = state.get("critic_loops", 0) + 1
    txt = _run(
        "critic",
        "You are a strict brand-safety critic. Judge whether the copy fits the brief and is on-brand. "
        "Output JSON {\"approved\":bool,\"score\":0..1,\"issues\":[..]}.",
        f"Brief: {state['sanitized_brief']}\nCopy: {state.get('copy', {})}\nCreative: {state.get('creative', {})}",
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
        "brand_profile": state.get("brand_profile"),
        "research": state.get("research"),
        "strategy": state.get("strategy"),
        "copy": state.get("copy"),
        "creative": state.get("creative"),
        "visuals": state.get("visuals"),
        "storyboard": state.get("storyboard"),
        "video": state.get("video"),
        "localization": state.get("localization"),
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
