"""Tool integrations used by the agents.

Each tool has a stable signature and returns structured data. Offline/mock
implementations return deterministic results; production swaps the body (Tavily for
web_search, a real image model for image_brief, a channel API for publish) without
touching callers. A tiny registry + `call_tool` gives one metered, traceable choke
point for every tool invocation [observability + reliable-tool-integration pillars]."""
from __future__ import annotations

from typing import Any, Callable

from app.core.config import settings
from app.memory.vector_store import get_store

BRAND_STORE = "brand"


def rag_retrieve(query: str, k: int = 3) -> dict[str, Any]:
    """Retrieve brand-doc chunks relevant to the query for grounded, cited research."""
    hits = get_store(BRAND_STORE).search(query, k=k)
    return {
        "chunks": [
            {"id": d.id, "text": d.text, "source": d.metadata.get("source", d.id), "score": round(s, 3)}
            for d, s in hits
        ]
    }


def web_search(query: str, k: int = 3) -> dict[str, Any]:
    """Web search. Mock returns canned, plausible market signals; production calls Tavily."""
    if settings.llm_provider == "anthropic" and settings.anthropic_api_key and False:
        # production: wire Tavily/Brave here
        raise NotImplementedError
    canned = [
        {"title": "LPG refill booking trends 2026", "url": "https://example.com/lpg-trends",
         "snippet": "App-based cylinder booking adoption rising in tier-2 Indian cities."},
        {"title": "Local trust drives utility signups", "url": "https://example.com/trust",
         "snippet": "Safety messaging and local presence lift conversion for essential services."},
    ]
    return {"results": canned[:k], "query": query, "source": "mock"}


def image_brief(concept: str, n: int = 2) -> dict[str, Any]:
    """Turn a creative concept into structured image-generation prompts (no pixels)."""
    return {
        "prompts": [
            f"{concept} — hero shot, warm domestic kitchen, safety-forward, {i+1}"
            for i in range(n)
        ]
    }


def publish(package: dict[str, Any], channel: str = "mock") -> dict[str, Any]:
    """Mock publish: safe stand-in for pushing a campaign to a real channel."""
    return {"published": True, "channel": channel, "items": len(package or {})}


TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "rag_retrieve": rag_retrieve,
    "web_search": web_search,
    "image_brief": image_brief,
    "publish": publish,
}


def call_tool(name: str, **kwargs) -> dict[str, Any]:
    if name not in TOOLS:
        raise KeyError(f"unknown tool: {name}")
    return TOOLS[name](**kwargs)
