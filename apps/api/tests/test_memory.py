"""M2: RAG retrieval relevance, episodic recall, tool integration."""
from __future__ import annotations

from app.memory.vector_store import get_store
from app.memory import episodic
from app.tools.registry import BRAND_STORE, call_tool


def _seed_brand():
    store = get_store(BRAND_STORE)
    store.add("safety", "Safety first: genuine cylinders, timely inspections, correct handling.", {"source": "brand"})
    store.add("price", "Aggressive discounting and coupon codes for cheap deals.", {"source": "brand"})
    return store


def test_rag_retrieval_is_relevant():
    _seed_brand()
    res = call_tool("rag_retrieve", query="how do we message safety to households", k=1)
    assert res["chunks"], "should retrieve at least one chunk"
    assert "safety" in res["chunks"][0]["text"].lower()


def test_web_search_tool_returns_results():
    res = call_tool("web_search", query="lpg refill trends", k=2)
    assert len(res["results"]) == 2
    assert res["results"][0]["url"].startswith("http")


def test_episodic_recall_finds_similar_run():
    episodic.record_episode(
        "run-1",
        "refill signup campaign for LPG households in Coimbatore",
        {"strategy": {"positioning": "trust", "channels": ["app"]}, "critique": {"score": 0.9}},
    )
    hits = episodic.recall_similar("LPG refill campaign Coimbatore households", k=1)
    assert hits and hits[0]["run_id"] == "run-1"


def test_image_brief_tool_expands_concept():
    res = call_tool("image_brief", concept="warm family kitchen safety", n=3)
    assert len(res["prompts"]) == 3
