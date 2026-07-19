"""M3: output moderation, publish-gating, rate limiting."""
from __future__ import annotations

from app.core.moderation import moderate_copy, moderate_text
from app.core.ratelimit import TokenBucket
from app.graph import agents


def test_moderation_flags_banned_content():
    bad = {"headline": "100% safe, guaranteed safe cylinders", "captions": ["their cylinders are unsafe"]}
    violations = moderate_copy(bad)
    cats = {v["category"] for v in violations}
    assert "unsafe_claims" in cats
    assert "competitor_attack" in cats


def test_clean_copy_passes_moderation():
    good = {"headline": "Reliable LPG refills, booked in the app", "captions": ["Trusted local service"]}
    assert moderate_copy(good) == []


def test_critic_blocks_violating_copy(monkeypatch):
    # Force the copywriter to emit banned content, verify the package is blocked & unpublished.
    def bad_copy(state):
        return {"copy": {"headline": "no risk, guaranteed safe gas", "captions": []},
                "events": []}
    monkeypatch.setattr(agents, "copywriter", bad_copy)
    from app.graph.build import build_graph
    agents.reset_run("t")
    state = build_graph().invoke({"run_id": "t", "brief": "safe LPG campaign"})
    assert state["package"]["status"] == "blocked"
    assert state["critique"]["violations"], "violations must be recorded"


def test_rate_limiter_enforces_capacity():
    b = TokenBucket(rate=0.0, capacity=2)  # no refill
    assert b.allow("k") and b.allow("k")   # burst of 2
    assert not b.allow("k")                # third denied


def test_moderate_text_direct():
    assert moderate_text("this will kill your family") != []
    assert moderate_text("safe, reliable, trusted") == []
