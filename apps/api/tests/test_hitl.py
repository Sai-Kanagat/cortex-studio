"""M6: human-in-the-loop interrupt + resume."""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.graph import agents
from app.graph.build import build_graph

BRIEF = "Refill-signup campaign for Sai Energy LPG, Coimbatore."


def _graph():
    return build_graph(checkpointer=MemorySaver())


def test_run_pauses_at_human_gate():
    agents.reset_run("hitl-1")
    g = _graph()
    cfg = {"configurable": {"thread_id": "hitl-1"}}
    res = g.invoke({"run_id": "hitl-1", "brief": BRIEF, "hitl_enabled": True}, cfg)
    # graph should be paused at the human gate, not yet packaged
    assert g.get_state(cfg).next == ("human_gate",)
    assert "package" not in res


def test_resume_approve_reaches_packager():
    agents.reset_run("hitl-2")
    g = _graph()
    cfg = {"configurable": {"thread_id": "hitl-2"}}
    g.invoke({"run_id": "hitl-2", "brief": BRIEF, "hitl_enabled": True}, cfg)
    final = g.invoke(Command(resume={"action": "approve"}), cfg)
    assert final["package"]["status"] == "approved"
    assert final["hitl_decision"] == "approve"


def test_resume_edit_applies_overrides():
    agents.reset_run("hitl-3")
    g = _graph()
    cfg = {"configurable": {"thread_id": "hitl-3"}}
    g.invoke({"run_id": "hitl-3", "brief": BRIEF, "hitl_enabled": True}, cfg)
    final = g.invoke(Command(resume={"action": "edit", "edits": {"headline": "Human-approved headline"}}), cfg)
    assert final["package"]["copy"]["headline"] == "Human-approved headline"


def test_batch_run_auto_approves_without_hitl():
    agents.reset_run("hitl-4")
    g = _graph()
    final = g.invoke({"run_id": "hitl-4", "brief": BRIEF}, {"configurable": {"thread_id": "hitl-4"}})
    assert final["package"]["status"] == "approved"
