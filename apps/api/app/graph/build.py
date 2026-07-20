r"""Assemble the LangGraph state machine.

Wiring (M1):

    ingest -> planner -> research -> strategy -> copywriter -> creative_director
        -> critic --(approved / cap reached)--> packager -> END
                 \--(revise)--> copywriter   (loop, bounded by max_critic_loops)

The checkpointer persists thread state so a run can be killed and resumed, and so
the HITL interrupt in M6 can suspend before `packager` and resume on human input.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph import agents
from app.graph.state import CampaignState


def build_graph(checkpointer=None):
    g = StateGraph(CampaignState)

    # Every node is wrapped with agents.resilient(name): a raised exception is recorded
    # and the graph continues (degraded) instead of crashing the run. Node names are
    # suffixed "_agent" where they would collide with a state key (LangGraph forbids that).
    def node(name: str, fn):
        g.add_node(name, agents.resilient(name)(fn))

    node("ingest", agents.ingest)
    node("planner", agents.planner)
    node("research_agent", agents.research)
    node("strategy_agent", agents.strategy)
    node("copywriter", agents.copywriter)
    node("creative_director", agents.creative_director)
    node("localize", agents.localize)
    node("critic", agents.critic)
    # human_gate is NOT wrapped: interrupt() raises GraphInterrupt, which resilient()
    # would swallow and break the HITL pause/resume.
    g.add_node("human_gate", agents.human_gate)
    node("packager", agents.packager)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "planner")
    g.add_edge("planner", "research_agent")
    g.add_edge("research_agent", "strategy_agent")
    g.add_edge("strategy_agent", "copywriter")
    g.add_edge("copywriter", "creative_director")
    g.add_edge("creative_director", "localize")
    g.add_edge("localize", "critic")

    # critic -> (approved/cap) human_gate -> (approve/edit) packager | (reject) copywriter
    g.add_conditional_edges(
        "critic",
        agents.route_after_critic,
        {"revise": "copywriter", "package": "human_gate"},
    )
    g.add_conditional_edges(
        "human_gate",
        agents.route_after_gate,
        {"revise": "copywriter", "package": "packager"},
    )
    g.add_edge("packager", END)

    return g.compile(checkpointer=checkpointer)
