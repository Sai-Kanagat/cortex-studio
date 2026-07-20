"""Shared typed state that flows through the LangGraph agent graph.

Using a TypedDict keeps it LangGraph-native (partial updates merge automatically)
while the Pydantic models below give each agent a validated output contract
[guardrails pillar]."""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field


# --- Per-agent structured outputs (guardrail schemas) ---

class ResearchFinding(BaseModel):
    claim: str
    source: str = Field(default="", description="URL or brand-doc id backing the claim")


class ResearchOutput(BaseModel):
    findings: list[ResearchFinding] = Field(default_factory=list)
    summary: str = ""


class StrategyOutput(BaseModel):
    segments: list[str] = Field(default_factory=list)
    positioning: str = ""
    channels: list[str] = Field(default_factory=list)


class CopyOutput(BaseModel):
    headline: str = ""
    captions: list[str] = Field(default_factory=list)
    email_subject: str = ""
    email_body: str = ""


class CreativeOutput(BaseModel):
    concept: str = ""
    image_prompts: list[str] = Field(default_factory=list)


class LocalizationOutput(BaseModel):
    # Coimbatore is Tamil-speaking; localizing lifts trust for a local LPG brand.
    headline_ta: str = ""
    captions_ta: list[str] = Field(default_factory=list)
    note: str = ""


class CritiqueOutput(BaseModel):
    approved: bool = False
    score: float = 0.0  # 0..1 brand+quality score
    issues: list[str] = Field(default_factory=list)


def _merge_events(left: list, right: list) -> list:
    return (left or []) + (right or [])


class CampaignState(TypedDict, total=False):
    # inputs
    run_id: str
    brief: str
    sanitized_brief: str

    # agent outputs
    plan: list[str]
    research: dict[str, Any]
    strategy: dict[str, Any]
    copy: dict[str, Any]
    creative: dict[str, Any]
    localization: dict[str, Any]
    critique: dict[str, Any]

    recalled_episodes: list[dict[str, Any]]

    # control
    hitl_enabled: bool
    critic_loops: int
    hitl_decision: str  # "approve" | "edit" | "reject" (set by human, M6)
    package: dict[str, Any]

    # cross-cutting
    events: Annotated[list[dict[str, Any]], _merge_events]  # streamed trace events
    cost: dict[str, Any]
    trace: dict[str, Any]
    guardrail_flags: Annotated[list[str], _merge_events]
