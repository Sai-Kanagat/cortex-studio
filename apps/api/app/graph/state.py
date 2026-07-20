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
    # Localizes to the campaign's target-market language (Italian for a Vespa launch,
    # Tamil for a Coimbatore brand, etc.) so copy reads native, not translated.
    language: str = ""
    headline_local: str = ""
    captions_local: list[str] = Field(default_factory=list)
    note: str = ""


class BrandProfile(BaseModel):
    """Extracted from an uploaded brand book (or a text brand doc)."""
    name: str = ""
    palette: list[str] = Field(default_factory=list)   # hex colours
    typography: str = ""
    voice: str = ""
    dos: list[str] = Field(default_factory=list)
    donts: list[str] = Field(default_factory=list)
    summary: str = ""


class VisualAsset(BaseModel):
    """A generated image asset + its self-QA verdict."""
    kind: str = ""            # hero | poster | carousel | storyboard_frame
    prompt: str = ""
    caption: str = ""
    url: str = ""             # served path, /assets/<run>/<file>
    file: str = ""
    mime: str = ""
    qa_score: float = 0.0
    qa_issues: list[str] = Field(default_factory=list)
    regenerated: int = 0


class StoryboardFrame(BaseModel):
    scene: int = 0
    shot: str = ""            # shot description
    caption: str = ""
    prompt: str = ""


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
    upload_path: str          # optional uploaded brand book (pdf/image)
    brand_profile: dict[str, Any]

    # agent outputs
    plan: list[str]
    research: dict[str, Any]
    strategy: dict[str, Any]
    copy: dict[str, Any]
    creative: dict[str, Any]
    localization: dict[str, Any]
    visuals: list[dict[str, Any]]        # hero, poster, carousel VisualAssets
    storyboard: list[dict[str, Any]]     # StoryboardFrame + generated asset
    video: dict[str, Any]                # slideshow mp4 descriptor (optional)
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
