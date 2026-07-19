"""Episodic memory: a searchable log of past campaign runs.

After each run the packager writes a compact episode (brief + key decisions +
critique score). The planner retrieves similar past episodes so the system learns
from prior campaigns instead of starting cold every time."""
from __future__ import annotations

from typing import Any

from app.memory.vector_store import get_store

_STORE = "episodic"


def record_episode(run_id: str, brief: str, package: dict[str, Any]) -> None:
    crit = (package or {}).get("critique", {})
    strat = (package or {}).get("strategy", {})
    summary = (
        f"Brief: {brief}\n"
        f"Positioning: {strat.get('positioning', '')}\n"
        f"Channels: {', '.join(strat.get('channels', []) or [])}\n"
        f"Critique score: {crit.get('score')}"
    )
    get_store(_STORE).add(
        run_id, summary, {"run_id": run_id, "score": crit.get("score")}
    )


def recall_similar(brief: str, k: int = 2) -> list[dict[str, Any]]:
    hits = get_store(_STORE).search(brief, k=k)
    return [
        {"run_id": d.metadata.get("run_id"), "summary": d.text, "similarity": round(score, 3)}
        for d, score in hits
        if score > 0.01
    ]
