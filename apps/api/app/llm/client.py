"""LLM access layer: model router + cost meter + mock fallback.

Every agent calls `complete()` — never the Anthropic SDK directly. That gives us
one place to (a) pick the cheapest capable model per step [cost pillar], (b) meter
tokens + cost [observability/cost pillar], and (c) swap in a deterministic mock so
the whole graph runs offline in tests [testing pillar].
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.config import settings


class Tier(str, Enum):
    CHEAP = "cheap"   # routing, classification, short structured extraction
    MID = "mid"       # most drafting/analysis
    HEAVY = "heavy"   # hard reasoning, final synthesis, adversarial critique


# Approx USD per 1M tokens (input, output). Update from the claude-api skill before
# quoting real numbers; these are placeholders for the cost meter's arithmetic.
_PRICING: dict[str, tuple[float, float]] = {
    settings.model_cheap: (1.0, 5.0),
    settings.model_mid: (3.0, 15.0),
    settings.model_heavy: (15.0, 75.0),
}

_TIER_MODEL: dict[Tier, str] = {
    Tier.CHEAP: settings.model_cheap,
    Tier.MID: settings.model_mid,
    Tier.HEAVY: settings.model_heavy,
}


@dataclass
class Usage:
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class CostMeter:
    """Accumulates usage across a whole run, broken down per agent."""

    total: Usage = field(default_factory=Usage)
    by_agent: dict[str, Usage] = field(default_factory=dict)

    def record(self, agent: str, u: Usage) -> None:
        self.total.input_tokens += u.input_tokens
        self.total.output_tokens += u.output_tokens
        self.total.cost_usd += u.cost_usd
        acc = self.by_agent.setdefault(agent, Usage(model=u.model))
        acc.input_tokens += u.input_tokens
        acc.output_tokens += u.output_tokens
        acc.cost_usd += u.cost_usd

    def snapshot(self) -> dict[str, Any]:
        return {
            "total": vars(self.total),
            "by_agent": {k: vars(v) for k, v in self.by_agent.items()},
        }


@dataclass
class Completion:
    text: str
    usage: Usage
    cached: bool = False


# Process-local prompt cache. On a cache hit we return zero-cost usage, which is what
# lets a repeated run show a dramatically lower cost in the dashboard [cost pillar].
_CACHE: dict[tuple[str, str, str], str] = {}
CACHE_ENABLED = True


def _price(model: str, in_tok: int, out_tok: int) -> float:
    pin, pout = _PRICING.get(model, (0.0, 0.0))
    return (in_tok / 1_000_000) * pin + (out_tok / 1_000_000) * pout


import re as _re


def _mock_text(system: str, prompt: str) -> str:
    """Deterministic stand-in. Reads the JSON key hints in the agent's system prompt
    and emits a matching object with believable placeholder values, so offline/demo
    runs produce populated packages while still exercising the real parse + schema
    validation paths. Keys followed by `[` become lists; `approved`->bool; `score`->num."""
    blob = system + prompt
    if "json" not in blob.lower():
        head = prompt.strip().splitlines()[0] if prompt.strip() else "brief"
        return f"[mock] response to: {head[:160]}"

    # find patterns like  "headline":  or  "captions":[
    fields = _re.findall(r'"(\w+)"\s*:\s*(\[?)', system)
    if not fields:
        return json.dumps({"summary": "[mock] structured output", "score": 0.9})

    obj: dict[str, Any] = {}
    for key, is_list in fields:
        if key == "findings":
            obj[key] = [
                {"claim": "[mock] Coimbatore households value safety + reliability.", "source": "brand-doc:sai-energy"},
                {"claim": "[mock] App-based refill booking is under-adopted locally.", "source": "web:market-scan"},
            ]
        elif key in ("claim", "source"):
            continue  # already emitted inside findings objects
        elif key == "approved":
            obj[key] = True
        elif key == "score":
            obj[key] = 0.9
        elif key in ("claim",):
            obj[key] = "[mock] Local trust and safety drive refill signups."
        elif key in ("source",):
            obj[key] = "brand-doc:sai-energy"
        elif is_list == "[":
            obj[key] = [f"[mock] {key} 1", f"[mock] {key} 2"]
        else:
            obj[key] = f"[mock] {key}"
    return json.dumps(obj)


def complete(
    system: str,
    prompt: str,
    *,
    tier: Tier = Tier.MID,
    max_tokens: int = 1024,
    force_model: str | None = None,
) -> Completion:
    model = force_model or _TIER_MODEL[tier]

    key = (model, system, prompt)
    if CACHE_ENABLED and key in _CACHE:
        # cache hit: no tokens billed
        return Completion(_CACHE[key], Usage(model, 0, 0, 0.0), cached=True)

    if not settings.use_real_llm:
        text = _mock_text(system, prompt)
        # rough token estimate so the cost meter has non-zero, testable numbers
        in_tok = (len(system) + len(prompt)) // 4
        out_tok = len(text) // 4
        if CACHE_ENABLED:
            _CACHE[key] = text
        return Completion(text, Usage(model, in_tok, out_tok, _price(model, in_tok, out_tok)))

    # Real path. Imported lazily so the package works without anthropic installed
    # in offline/mock environments.
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    in_tok = resp.usage.input_tokens
    out_tok = resp.usage.output_tokens
    if CACHE_ENABLED:
        _CACHE[key] = text
    return Completion(text, Usage(model, in_tok, out_tok, _price(model, in_tok, out_tok)))
