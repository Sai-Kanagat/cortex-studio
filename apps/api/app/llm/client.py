"""LLM access layer: model router + cost meter + mock fallback.

Every agent calls `complete()` — never the Anthropic SDK directly. That gives us
one place to (a) pick the cheapest capable model per step [cost pillar], (b) meter
tokens + cost [observability/cost pillar], and (c) swap in a deterministic mock so
the whole graph runs offline in tests [testing pillar].
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.config import settings


class Tier(str, Enum):
    CHEAP = "cheap"   # routing, classification, short structured extraction
    MID = "mid"       # most drafting/analysis
    HEAVY = "heavy"   # hard reasoning, final synthesis, adversarial critique


# Approx USD per 1M tokens (input, output). Reference/paid-tier prices so the cost
# meter is illustrative even on a provider's free tier (where real spend is ~$0).
_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-4-8": (15.0, 75.0),
    # Gemini (paid reference; free tier actual cost = $0)
    "gemini-flash-lite-latest": (0.10, 0.40),
    "gemini-flash-latest": (0.30, 2.50),
    "gemini-3.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.0),
    # Mock tiers (illustrative, so the cost meter is non-zero offline)
    "mock-cheap": (1.0, 5.0),
    "mock-mid": (3.0, 15.0),
    "mock-heavy": (15.0, 75.0),
}


def _tier_model(tier: Tier) -> str:
    return {
        Tier.CHEAP: settings.tier_cheap,
        Tier.MID: settings.tier_mid,
        Tier.HEAVY: settings.tier_heavy,
    }[tier]


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
    model = force_model or _tier_model(tier)

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

    # Real path. Providers are imported lazily so the package runs in mock mode with
    # neither SDK installed. Wrapped in backoff-retry so free-tier rate limits (e.g.
    # Gemini's 5 req/min/model) self-heal instead of failing the whole run.
    fn = _gemini if settings.llm_provider == "gemini" else _anthropic
    text, in_tok, out_tok = _with_retry(fn, system, prompt, model, max_tokens)

    if CACHE_ENABLED:
        _CACHE[key] = text
    return Completion(text, Usage(model, in_tok, out_tok, _price(model, in_tok, out_tok)))


def _with_retry(fn, system, prompt, model, max_tokens, attempts: int = 5):
    import time

    delay = 4.0
    for i in range(attempts):
        try:
            return fn(system, prompt, model, max_tokens)
        except Exception as e:
            msg = str(e)
            transient = "429" in msg or "RESOURCE_EXHAUSTED" in msg or "overloaded" in msg or "503" in msg
            if not transient or i == attempts - 1:
                raise
            # honor the API's suggested retry delay when present
            m = re.search(r"retryDelay['\"]?:?\s*['\"]?(\d+)", msg)
            wait = float(m.group(1)) + 1 if m else delay
            time.sleep(min(wait, 30))
            delay *= 2


def _anthropic(system: str, prompt: str, model: str, max_tokens: int) -> tuple[str, int, int]:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def _gemini(system: str, prompt: str, model: str, max_tokens: int) -> tuple[str, int, int]:
    """Google Gemini via the google-genai SDK. Free API key: aistudio.google.com.
    We request JSON mime-type so the agents' structured-output parsing stays reliable."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)
    wants_json = "json" in (system + prompt).lower()
    cfg = dict(
        system_instruction=system,
        # Gemini 2.5/3.x spend "thinking" tokens from the output budget; disable it so
        # short structured replies aren't truncated before the JSON is emitted, and give
        # generous headroom.
        max_output_tokens=max(max_tokens, 2048),
        response_mime_type="application/json" if wants_json else "text/plain",
    )
    try:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        pass  # older SDK without thinking_config
    resp = client.models.generate_content(
        model=model, contents=prompt, config=types.GenerateContentConfig(**cfg)
    )
    text = resp.text or ""
    um = getattr(resp, "usage_metadata", None)
    in_tok = getattr(um, "prompt_token_count", 0) or 0
    out_tok = getattr(um, "candidates_token_count", 0) or 0
    return text, in_tok, out_tok
