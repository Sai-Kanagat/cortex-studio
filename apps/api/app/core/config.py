"""Central configuration. All secrets/knobs come from the environment.

Kept dependency-light on purpose so the settings object can be imported anywhere
(agents, routes, tests) without side effects.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # --- LLM ---
    # provider = "anthropic" | "gemini" for real calls, "mock" for deterministic offline runs.
    llm_provider: str = _get("LLM_PROVIDER", "mock")
    anthropic_api_key: str = _get("ANTHROPIC_API_KEY", "")
    gemini_api_key: str = _get("GEMINI_API_KEY", "")

    # Model tiers used by the router. Cheap -> expensive. Defaults track the active
    # provider so switching provider doesn't require re-setting three model envs.
    model_cheap: str = _get("MODEL_CHEAP", "")
    model_mid: str = _get("MODEL_MID", "")
    model_heavy: str = _get("MODEL_HEAVY", "")

    # --- Infra ---
    database_url: str = _get(
        "DATABASE_URL", "postgresql://cortex:cortex@localhost:5432/cortex"
    )
    redis_url: str = _get("REDIS_URL", "redis://localhost:6379/0")

    # --- Observability (Langfuse) ---
    langfuse_public_key: str = _get("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = _get("LANGFUSE_SECRET_KEY", "")
    langfuse_host: str = _get("LANGFUSE_HOST", "http://localhost:3001")

    # --- Security ---
    api_key: str = _get("CORTEX_API_KEY", "")  # empty => auth disabled (local dev)

    # --- Agent behaviour ---
    max_critic_loops: int = int(_get("MAX_CRITIC_LOOPS", "2"))

    # Per-provider default model tiers (cheap, mid, heavy).
    _DEFAULT_TIERS = {
        "anthropic": ("claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-opus-4-8"),
        "gemini": ("gemini-flash-lite-latest", "gemini-flash-latest", "gemini-3.5-flash"),
        "mock": ("mock-cheap", "mock-mid", "mock-heavy"),
    }

    def _tier(self, i: int) -> str:
        override = (self.model_cheap, self.model_mid, self.model_heavy)[i]
        if override:
            return override
        return self._DEFAULT_TIERS.get(self.llm_provider, self._DEFAULT_TIERS["mock"])[i]

    @property
    def tier_cheap(self) -> str:
        return self._tier(0)

    @property
    def tier_mid(self) -> str:
        return self._tier(1)

    @property
    def tier_heavy(self) -> str:
        return self._tier(2)

    @property
    def use_real_llm(self) -> bool:
        if self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key)
        if self.llm_provider == "gemini":
            return bool(self.gemini_api_key)
        return False


settings = Settings()
