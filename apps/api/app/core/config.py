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
    # provider = "anthropic" for real calls, "mock" for deterministic offline runs.
    llm_provider: str = _get("LLM_PROVIDER", "mock")
    anthropic_api_key: str = _get("ANTHROPIC_API_KEY", "")

    # Model tiers used by the router. Cheap -> expensive.
    model_cheap: str = _get("MODEL_CHEAP", "claude-haiku-4-5-20251001")
    model_mid: str = _get("MODEL_MID", "claude-sonnet-5")
    model_heavy: str = _get("MODEL_HEAVY", "claude-opus-4-8")

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

    @property
    def use_real_llm(self) -> bool:
        return self.llm_provider == "anthropic" and bool(self.anthropic_api_key)


settings = Settings()
